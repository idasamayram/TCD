#!/usr/bin/env python3
"""
Full TCD Pipeline Orchestrator.

Runs all three pipeline steps in sequence:
  Step 1 — CRP Analysis     (run_analysis.py):       collect per-layer concept features
  Step 2 — Concept Discovery (discover_concepts.py): apply TCD variant (A/B/C/D)
  Step 3 — Evaluation        (evaluate_concepts.py): faithfulness / stability / purity

Execution flow
--------------
Raw vibration signals  (3 × 2000 tensor per sample)
        │
        ▼  Step 1: TimeSeriesCondAttribution + ChannelConcept
CRP features           (eps_relevances_class_*.hdf5, heatmaps_class_*.hdf5)
        │
        ▼  Step 2: TCD variant (A=filterbank, B=descriptors, C=GMM clusters, D=vib. features)
Discovered concepts    (results.pkl, tcd_model.pkl)
        │
        ▼  Step 3: ConceptInterventionHook + evaluation metrics
Evaluation report      (evaluation.pkl, plots)

Usage
-----
# Run all three steps with default config
python scripts/run_pipeline.py --config configs/default.yaml \\
    --model path/to/model.ckpt --data path/to/data \\
    --variant C --output results/run1

# Run only steps 1 and 2 (skip evaluation)
python scripts/run_pipeline.py --config configs/default.yaml \\
    --model path/to/model.ckpt --data path/to/data \\
    --variant A --output results/run1 --steps 1 2

# Resume from step 2 (CRP features already computed)
python scripts/run_pipeline.py --config configs/default.yaml \\
    --model path/to/model.ckpt --data path/to/data \\
    --variant C --output results/run1 --steps 2 3
"""
import sys
import time
import argparse
import os
import subprocess
from pathlib import Path

# Ensure the repository root is on the Python path when called directly
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str, width: int = 70) -> None:
    """Print a clearly visible section header."""
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _step_banner(step: int, name: str, width: int = 70) -> None:
    """Print a banner that marks the start of a pipeline step."""
    _section(f"STEP {step}: {name}", width)


def _success(msg: str) -> None:
    print(f"✓ {msg}")


def _warn(msg: str) -> None:
    print(f"⚠ {msg}")


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def run_step1_crp_analysis(
    config: str,
    model: str,
    data: str,
    output: str,
    batch_size: int | None,
    device: str | None,
) -> bool:
    """
    Step 1 — CRP Analysis.

    Calls run_analysis.py as a subprocess so that argument handling, imports
    and CUDA context are isolated, exactly as a user would invoke each step.

    Returns True on success, False on failure.
    """
    _step_banner(1, "CRP Analysis — collect per-layer concept features")
    print("  Input : raw vibration signals from dataset")
    print("  Output: eps_relevances_class_*.hdf5  (concept relevance vectors)")
    print("          heatmaps_class_*.hdf5         (input-level attribution maps)")
    print("          outputs_class_*.pt            (model logit vectors)")
    print("          sample_ids_class_*.pt         (dataset indices)")
    print()

    script = Path(__file__).parent / "run_analysis.py"
    cmd = [sys.executable, str(script),
           "--config", config,
           "--model", model,
           "--data", data,
           "--output", output]
    if batch_size is not None:
        cmd += ["--batch-size", str(batch_size)]
    if device is not None:
        cmd += ["--device", device]

    print(f"  Command: {' '.join(cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode == 0:
        _success(f"Step 1 completed in {elapsed:.1f}s")
        return True
    else:
        _warn(f"Step 1 failed (return code {result.returncode})")
        return False


def run_step2_discover_concepts(
    config: str,
    features: str,
    output: str,
    variant: str,
    layer: str,
    data: str | None,
    window_based: bool,
) -> bool:
    """
    Step 2 — Concept Discovery.

    Returns True on success, False on failure.
    """
    _step_banner(2, f"Concept Discovery — Variant {variant}")
    variant_desc = {
        "A": "Frequency-band filterbank (physics-informed bands: 0-10, 10-50, 50-100, 100-200 Hz)",
        "B": "Temporal descriptors (slope, peak, autocorrelation, spectral)",
        "C": "Learned GMM clusters in CRP concept space (PCX-style prototypes)",
        "D": "Comprehensive vibration feature extraction + automatic feature selection",
    }
    print(f"  Variant: {variant} — {variant_desc.get(variant, '')}")
    print(f"  Input : CRP features from Step 1 at {features}")
    print(f"  Output: results.pkl  (concept relevances, importance scores, labels)")
    print(f"          tcd_model.pkl (fitted TCD model for later use)")
    print()

    script = Path(__file__).parent / "discover_concepts.py"
    cmd = [sys.executable, str(script),
           "--config", config,
           "--variant", variant,
           "--features", features,
           "--output", output,
           "--layer", layer]
    if data is not None:
        cmd += ["--data", data]
    if window_based:
        cmd.append("--window-based")

    print(f"  Command: {' '.join(cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode == 0:
        _success(f"Step 2 completed in {elapsed:.1f}s")
        return True
    else:
        _warn(f"Step 2 failed (return code {result.returncode})")
        return False


def run_step3_evaluate_concepts(
    config: str,
    concepts: str,
    model: str,
    data: str,
    output: str,
    device: str | None,
) -> bool:
    """
    Step 3 — Evaluation.

    Returns True on success, False on failure.
    """
    _step_banner(3, "Evaluation — faithfulness, stability, purity")
    print("  Input : discovered concepts from Step 2, model, dataset")
    print("  Output: evaluation.pkl  (faithfulness / stability / purity metrics)")
    print("          deviation_matrix_class_*.png (deviation visualizations)")
    print()

    script = Path(__file__).parent / "evaluate_concepts.py"
    cmd = [sys.executable, str(script),
           "--config", config,
           "--concepts", concepts,
           "--model", model,
           "--data", data,
           "--output", output]
    if device is not None:
        cmd += ["--device", device]

    print(f"  Command: {' '.join(cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode == 0:
        _success(f"Step 3 completed in {elapsed:.1f}s")
        return True
    else:
        _warn(f"Step 3 failed (return code {result.returncode})")
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "TCD Full Pipeline — runs CRP analysis → concept discovery → evaluation "
            "in a single command, printing the complete execution flow."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file (default: configs/default.yaml)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint (overrides config)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to data directory (overrides config)")
    parser.add_argument("--output", type=str, default="results/pipeline",
                        help="Root output directory; sub-dirs are created per step")
    parser.add_argument("--variant", type=str, default=None, choices=["A", "B", "C", "D"],
                        help="TCD variant for Step 2 (overrides config tcd.variant)")
    parser.add_argument("--layer", type=str, default="conv3",
                        help="Layer for Variant C concept discovery (default: conv3)")
    parser.add_argument("--window-based", action="store_true",
                        help="Use window-based (data-driven) mode for Variant A")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3],
                        metavar="N", choices=[1, 2, 3],
                        help="Which steps to run, e.g. --steps 1 2  (default: 1 2 3)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for Step 1 (overrides config)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda or cpu (auto-detected if omitted)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Resolve config to get defaults for model/data/variant               #
    # ------------------------------------------------------------------ #
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML is required. Install it with:  pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    cfg: dict = {}
    if os.path.exists(args.config):
        with open(args.config) as fh:
            cfg = yaml.safe_load(fh) or {}
    else:
        _warn(f"Config file not found: {args.config}. Proceeding with CLI arguments only.")

    model_path = args.model or cfg.get("model", {}).get("path", "")
    data_path  = args.data  or cfg.get("data",  {}).get("path", "")
    variant    = args.variant or cfg.get("tcd",  {}).get("variant", "C")
    steps      = sorted(set(args.steps))

    # Validate required paths when Step 1 will run
    if 1 in steps:
        if not model_path:
            print("ERROR: --model is required for Step 1 (or set model.path in config).", file=sys.stderr)
            sys.exit(1)
        if not data_path:
            print("ERROR: --data is required for Step 1 (or set data.path in config).", file=sys.stderr)
            sys.exit(1)

    # ------------------------------------------------------------------ #
    # Print execution plan                                                 #
    # ------------------------------------------------------------------ #
    _section("TCD FULL PIPELINE — execution plan")
    print(f"  Config  : {args.config}")
    print(f"  Model   : {model_path}")
    print(f"  Data    : {data_path}")
    print(f"  Variant : {variant}")
    print(f"  Output  : {args.output}")
    print(f"  Steps   : {sorted(args.steps)}")
    print()
    print("  Pipeline flow:")
    print("    Raw signals (3 × 2000 @ 400 Hz)")
    print("         │")
    print("         ▼  Step 1 — TimeSeriesCondAttribution + ChannelConcept")
    print("    CRP features  (per-layer concept relevance vectors + heatmaps)")
    print("         │")
    print("         ▼  Step 2 — TCD Variant " + variant)
    print("    Concepts      (prototypes / frequency bands / descriptors)")
    print("         │")
    print("         ▼  Step 3 — ConceptInterventionHook + evaluation metrics")
    print("    Evaluation    (faithfulness, stability, purity, deviation plots)")

    # ------------------------------------------------------------------ #
    # Derive per-step output directories                                   #
    # ------------------------------------------------------------------ #
    crp_output      = os.path.join(args.output, "crp_features")
    concepts_output = os.path.join(args.output, f"concepts_{variant}")
    eval_output     = os.path.join(args.output, "evaluation")

    t_pipeline_start = time.time()
    failed = False

    # ------------------------------------------------------------------ #
    # Step 1                                                               #
    # ------------------------------------------------------------------ #
    if 1 in steps:
        ok = run_step1_crp_analysis(
            config=args.config,
            model=model_path,
            data=data_path,
            output=crp_output,
            batch_size=args.batch_size,
            device=args.device,
        )
        if not ok:
            _warn("Stopping pipeline after Step 1 failure.")
            sys.exit(1)

    # ------------------------------------------------------------------ #
    # Step 2                                                               #
    # ------------------------------------------------------------------ #
    if 2 in steps:
        ok = run_step2_discover_concepts(
            config=args.config,
            features=crp_output,
            output=concepts_output,
            variant=variant,
            layer=args.layer,
            data=data_path or None,
            window_based=args.window_based,
        )
        if not ok:
            _warn("Stopping pipeline after Step 2 failure.")
            sys.exit(1)

    # ------------------------------------------------------------------ #
    # Step 3                                                               #
    # ------------------------------------------------------------------ #
    if 3 in steps:
        ok = run_step3_evaluate_concepts(
            config=args.config,
            concepts=concepts_output,
            model=model_path,
            data=data_path,
            output=eval_output,
            device=args.device,
        )
        if not ok:
            failed = True  # evaluation failure is non-fatal for pipeline summary

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    elapsed_total = time.time() - t_pipeline_start
    _section("PIPELINE SUMMARY")
    print(f"  Total wall time : {elapsed_total:.1f}s")
    print(f"  Steps executed  : {steps}")
    print()
    print("  Output layout:")
    if 1 in steps:
        print(f"    {crp_output}/")
        print(f"      eps_relevances_class_{{0,1}}.hdf5  — concept relevance vectors")
        print(f"      heatmaps_class_{{0,1}}.hdf5         — input attribution maps")
        print(f"      outputs_class_{{0,1}}.pt             — model predictions")
        print(f"      sample_ids_class_{{0,1}}.pt          — dataset indices")
    if 2 in steps:
        print(f"    {concepts_output}/")
        print(f"      results.pkl    — concept relevances, importance, labels")
        print(f"      tcd_model.pkl  — fitted TCD model")
    if 3 in steps:
        print(f"    {eval_output}/")
        print(f"      evaluation.pkl — faithfulness / stability / purity metrics")

    print()
    if failed:
        _warn("Pipeline finished with errors in Step 3. Check output above.")
        sys.exit(1)
    else:
        _success("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
