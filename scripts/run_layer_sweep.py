#!/usr/bin/env python3
"""
Layer-wise summary for Variant C concept discovery.

The script can either summarize existing ``discover_concepts.py`` outputs or run
concept discovery for each requested layer first.  It focuses on lightweight
geometry/coverage metrics that are useful for the paper's layer-dependence and
concept-collapse analysis.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
import os
import pickle
import subprocess
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LAYERS = ["conv1", "conv2", "conv3", "conv4"]


def _effective_rank(features: np.ndarray) -> float:
    """Participation-ratio effective rank of the covariance eigenvalues."""
    features = np.asarray(features, dtype=np.float64)
    if features.shape[0] < 2:
        return 0.0
    cov = np.cov(features, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    denom = float(np.sum(eigvals ** 2))
    if denom <= 1e-12:
        return 0.0
    return float((np.sum(eigvals) ** 2) / denom)


def _cov_trace(features: np.ndarray) -> float:
    features = np.asarray(features, dtype=np.float64)
    if features.shape[0] < 2:
        return 0.0
    return float(np.trace(np.cov(features, rowvar=False)))


def _entropy(assignments: np.ndarray, n_components: int) -> float:
    if len(assignments) == 0 or n_components <= 0:
        return 0.0
    counts = np.bincount(assignments.astype(int), minlength=n_components).astype(np.float64)
    probs = counts / (counts.sum() + 1e-12)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def _mean_pairwise_distance(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    dists = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dists.append(np.linalg.norm(points[i] - points[j]))
    return float(np.mean(dists)) if dists else 0.0


def _summarize_layer(layer: str, concept_dir: str) -> Dict[str, float]:
    results_path = os.path.join(concept_dir, "results.pkl")
    tcd_path = os.path.join(concept_dir, "tcd_model.pkl")
    if not os.path.exists(results_path) or not os.path.exists(tcd_path):
        raise FileNotFoundError(
            f"Missing results.pkl or tcd_model.pkl for layer {layer} in {concept_dir}"
        )

    with open(results_path, "rb") as f:
        results = pickle.load(f)
    with open(tcd_path, "rb") as f:
        tcd = pickle.load(f)

    features = np.asarray(results["features"], dtype=np.float64)
    labels = np.asarray(results["labels"], dtype=int)
    gmms = getattr(tcd.prototype_discovery, "gmms", {})

    row: Dict[str, float] = {
        "layer": layer,
        "n_samples": int(len(features)),
        "n_features": int(features.shape[1]),
        "effective_rank_global": _effective_rank(features),
        "cov_trace_global": _cov_trace(features),
        "n_total_prototypes": int(sum(gmm.n_components for gmm in gmms.values())),
    }

    all_coverages: List[float] = []
    all_separations: List[float] = []
    entropies: List[float] = []

    for class_id in [0, 1]:
        class_mask = labels == class_id
        class_features = features[class_mask]
        row[f"effective_rank_class{class_id}"] = _effective_rank(class_features)
        row[f"cov_trace_class{class_id}"] = _cov_trace(class_features)

        if class_id not in gmms or len(class_features) == 0:
            row[f"n_prototypes_class{class_id}"] = 0
            row[f"assignment_entropy_class{class_id}"] = 0.0
            row[f"coverage_min_class{class_id}"] = 0.0
            row[f"coverage_max_class{class_id}"] = 0.0
            row[f"prototype_separation_class{class_id}"] = 0.0
            continue

        gmm = gmms[class_id]
        assignments = gmm.predict(class_features)
        counts = np.bincount(assignments, minlength=gmm.n_components).astype(np.float64)
        coverage = counts / (counts.sum() + 1e-12)
        entropy = _entropy(assignments, gmm.n_components)
        separation = _mean_pairwise_distance(gmm.means_)

        row[f"n_prototypes_class{class_id}"] = int(gmm.n_components)
        row[f"assignment_entropy_class{class_id}"] = entropy
        row[f"assignment_entropy_norm_class{class_id}"] = (
            entropy / np.log(gmm.n_components) if gmm.n_components > 1 else 0.0
        )
        row[f"coverage_min_class{class_id}"] = float(np.min(coverage))
        row[f"coverage_max_class{class_id}"] = float(np.max(coverage))
        row[f"prototype_separation_class{class_id}"] = separation

        all_coverages.extend(coverage.tolist())
        all_separations.append(separation)
        entropies.append(entropy)

    row["coverage_min_all"] = float(np.min(all_coverages)) if all_coverages else 0.0
    row["coverage_max_all"] = float(np.max(all_coverages)) if all_coverages else 0.0
    row["assignment_entropy_mean"] = float(np.mean(entropies)) if entropies else 0.0
    row["prototype_separation_mean"] = float(np.mean(all_separations)) if all_separations else 0.0
    return row


def _run_discovery(args, layer: str, output_dir: str) -> None:
    cmd = [
        sys.executable,
        "scripts/discover_concepts.py",
        "--config", args.config,
        "--variant", "C",
        "--features", args.features,
        "--output", output_dir,
        "--layer", layer,
    ]
    if args.data:
        cmd.extend(["--data", args.data])
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _plot_summary(rows: List[Dict[str, float]], output_dir: str) -> None:
    layers = [row["layer"] for row in rows]
    x = np.arange(len(layers))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(x, [row["effective_rank_global"] for row in rows], marker="o")
    axes[0].set_title("Effective rank")
    axes[0].set_ylabel("Participation rank")

    axes[1].plot(x, [row["assignment_entropy_mean"] for row in rows], marker="o")
    axes[1].set_title("Assignment entropy")

    axes[2].plot(x, [row["prototype_separation_mean"] for row in rows], marker="o")
    axes[2].set_title("Prototype separation")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=30)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    path = os.path.join(output_dir, "layer_sweep_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run/summarize a Variant C layer sweep")
    parser.add_argument('--config', default='configs/default.yaml', help='Config for discover_concepts.py')
    parser.add_argument('--features', required=True, help='CRP feature directory from run_analysis.py')
    parser.add_argument('--data', default=None, help='Optional data path forwarded to discover_concepts.py')
    parser.add_argument('--output-root', default='results/layer_sweep', help='Directory for per-layer outputs')
    parser.add_argument('--summary', default='results/layer_sweep_summary.csv', help='Summary CSV path')
    parser.add_argument('--layers', nargs='+', default=DEFAULT_LAYERS, help='Layers to evaluate')
    parser.add_argument('--run-discovery', action='store_true', help='Run discover_concepts.py for each layer')
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)

    rows: List[Dict[str, float]] = []
    for layer in args.layers:
        concept_dir = os.path.join(args.output_root, f"variantC_{layer}")
        if args.run_discovery:
            _run_discovery(args, layer, concept_dir)
        row = _summarize_layer(layer, concept_dir)
        rows.append(row)
        print(f"{layer}: effective_rank={row['effective_rank_global']:.2f}, "
              f"entropy={row['assignment_entropy_mean']:.2f}")

    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved layer sweep summary to {args.summary}")

    _plot_summary(rows, os.path.dirname(os.path.abspath(args.summary)))
    print("✓ Layer sweep complete")


if __name__ == "__main__":
    main()
