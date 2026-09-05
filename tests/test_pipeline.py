"""
Tests for the run_pipeline.py orchestration script.

Verifies that the pipeline script can be imported, the argument parser is
configured correctly, and the execution-flow helpers work as expected.
No actual model or dataset is required — these are unit tests for the
orchestration logic only.
"""

import importlib
import sys
import types
from pathlib import Path
import pytest

# Make sure the repo root is importable
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helper: load run_pipeline as a module without executing main()
# ---------------------------------------------------------------------------

def _load_pipeline_module():
    pipeline_path = REPO_ROOT / "scripts" / "run_pipeline.py"
    if not pipeline_path.exists():
        pytest.fail(f"run_pipeline.py not found at {pipeline_path}")
    spec = importlib.util.spec_from_file_location("run_pipeline", pipeline_path)
    if spec is None or spec.loader is None:
        pytest.fail(f"Could not create module spec for {pipeline_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_module_imports():
    """run_pipeline.py must be importable without errors."""
    mod = _load_pipeline_module()
    assert mod is not None


def test_build_parser_returns_parser():
    """build_parser() must return an ArgumentParser with the expected arguments."""
    mod = _load_pipeline_module()
    parser = mod.build_parser()

    # Check required arguments are registered
    option_strings = {
        action.dest for action in parser._actions
    }
    for expected in ("config", "model", "data", "output", "variant",
                     "layer", "steps", "batch_size", "device"):
        assert expected in option_strings, f"Missing argument: {expected}"


def test_default_steps_are_all_three():
    """The default value for --steps should be [1, 2, 3]."""
    mod = _load_pipeline_module()
    parser = mod.build_parser()
    args = parser.parse_args([])
    assert sorted(args.steps) == [1, 2, 3]


def test_steps_subset():
    """--steps should accept a subset of pipeline steps."""
    mod = _load_pipeline_module()
    parser = mod.build_parser()
    args = parser.parse_args(["--steps", "1", "2"])
    assert sorted(args.steps) == [1, 2]


def test_invalid_step_rejected():
    """--steps should reject values outside [1, 2, 3]."""
    mod = _load_pipeline_module()
    parser = mod.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--steps", "4"])


def test_variant_choices():
    """--variant must accept A, B, C, D and reject anything else."""
    mod = _load_pipeline_module()
    parser = mod.build_parser()

    for v in ("A", "B", "C", "D"):
        args = parser.parse_args(["--variant", v])
        assert args.variant == v

    with pytest.raises(SystemExit):
        parser.parse_args(["--variant", "Z"])


def test_default_output():
    """The default output directory should be 'results/pipeline'."""
    mod = _load_pipeline_module()
    parser = mod.build_parser()
    args = parser.parse_args([])
    assert args.output == "results/pipeline"


def test_helper_functions_exist():
    """Key helper functions must be present in the module."""
    mod = _load_pipeline_module()
    for name in ("run_step1_crp_analysis", "run_step2_discover_concepts",
                 "run_step3_evaluate_concepts", "main"):
        assert hasattr(mod, name), f"Missing function: {name}"


if __name__ == "__main__":
    print("Running pipeline orchestration tests...\n")

    test_module_imports()
    test_build_parser_returns_parser()
    test_default_steps_are_all_three()
    test_steps_subset()
    test_invalid_step_rejected()
    test_variant_choices()
    test_default_output()
    test_helper_functions_exist()

    print("\n✓ All pipeline tests passed!")
