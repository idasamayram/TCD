"""
Test new features added for TCD improvements.

Tests:
1. CNCValidatedComposite creation
2. VibrationFeatureTCD basic functionality
3. Class weights parameter passing
4. Config validation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import yaml


def test_cnc_validated_composite():
    """Test CNC-validated composite creation."""
    from tcd.composites import get_composite, CNCValidatedComposite
    
    # Test get_composite function
    composite = get_composite('cnc_validated')
    assert composite is not None, "Failed to create cnc_validated composite"
    
    # Test direct instantiation
    composite = CNCValidatedComposite()
    assert composite is not None, "Failed to create CNCValidatedComposite directly"
    assert hasattr(composite, 'layer_map'), "Composite missing layer_map"
    assert hasattr(composite, 'first_map'), "Composite missing first_map"
    
    print("✓ CNCValidatedComposite test passed")


def test_vibration_feature_tcd_basic():
    """Test VibrationFeatureTCD basic functionality."""
    from tcd.variants.vibration_features import VibrationFeatureTCD
    
    # Create synthetic data
    n_samples = 20
    n_channels = 3
    n_timesteps = 500
    
    heatmaps = torch.randn(n_samples, n_channels, n_timesteps)
    labels = torch.cat([torch.zeros(10), torch.ones(10)]).long()
    
    # Initialize
    tcd = VibrationFeatureTCD(
        sample_rate=400,
        window_size=50,
        n_concepts=10,
        use_feature_selection=True
    )
    
    # Fit
    tcd.fit(heatmaps, labels=labels)
    assert tcd.fitted, "VibrationFeatureTCD not marked as fitted"
    assert len(tcd.feature_names) > 0, "No features extracted"
    
    # Extract concepts
    concept_vectors = tcd.extract_concepts(heatmaps)
    assert concept_vectors.shape[0] == n_samples, "Wrong number of samples"
    assert concept_vectors.shape[1] <= 10, "Too many concepts extracted"
    
    # Get labels
    labels = tcd.get_concept_labels()
    assert len(labels) > 0, "No concept labels"
    
    # Compute importance
    importance = tcd.compute_concept_importance(heatmaps)
    assert len(importance) == len(labels), "Importance length mismatch"
    
    print("✓ VibrationFeatureTCD basic test passed")


def test_time_domain_features():
    """Test time-domain feature extraction."""
    from tcd.variants.vibration_features import VibrationFeatureTCD
    
    tcd = VibrationFeatureTCD(sample_rate=400)
    
    # Create test signal
    signal = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 400))
    
    features = tcd.extract_time_domain_features(signal)
    
    # Check expected features exist
    expected = ['rms', 'peak', 'crest_factor', 'kurtosis', 'skewness', 
                'zero_crossing_rate', 'std', 'variance']
    
    for feat in expected:
        assert feat in features, f"Missing feature: {feat}"
        assert not np.isnan(features[feat]), f"Feature {feat} is NaN"
    
    print("✓ Time-domain features test passed")


def test_frequency_domain_features():
    """Test frequency-domain feature extraction."""
    from tcd.variants.vibration_features import VibrationFeatureTCD
    
    tcd = VibrationFeatureTCD(sample_rate=400)
    
    # Create test signal with known frequency
    signal = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 400))
    
    features = tcd.extract_frequency_domain_features(signal)
    
    # Check expected features exist
    expected = ['spectral_centroid', 'spectral_entropy', 'dominant_frequency',
                'spectral_kurtosis', 'spectral_rolloff']
    
    for feat in expected:
        assert feat in features, f"Missing feature: {feat}"
        assert not np.isnan(features[feat]), f"Feature {feat} is NaN"
    
    # Band energy features
    assert 'band_0_10Hz' in features
    assert 'band_10_50Hz' in features
    
    print("✓ Frequency-domain features test passed")


def test_multi_axis_features():
    """Test multi-axis feature extraction."""
    from tcd.variants.vibration_features import VibrationFeatureTCD
    
    tcd = VibrationFeatureTCD(sample_rate=400)
    
    # Create 3-channel signal
    signals = np.random.randn(3, 400)
    
    features = tcd.extract_multi_axis_features(signals)
    
    # Check correlation features
    assert 'corr_xy' in features
    assert 'corr_xz' in features
    assert 'corr_yz' in features
    
    # Check energy ratios
    assert 'axis_0_energy_ratio' in features
    assert 'axis_1_energy_ratio' in features
    assert 'axis_2_energy_ratio' in features
    
    print("✓ Multi-axis features test passed")


def test_class_weights_parameter():
    """Test that class_weights parameter exists in relevant functions."""
    from tcd.prototypes import TemporalPrototypeDiscovery
    import inspect
    
    # Check TemporalPrototypeDiscovery.fit signature
    sig = inspect.signature(TemporalPrototypeDiscovery.fit)
    assert 'class_weights' in sig.parameters, "class_weights not in TemporalPrototypeDiscovery.fit"
    
    # Check LearnedClusterTCD.fit signature
    from tcd.variants.learned_clusters import LearnedClusterTCD
    sig = inspect.signature(LearnedClusterTCD.fit)
    assert 'class_weights' in sig.parameters, "class_weights not in LearnedClusterTCD.fit"
    
    print("✓ Class weights parameter test passed")


def test_config_validation():
    """Test configuration file has all expected fields."""
    config_path = 'configs/default.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Phase 1: use_class_weights
    assert 'use_class_weights' in config['analysis']
    assert config['analysis']['use_class_weights'] is True
    
    # Phase 3: cnc_validated composite
    assert config['analysis']['composite'] == 'cnc_validated'
    
    # Phase 4: vibration_features config
    assert 'vibration_features' in config['tcd']
    vf = config['tcd']['vibration_features']
    assert 'window_size' in vf
    assert 'n_concepts' in vf
    assert 'use_feature_selection' in vf
    assert 'selection_method' in vf
    
    print("✓ Config validation test passed")


def test_variant_d_in_discover_concepts():
    """Test that variant D is supported in discover_concepts.py."""
    import ast
    
    with open('scripts/discover_concepts.py', 'r') as f:
        tree = ast.parse(f.read())
    
    # Find run_variant_d function
    found_run_variant_d = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'run_variant_d':
            found_run_variant_d = True
            break
    
    assert found_run_variant_d, "run_variant_d function not found"
    
    # Check that 'D' is in variant choices
    with open('scripts/discover_concepts.py', 'r') as f:
        content = f.read()
        assert "choices=['A', 'B', 'C', 'D']" in content, "'D' not in variant choices"
    
    print("✓ Variant D support test passed")


def test_exports():
    """Test that new classes are properly exported."""
    # Test CNCValidatedComposite export
    from tcd import CNCValidatedComposite
    assert CNCValidatedComposite is not None
    
    # Test VibrationFeatureTCD export from variants
    from tcd.variants import VibrationFeatureTCD
    assert VibrationFeatureTCD is not None
    
    print("✓ Exports test passed")


if __name__ == "__main__":
    print("Running new feature tests...\n")
    
    try:
        test_cnc_validated_composite()
        test_vibration_feature_tcd_basic()
        test_time_domain_features()
        test_frequency_domain_features()
        test_multi_axis_features()
        test_class_weights_parameter()
        test_config_validation()
        test_variant_d_in_discover_concepts()
        test_exports()
        
        print("\n" + "="*60)
        print("✓ ALL NEW FEATURE TESTS PASSED!")
        print("="*60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
