"""
Test global window analysis and concept interpretation (Phase 5 improvements).

Verifies that the new global window analysis and concept interpreter
work correctly with CRP-native concepts.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest
from sklearn.mixture import GaussianMixture
from tcd.variants.global_concepts import GlobalWindowAnalysis
from tcd.interpretation import ConceptInterpreter
from tcd.prototypes import TemporalPrototypeDiscovery


def test_global_window_analysis_basic():
    """Test basic global window analysis functionality."""
    # Synthetic heatmaps: 100 samples, 3 channels, 2000 timesteps
    torch.manual_seed(42)
    heatmaps = torch.randn(100, 3, 2000) * 0.1
    
    # Add important region at timesteps 400-480
    heatmaps[:, :, 400:480] += torch.randn(100, 3, 80) * 2.0
    
    labels = torch.zeros(100).long()  # All same class for simplicity
    
    # Test global window analysis
    analyzer = GlobalWindowAnalysis(window_size=40, n_top_positions=5)
    important_windows = analyzer.find_important_windows(heatmaps, labels)
    
    # Should have results for class 0
    assert 0 in important_windows, "Should have results for class 0"
    
    # Should find 5 windows
    assert len(important_windows[0]) == 5, f"Expected 5 windows, got {len(important_windows[0])}"
    
    # Each window should be a tuple (start, end, importance)
    for start, end, importance in important_windows[0]:
        assert isinstance(start, int), "Start should be int"
        assert isinstance(end, int), "End should be int"
        assert isinstance(importance, float), "Importance should be float"
        assert end > start, "End should be greater than start"
        assert importance > 0, "Importance should be positive"
    
    # The most important window should include the region we added (400-480)
    top_window_start, top_window_end, _ = important_windows[0][0]
    assert top_window_start <= 440 <= top_window_end or 400 <= top_window_start <= 480, \
        f"Top window [{top_window_start}-{top_window_end}] should overlap with important region [400-480]"
    
    print("✓ Global window analysis basic test passed")


def test_global_window_analysis_per_class():
    """Test global window analysis with per-class mode."""
    torch.manual_seed(42)
    
    # Create two classes with different important regions
    heatmaps_class0 = torch.randn(50, 3, 2000) * 0.1
    heatmaps_class1 = torch.randn(50, 3, 2000) * 0.1
    
    # Class 0: important at 400-480
    heatmaps_class0[:, :, 400:480] += torch.randn(50, 3, 80) * 2.0
    
    # Class 1: important at 1200-1280
    heatmaps_class1[:, :, 1200:1280] += torch.randn(50, 3, 80) * 2.0
    
    heatmaps = torch.cat([heatmaps_class0, heatmaps_class1])
    labels = torch.cat([torch.zeros(50), torch.ones(50)]).long()
    
    # Test per-class analysis
    analyzer = GlobalWindowAnalysis(window_size=40, n_top_positions=3, per_class=True)
    important_windows = analyzer.find_important_windows(heatmaps, labels)
    
    # Should have results for both classes
    assert 0 in important_windows, "Should have results for class 0"
    assert 1 in important_windows, "Should have results for class 1"
    
    # Each class should have 3 windows
    assert len(important_windows[0]) == 3, f"Expected 3 windows for class 0"
    assert len(important_windows[1]) == 3, f"Expected 3 windows for class 1"
    
    # Class 0's top window should be in the 400-480 region
    top0_start, top0_end, _ = important_windows[0][0]
    assert 360 <= top0_start <= 480, f"Class 0 top window should be near 400-480, got {top0_start}-{top0_end}"
    
    # Class 1's top window should be in the 1200-1280 region
    top1_start, top1_end, _ = important_windows[1][0]
    assert 1160 <= top1_start <= 1280, f"Class 1 top window should be near 1200-1280, got {top1_start}-{top1_end}"
    
    print("✓ Global window analysis per-class test passed")


def test_global_window_coverage():
    """Test window coverage computation."""
    torch.manual_seed(42)
    
    # Create heatmaps where most relevance is in a specific region
    heatmaps = torch.randn(50, 3, 2000) * 0.1
    heatmaps[:, :, 800:880] += torch.randn(50, 3, 80) * 5.0  # Very important region
    
    labels = torch.zeros(50).long()
    
    # Find important windows
    analyzer = GlobalWindowAnalysis(window_size=40, n_top_positions=5)
    important_windows = analyzer.find_important_windows(heatmaps, labels)
    
    # Compute coverage
    coverage = analyzer.get_window_coverage_per_sample(heatmaps, labels)
    
    assert 0 in coverage, "Should have coverage for class 0"
    assert coverage[0].shape[0] == 50, "Coverage should have one value per sample"
    
    # Coverage should be between 0 and 1
    assert (coverage[0] >= 0).all() and (coverage[0] <= 1).all(), \
        "Coverage should be in [0, 1]"
    
    # Mean coverage should be reasonable (at least 20% since we added a strong region)
    assert coverage[0].mean() > 0.2, f"Mean coverage too low: {coverage[0].mean():.2%}"
    
    print("✓ Global window coverage test passed")


def test_global_window_threshold_mode():
    """Test threshold-based window selection."""
    torch.manual_seed(42)
    
    heatmaps = torch.randn(50, 3, 2000) * 0.1
    heatmaps[:, :, 400:480] += torch.randn(50, 3, 80) * 3.0
    labels = torch.zeros(50).long()
    
    # Test with threshold mode (n_top_positions=None)
    analyzer = GlobalWindowAnalysis(
        window_size=40,
        n_top_positions=None,  # Use threshold
        threshold_factor=1.5
    )
    important_windows = analyzer.find_important_windows(heatmaps, labels)
    
    assert 0 in important_windows, "Should have results"
    # Should select at least 1 window (the important one)
    assert len(important_windows[0]) >= 1, "Should select at least one window"
    # Should not select all windows
    n_total_windows = 2000 // 40
    assert len(important_windows[0]) < n_total_windows, \
        f"Should not select all {n_total_windows} windows"
    
    print("✓ Global window threshold mode test passed")


def test_concept_interpreter_basic():
    """Test basic concept interpreter functionality."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Synthetic CRP features: 100 samples, 64 filters
    n_filters = 64
    features_class0 = torch.randn(50, n_filters) + torch.randn(n_filters)
    features_class1 = torch.randn(50, n_filters) + torch.randn(n_filters)
    features = torch.cat([features_class0, features_class1])
    labels = torch.cat([torch.zeros(50), torch.ones(50)]).long()
    
    # Fit GMMs
    gmms = {}
    for class_id in [0, 1]:
        class_mask = labels == class_id
        class_features = features[class_mask].cpu().numpy()
        
        gmm = GaussianMixture(
            n_components=2,
            covariance_type='diag',
            n_init=3,
            max_iter=100,
            random_state=42
        )
        gmm.fit(class_features)
        gmms[class_id] = gmm
    
    # Create interpreter
    interpreter = ConceptInterpreter(gmms, features, labels, layer_name="conv3")
    
    # Create synthetic global windows
    global_windows = {
        0: [(400, 440, 0.85), (1200, 1240, 0.72)],
        1: [(800, 840, 0.91), (1600, 1640, 0.78)]
    }
    
    # Test interpretation
    interpretations = interpreter.interpret_prototypes(
        global_windows=global_windows,
        top_k_filters=10
    )
    
    # Should have interpretations for both classes
    assert 0 in interpretations, "Should have interpretations for class 0"
    assert 1 in interpretations, "Should have interpretations for class 1"
    
    # Each class should have 2 prototypes
    assert len(interpretations[0]) == 2, "Class 0 should have 2 prototypes"
    assert len(interpretations[1]) == 2, "Class 1 should have 2 prototypes"
    
    # Check structure of each interpretation
    for class_id in [0, 1]:
        for proto_idx in range(2):
            proto = interpretations[class_id][proto_idx]
            
            assert 'top_filters' in proto, "Should have top_filters"
            assert 'n_samples' in proto, "Should have n_samples"
            assert 'coverage' in proto, "Should have coverage"
            assert 'description' in proto, "Should have description"
            assert 'filter_summary' in proto, "Should have filter_summary"
            
            # Check types
            assert isinstance(proto['top_filters'], list), "top_filters should be list"
            assert isinstance(proto['n_samples'], int), "n_samples should be int"
            assert isinstance(proto['coverage'], float), "coverage should be float"
            assert isinstance(proto['description'], str), "description should be str"
            
            # Check top_filters structure
            assert len(proto['top_filters']) == 10, "Should have 10 top filters"
            for filter_idx, weight in proto['top_filters']:
                assert isinstance(filter_idx, int), "Filter idx should be int"
                assert isinstance(weight, float), "Weight should be float"
                assert 0 <= filter_idx < n_filters, f"Filter idx {filter_idx} out of range"
            
            # Check coverage is in valid range
            assert 0 <= proto['coverage'] <= 1, "Coverage should be in [0, 1]"
            
            # Check n_samples is positive
            assert proto['n_samples'] > 0, "Should have at least one sample"
    
    print("✓ Concept interpreter basic test passed")


def test_concept_interpreter_comparison():
    """Test prototype comparison between classes."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create features with some shared and some distinct filters
    n_filters = 32
    
    # Class 0: emphasize filters 0-15
    features_class0 = torch.randn(30, n_filters) * 0.5
    features_class0[:, :16] += 2.0
    
    # Class 1: emphasize filters 10-25 (some overlap with class 0)
    features_class1 = torch.randn(30, n_filters) * 0.5
    features_class1[:, 10:26] += 2.0
    
    features = torch.cat([features_class0, features_class1])
    labels = torch.cat([torch.zeros(30), torch.ones(30)]).long()
    
    # Fit GMMs
    gmms = {}
    for class_id in [0, 1]:
        class_mask = labels == class_id
        class_features = features[class_mask].cpu().numpy()
        
        gmm = GaussianMixture(
            n_components=2,
            covariance_type='diag',
            n_init=3,
            max_iter=100,
            random_state=42
        )
        gmm.fit(class_features)
        gmms[class_id] = gmm
    
    # Create interpreter
    interpreter = ConceptInterpreter(gmms, features, labels, layer_name="conv2")
    
    # Interpret
    interpretations = interpreter.interpret_prototypes(
        global_windows={},
        top_k_filters=5
    )
    
    # Compare classes
    comparison = interpreter.compare_prototypes_between_classes(interpretations)
    
    assert 'ok_n_prototypes' in comparison
    assert 'nok_n_prototypes' in comparison
    assert 'shared_filters' in comparison
    assert 'ok_only_filters' in comparison
    assert 'nok_only_filters' in comparison
    assert 'filter_overlap_ratio' in comparison
    
    # Check values
    assert comparison['ok_n_prototypes'] == 2
    assert comparison['nok_n_prototypes'] == 2
    
    # Should have some shared filters (overlap region 10-15)
    assert len(comparison['shared_filters']) > 0, "Should have some shared filters"
    
    # Overlap ratio should be between 0 and 1
    assert 0 <= comparison['filter_overlap_ratio'] <= 1
    
    print("✓ Concept interpreter comparison test passed")


def test_prototype_discovery_optimal_n_selection():
    """Test BIC-based optimal n_prototypes selection."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create synthetic data with 3 clear clusters
    cluster1 = torch.randn(30, 16) + torch.tensor([2.0] * 8 + [0.0] * 8)
    cluster2 = torch.randn(30, 16) + torch.tensor([0.0] * 8 + [2.0] * 8)
    cluster3 = torch.randn(30, 16) + torch.tensor([1.0] * 16)
    features = torch.cat([cluster1, cluster2, cluster3])
    
    # Test optimal n selection
    optimal_n, scores = TemporalPrototypeDiscovery.select_optimal_n_prototypes(
        features,
        min_prototypes=2,
        max_prototypes=5,
        covariance_type='diag',
        n_init=3,
        max_iter=100,
        criterion='bic'
    )
    
    # Should return valid n and scores
    assert isinstance(optimal_n, int), "optimal_n should be int"
    assert 2 <= optimal_n <= 5, "optimal_n should be in valid range"
    assert isinstance(scores, dict), "scores should be dict"
    assert len(scores) == 4, "Should have scores for 2, 3, 4, 5"
    
    # Scores should all be finite
    for n, score in scores.items():
        assert np.isfinite(score), f"Score for n={n} should be finite"
    
    # Lower scores are better for BIC
    # The optimal should have the lowest score
    assert scores[optimal_n] == min(scores.values()), \
        "Optimal n should have lowest BIC score"
    
    print(f"✓ Optimal n selection test passed (selected n={optimal_n})")


def test_prototype_discovery_with_improved_defaults():
    """Test that prototype discovery works with improved GMM defaults."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Synthetic features in 64-dim space (like conv3)
    features_class0 = torch.randn(50, 64) + torch.randn(64) * 2
    features_class1 = torch.randn(50, 64) + torch.randn(64) * 2
    features = torch.cat([features_class0, features_class1])
    
    labels = torch.cat([torch.zeros(50), torch.ones(50)]).long()
    
    # Simulate perfect predictions
    outputs = torch.zeros(100, 2)
    outputs[:50, 0] = 2.0
    outputs[50:, 1] = 2.0
    
    # Test with improved defaults (diag covariance, n_init=5, max_iter=200)
    discovery = TemporalPrototypeDiscovery(
        n_prototypes=4,
        covariance_type='diag',  # Better for 64-dim
        n_init=5,
        max_iter=200
    )
    
    # Should fit without issues
    discovery.fit(features, labels, outputs)
    
    # Should have GMMs for both classes
    assert 0 in discovery.gmms, "Should have GMM for class 0"
    assert 1 in discovery.gmms, "Should have GMM for class 1"
    
    # Should be able to find prototypes
    proto_samples_0 = discovery.find_prototypes(class_id=0, top_k=3)
    assert len(proto_samples_0) == 4, "Should have 4 prototypes"
    
    # Each prototype should have 3 samples
    for proto_idx, sample_indices in proto_samples_0.items():
        assert len(sample_indices) == 3, f"Prototype {proto_idx} should have 3 samples"
    
    # Coverage should sum to ~100%
    coverage = discovery.get_prototype_coverage(class_id=0)
    assert abs(coverage.sum() - 100.0) < 1.0, \
        f"Coverage should sum to ~100%, got {coverage.sum()}"
    
    print("✓ Prototype discovery with improved defaults test passed")


def test_config_has_new_settings():
    """Test that config has new settings for improved GMM and global windows."""
    import yaml
    
    # Use absolute path relative to test file location
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'default.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check primary_layer
    assert 'primary_layer' in config['tcd'], "Missing primary_layer in config"
    assert config['tcd']['primary_layer'] == 'conv3', "primary_layer should be conv3"
    
    # Check improved GMM settings
    assert config['tcd']['gmm_covariance'] == 'diag', "Should use diag covariance"
    assert config['tcd']['gmm_n_init'] == 5, "Should have n_init=5"
    assert config['tcd']['gmm_max_iter'] == 200, "Should have max_iter=200"
    
    # Check global_windows section
    assert 'global_windows' in config['tcd'], "Missing global_windows section"
    gw = config['tcd']['global_windows']
    assert 'window_size' in gw
    assert 'n_top_positions' in gw
    assert 'threshold_factor' in gw
    assert 'per_class' in gw
    
    # Check interpretation_features
    assert 'interpretation_features' in config['tcd'], "Missing interpretation_features"
    features = config['tcd']['interpretation_features']
    assert 'rms' in features
    assert 'crest_factor' in features
    assert 'kurtosis' in features
    
    print("✓ Config has new settings test passed")


if __name__ == "__main__":
    print("Running Phase 5 improvement tests...\n")
    
    try:
        test_global_window_analysis_basic()
        test_global_window_analysis_per_class()
        test_global_window_coverage()
        test_global_window_threshold_mode()
        test_concept_interpreter_basic()
        test_concept_interpreter_comparison()
        test_prototype_discovery_optimal_n_selection()
        test_prototype_discovery_with_improved_defaults()
        test_config_has_new_settings()
        
        print("\n" + "="*60)
        print("✓ ALL PHASE 5 TESTS PASSED!")
        print("="*60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
