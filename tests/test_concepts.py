"""
Test concept extraction for 1D time series.

Verifies that concept extraction produces correct shapes
and works with different TCD variants.
"""

import torch
import numpy as np
import pytest
from tcd.concepts import ChannelConcept, FilterBankConcept
from tcd.variants.filterbank import FilterBankTCD


def test_channel_concept():
    """Test ChannelConcept on 1D data."""
    cc = ChannelConcept()
    
    # Simulate layer relevance: (batch=4, filters=16, timesteps=500)
    relevance = torch.randn(4, 16, 500)
    
    # Extract per-channel concepts
    concept_rel = cc.attribute(relevance, abs_norm=True)
    
    # Should return (batch, filters)
    assert concept_rel.shape == (4, 16), \
        f"Expected shape (4, 16), got {concept_rel.shape}"
    
    # All values should be non-negative (abs_norm=True)
    assert (concept_rel >= 0).all(), "Expected non-negative values with abs_norm=True"
    
    print("✓ ChannelConcept test passed")


def test_channel_concept_without_abs_norm():
    """Test ChannelConcept without absolute normalization."""
    cc = ChannelConcept()
    
    relevance = torch.randn(2, 8, 100)
    concept_rel = cc.attribute(relevance, abs_norm=False)
    
    assert concept_rel.shape == (2, 8)
    
    # Values can be negative without abs_norm
    # Just check it runs and has correct shape
    
    print("✓ ChannelConcept (no abs_norm) test passed")


def test_filterbank_tcd():
    """Test FilterBankTCD on synthetic data."""
    # Create synthetic multi-frequency signal
    sample_rate = 400
    duration = 5
    n_samples = sample_rate * duration
    
    t = np.linspace(0, duration, n_samples)
    
    # Signal with multiple frequency components
    signal_10hz = np.sin(2 * np.pi * 10 * t)
    signal_50hz = np.sin(2 * np.pi * 50 * t)
    signal_150hz = np.sin(2 * np.pi * 150 * t)
    
    # 3-channel heatmap
    heatmap = np.array([
        signal_10hz + 0.5 * signal_150hz,
        signal_50hz,
        signal_150hz
    ])
    
    # Batch of 2 samples
    heatmaps = torch.from_numpy(np.stack([heatmap, heatmap])).float()
    
    # Initialize FilterBankTCD
    bands = [[0, 10], [10, 50], [50, 100], [100, 200]]
    tcd = FilterBankTCD(bands=bands, sample_rate=sample_rate)
    
    # Extract concepts
    concept_relevances = tcd.extract_concepts(heatmaps)
    
    # Should return (batch, n_concepts)
    assert concept_relevances.shape == (2, 4), \
        f"Expected shape (2, 4), got {concept_relevances.shape}"
    
    # All values should be non-negative (energies)
    assert (concept_relevances >= 0).all(), "Expected non-negative energies"
    
    print("✓ FilterBankTCD test passed")


def test_filterbank_concept_labels():
    """Test concept label generation."""
    bands = [[0, 10], [10, 50], [50, 100], [100, 200]]
    tcd = FilterBankTCD(bands=bands, sample_rate=400)
    
    labels = tcd.get_concept_labels()
    
    assert len(labels) == 4
    assert labels[0] == '0-10 Hz'
    assert labels[3] == '100-200 Hz'
    
    print("✓ FilterBankTCD concept labels test passed")


def test_filterbank_decomposition():
    """Test frequency decomposition visualization."""
    sample_rate = 400
    duration = 5
    n_samples = sample_rate * duration
    
    t = np.linspace(0, duration, n_samples)
    signal = np.sin(2 * np.pi * 50 * t)  # 50 Hz signal
    
    # Create heatmap
    heatmap = np.array([signal, signal, signal])
    
    # Initialize TCD
    bands = [[0, 20], [20, 60], [60, 100], [100, 200]]
    tcd = FilterBankTCD(bands=bands, sample_rate=sample_rate)
    
    # Get decomposition
    decomp = tcd.visualize_concept_decomposition(heatmap)
    
    # Should have one entry per band
    assert len(decomp) == 4
    
    # Each filtered signal should have same length
    for label, filtered in decomp.items():
        assert len(filtered) == n_samples
    
    # 50 Hz signal should have most energy in 20-60 Hz band
    energies = {label: np.abs(sig).sum() for label, sig in decomp.items()}
    max_band = max(energies, key=energies.get)
    assert max_band == '20-60 Hz', f"Expected max energy in 20-60 Hz, got {max_band}"
    
    print("✓ FilterBankTCD decomposition test passed")


def test_filterbank_importance():
    """Test concept importance computation."""
    sample_rate = 400
    n_samples = 2000
    
    # Create batch of heatmaps
    heatmaps = torch.randn(10, 3, n_samples)
    
    # Initialize TCD
    bands = [[0, 10], [10, 50], [50, 100], [100, 200]]
    tcd = FilterBankTCD(bands=bands, sample_rate=sample_rate)
    
    # Compute importance
    importance = tcd.compute_concept_importance(heatmaps)
    
    assert importance.shape == (4,), f"Expected shape (4,), got {importance.shape}"
    assert (importance >= 0).all(), "Expected non-negative importance"
    
    print("✓ FilterBankTCD importance test passed")


def test_different_band_configurations():
    """Test TCD with different frequency band configurations."""
    sample_rate = 400
    n_samples = 2000
    heatmap = torch.randn(1, 3, n_samples)
    
    # Test various band configurations
    band_configs = [
        [[0, 50], [50, 200]],  # 2 bands
        [[0, 25], [25, 75], [75, 150], [150, 200]],  # 4 bands
        [[0, 10], [10, 30], [30, 60], [60, 100], [100, 200]],  # 5 bands
    ]
    
    for bands in band_configs:
        tcd = FilterBankTCD(bands=bands, sample_rate=sample_rate)
        concepts = tcd.extract_concepts(heatmap)
        
        assert concepts.shape == (1, len(bands)), \
            f"Expected shape (1, {len(bands)}), got {concepts.shape}"
        
        print(f"✓ FilterBankTCD with {len(bands)} bands passed")


if __name__ == "__main__":
    print("Running concept extraction tests...\n")
    
    test_channel_concept()
    test_channel_concept_without_abs_norm()
    test_filterbank_tcd()
    test_filterbank_concept_labels()
    test_filterbank_decomposition()
    test_filterbank_importance()
    test_different_band_configurations()
    
    print("\n✓ All concept tests passed!")
