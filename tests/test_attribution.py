"""
Test CRP attribution for 1D time series.

Verifies that TimeSeriesCondAttribution correctly preserves
channel structure (batch, 3, 2000) instead of collapsing it.
"""

import torch
import pytest
from models.cnn1d_model import CNN1D_Wide
from tcd.attribution import TimeSeriesCondAttribution
from tcd.composites import get_composite


def test_heatmap_shape_preservation():
    """Test that heatmap shape is preserved (not collapsed)."""
    # Create model
    model = CNN1D_Wide()
    model.eval()
    
    # Create synthetic input (batch=2, channels=3, timesteps=2000)
    x = torch.randn(2, 3, 2000, requires_grad=True)
    
    # Initialize attribution
    attributor = TimeSeriesCondAttribution(model)
    composite = get_composite('epsilon_plus')
    
    # Compute heatmap
    conditions = [{"y": 0}, {"y": 1}]
    result = attributor(x, conditions, composite)
    
    # Verify shape preservation
    assert result.heatmap.shape == (2, 3, 2000), \
        f"Expected shape (2, 3, 2000), got {result.heatmap.shape}"
    
    print("✓ Heatmap shape preservation test passed")


def test_heatmap_gradient_flow():
    """Test that gradients flow correctly through attribution."""
    model = CNN1D_Wide()
    model.eval()
    
    x = torch.randn(1, 3, 2000, requires_grad=True)
    
    attributor = TimeSeriesCondAttribution(model)
    composite = get_composite('epsilon_plus')
    
    conditions = [{"y": 0}]
    result = attributor(x, conditions, composite)
    
    # Heatmap should be computed from gradients
    assert result.heatmap is not None
    assert result.heatmap.shape == x.shape
    assert not result.heatmap.requires_grad  # Should be detached
    
    print("✓ Gradient flow test passed")


def test_attribution_with_layer_recording():
    """Test attribution with layer recording for CRP analysis."""
    model = CNN1D_Wide()
    model.eval()
    
    x = torch.randn(2, 3, 2000, requires_grad=True)
    
    attributor = TimeSeriesCondAttribution(model)
    composite = get_composite('epsilon_plus')
    
    # Record specific layers
    layer_names = ['conv1', 'conv2', 'conv3']
    conditions = [{"y": 0}, {"y": 0}]
    
    result = attributor(
        x, 
        conditions, 
        composite,
        record_layer=layer_names
    )
    
    # Verify layer relevances are recorded
    assert hasattr(result, 'relevances')
    assert all(layer in result.relevances for layer in layer_names)
    
    # Check shapes of layer relevances
    for layer in layer_names:
        rel = result.relevances[layer]
        assert rel.shape[0] == 2  # Batch size
        assert rel.ndim == 3  # (batch, channels, timesteps)
        print(f"Layer {layer} relevance shape: {rel.shape}")
    
    print("✓ Layer recording test passed")


def test_different_composites():
    """Test attribution with different LRP composites."""
    model = CNN1D_Wide()
    model.eval()
    
    x = torch.randn(1, 3, 2000, requires_grad=True)
    attributor = TimeSeriesCondAttribution(model)
    conditions = [{"y": 0}]
    
    for composite_name in ['epsilon_plus', 'custom_cnn1d']:
        composite = get_composite(composite_name)
        result = attributor(x, conditions, composite)
        
        assert result.heatmap.shape == (1, 3, 2000), \
            f"Composite {composite_name} failed shape check"
        
        print(f"✓ Composite {composite_name} works correctly")


def test_batch_consistency():
    """Test that attribution is consistent across batch dimension."""
    model = CNN1D_Wide()
    model.eval()
    
    # Create identical inputs
    x = torch.randn(1, 3, 2000)
    x_batch = x.repeat(3, 1, 1)  # Batch of 3 identical samples
    x_batch.requires_grad_(True)
    
    attributor = TimeSeriesCondAttribution(model)
    composite = get_composite('epsilon_plus')
    
    # Same conditions for all
    conditions = [{"y": 0}, {"y": 0}, {"y": 0}]
    result = attributor(x_batch, conditions, composite)
    
    # Heatmaps should be similar (allowing for numerical differences)
    for i in range(1, 3):
        diff = (result.heatmap[0] - result.heatmap[i]).abs().mean()
        assert diff < 1e-5, f"Batch inconsistency: diff = {diff}"
    
    print("✓ Batch consistency test passed")


if __name__ == "__main__":
    print("Running attribution tests...\n")
    
    test_heatmap_shape_preservation()
    test_heatmap_gradient_flow()
    test_attribution_with_layer_recording()
    test_different_composites()
    test_batch_consistency()
    
    print("\n✓ All attribution tests passed!")
