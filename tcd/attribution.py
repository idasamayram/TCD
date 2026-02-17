"""
TimeSeriesCondAttribution - Fixed CRP attribution for 1D time series.

CRITICAL FIX: The default CondAttribution.heatmap_modifier() in zennit-crp
sums/collapses spatial dimensions, which destroys the 3-channel structure
(X/Y/Z accelerometer axes) of vibration data.

For 1D time series with shape (batch, channels, timesteps), we must preserve
all dimensions: heatmap shape MUST be (batch, 3, 2000), NOT (batch, 1, 2000).

Adapted from idasamayram/zennit-crp tutorials/cnn1d_attribution.py
"""

import torch
from crp.attribution import CondAttribution
from typing import Optional


class TimeSeriesCondAttribution(CondAttribution):
    """
    CRP attribution with fixed heatmap modifier for 1D time series.
    
    The key difference from the base CondAttribution:
    - Base class: collapses spatial dimensions with .sum(dim=tuple(range(2, heatmap.ndim)))
    - This class: preserves full (batch, channels, timesteps) shape
    
    This is essential for multi-channel time series where each channel
    (e.g., X/Y/Z accelerometer) has independent physical meaning.
    
    Usage:
        attributor = TimeSeriesCondAttribution(model)
        heatmap, _, _, _ = attributor(
            input_tensor, 
            conditions=[{"y": target_class}], 
            composite=composite
        )
        # heatmap shape: (batch, 3, 2000) ✓ NOT (batch, 1, 2000) ✗
    """
    
    def heatmap_modifier(
        self, 
        data: torch.Tensor, 
        on_device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Extract heatmap from gradient WITHOUT collapsing channels.
        
        Args:
            data: Input tensor with requires_grad=True, shape (B, C, T)
            on_device: Optional device to move heatmap to
            
        Returns:
            Heatmap with shape (B, C, T) preserving all dimensions
        """
        # Safety check: ensure gradient was computed
        if data.grad is None:
            raise RuntimeError(
                "data.grad is None. This can happen if:\n"
                "1. Model and data are on different devices (check device placement)\n"
                "2. Gradient computation was disabled\n"
                "3. The backward pass didn't reach the input tensor"
            )
        
        heatmap = data.grad.detach()
        
        # DO NOT collapse channels - keep full (batch, channels, timesteps) shape
        # The base class does: heatmap.sum(dim=tuple(range(2, heatmap.ndim)))
        # We skip that to preserve multi-channel structure
        
        if on_device is not None:
            heatmap = heatmap.to(on_device)
        
        return heatmap


if __name__ == "__main__":
    # Test shape preservation
    from models.cnn1d_model import CNN1D_Wide
    from tcd.composites import get_composite
    
    model = CNN1D_Wide()
    model.eval()
    
    x = torch.randn(2, 3, 2000, requires_grad=True)
    
    # Test with TimeSeriesCondAttribution
    attributor = TimeSeriesCondAttribution(model)
    composite = get_composite('epsilon_plus')
    
    heatmap, _, _, _ = attributor(
        x, 
        conditions=[{"y": 0}, {"y": 0}],
        composite=composite
    )
    
    print(f"Input shape: {x.shape}")
    print(f"Heatmap shape: {heatmap.shape}")
    assert heatmap.shape == (2, 3, 2000), f"Expected (2, 3, 2000), got {heatmap.shape}"
    print("✓ Shape preservation test passed!")
