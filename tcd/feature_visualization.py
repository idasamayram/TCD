"""
FeatureVisualization adapted for 1D time-series data.

Wraps crp.FeatureVisualization with 1D-specific handling.
The base class is designed for images, so we adapt its internal
mechanics to work with time-series signals.
"""

import torch
import numpy as np
from crp.helper import get_layer_names
from crp.visualization import FeatureVisualization as _FeatureVisualization
from typing import Dict, Callable, Optional, List, Tuple
import os


class FeatureVisualization(_FeatureVisualization):
    """
    Feature visualization for 1D time-series CRP analysis.
    
    Extends crp.FeatureVisualization to handle 1D signals instead of images.
    The base class expects 2D spatial data, so we override methods that
    assume image-specific shapes.
    
    Usage:
        from tcd.attribution import TimeSeriesCondAttribution
        from tcd.concepts import ChannelConcept
        
        cc = ChannelConcept()
        layer_map = {layer: cc for layer in layer_names}
        
        attributor = TimeSeriesCondAttribution(model)
        fv = FeatureVisualization(
            attributor, 
            dataset, 
            layer_map,
            preprocess_fn=dataset.preprocessing,
            path='crp_files/run1',
            cache=None
        )
        fv.run(composite, start=0, end=100, batch_size=32)
    """
    
    def __init__(
        self,
        attribution,
        dataset,
        layer_map: Dict,
        preprocess_fn: Optional[Callable] = None,
        path: str = "crp_files",
        cache: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize feature visualization.
        
        Args:
            attribution: TimeSeriesCondAttribution instance
            dataset: VibrationDataset instance
            layer_map: Dict mapping layer names to Concept instances
            preprocess_fn: Optional preprocessing function
            path: Path to save CRP files
            cache: Optional cache path
            device: Device to use
        """
        # Ensure path exists
        os.makedirs(path, exist_ok=True)
        
        super().__init__(
            attribution=attribution,
            dataset=dataset,
            layer_map=layer_map,
            preprocess_fn=preprocess_fn,
            path=path,
            cache=cache,
            device=device
        )
    
    def _get_data_shape(self) -> Tuple[int, ...]:
        """
        Get shape of dataset samples.
        
        Overrides base method which assumes 4D image tensors.
        
        Returns:
            Tuple of (channels, timesteps) e.g., (3, 2000)
        """
        sample, _ = self.dataset[0]
        return tuple(sample.shape)
    
    def get_max_reference(
        self,
        concept_ids: List[int],
        layer_name: str,
        mode: str = "max",
        ref_range: Tuple[int, int] = (0, 10),
        composite=None,
        rf: bool = True,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Get reference signals for top concept activations.
        
        Adapted from pcx_codes crp_plot_prototype.py.
        Returns signals that maximally activate given concepts.
        
        Args:
            concept_ids: List of concept indices to visualize
            layer_name: Layer name to analyze
            mode: 'max' for maximum activation samples
            ref_range: (start, end) range of reference samples
            composite: LRP composite to use
            rf: Whether to use receptive field (keep True for CRP)
            **kwargs: Additional arguments
            
        Returns:
            List of reference signals as tensors
        """
        # This method requires implementing the base class's caching mechanism
        # For now, return placeholder - full implementation requires
        # adapting the internal _get_activations and caching logic
        
        # TODO: Implement full reference signal extraction
        # This requires:
        # 1. Loading pre-computed activations from cache
        # 2. Finding samples with max activation for each concept
        # 3. Extracting the relevant signal segments (receptive field)
        
        print(f"Warning: get_max_reference not fully implemented for 1D")
        return []


def get_layer_names_model(model: torch.nn.Module, model_name: str = 'cnn1d') -> List[str]:
    """
    Get convolutional layer names from model.
    
    Helper function adapted from pcx_codes crp_run.py.
    
    Args:
        model: PyTorch model
        model_name: Model type (for special handling)
        
    Returns:
        List of layer names suitable for CRP analysis
    """
    layer_names = []
    
    # For CNN1D_Wide, get Conv1d layers from features module
    if hasattr(model, 'features'):
        for i, module in enumerate(model.features):
            if isinstance(module, torch.nn.Conv1d):
                layer_names.append(f"features.{i}")
    else:
        # Fallback: search all Conv1d layers
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv1d):
                layer_names.append(name)
    
    return layer_names


if __name__ == "__main__":
    from models.cnn1d_model import CNN1D_Wide, get_layer_names
    
    model = CNN1D_Wide()
    
    # Test layer name extraction
    layer_names = get_layer_names_model(model)
    print(f"Layer names: {layer_names}")
    
    expected = ['features.0', 'features.3', 'features.6']
    assert layer_names == expected, f"Expected {expected}, got {layer_names}"
    
    print("✓ FeatureVisualization test passed!")
