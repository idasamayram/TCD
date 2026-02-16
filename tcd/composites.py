"""
LRP composite rules for 1D convolutional networks.

Adapted from pcx_codes/utils/lrp_composites.py for Conv1d.
Uses zennit.types.Convolution (covers both Conv1d and Conv2d).
No canonizers needed since CNN1D_Wide has no BatchNorm/GroupNorm.
"""

import torch
import torch.nn as nn
from zennit.composites import LayerMapComposite, EpsilonPlusFlat as ZennitEpsilonPlusFlat
from zennit.rules import Epsilon, ZPlus, Flat, Pass
from zennit.types import Convolution, Linear, AvgPool, Activation
from typing import Dict, Callable


def get_composite(name: str = 'epsilon_plus') -> LayerMapComposite:
    """
    Get LRP composite rule for CNN1D models.
    
    Args:
        name: Composite name - 'epsilon_plus', 'epsilon', or 'gradient'
        
    Returns:
        LayerMapComposite for use with CondAttribution
    """
    if name == 'epsilon_plus':
        # Use zennit's built-in EpsilonPlusFlat (no canonizers needed)
        return ZennitEpsilonPlusFlat(canonizers=None)
    elif name == 'epsilon':
        return EpsilonComposite()
    elif name == 'gradient':
        return GradientComposite()
    else:
        raise ValueError(f"Unknown composite: {name}")


class EpsilonComposite(LayerMapComposite):
    """
    LRP-ε composite - Epsilon rule for all layers.
    
    Simpler variant using only Epsilon rule throughout.
    Good baseline for testing.
    """
    
    def __init__(self, epsilon: float = 1e-6):
        """
        Initialize composite.
        
        Args:
            epsilon: Stabilizer for Epsilon rule
        """
        layer_map = [
            (Activation, Pass()),
            (AvgPool, Pass()),
            (Linear, Epsilon(epsilon=epsilon)),
            (Convolution, Epsilon(epsilon=epsilon)),
        ]
        
        super().__init__(layer_map=layer_map, canonizers=None)


class GradientComposite(LayerMapComposite):
    """
    Gradient-based composite for comparison.
    
    Equivalent to gradient × input (not true LRP).
    Useful as baseline to show LRP's advantages.
    """
    
    def __init__(self):
        """Initialize gradient composite."""
        layer_map = [
            (Activation, Pass()),
            (AvgPool, Pass()),
            (Linear, Pass()),
            (Convolution, Pass()),
        ]
        
        super().__init__(layer_map=layer_map, canonizers=None)


def test_composite():
    """Test composites on CNN1D model."""
    from models.cnn1d_model import CNN1D_Wide
    
    model = CNN1D_Wide()
    x = torch.randn(2, 3, 2000)
    
    for name in ['epsilon_plus', 'epsilon', 'gradient']:
        composite = get_composite(name)
        print(f"✓ Created composite: {name}")
        print(f"  Layer map: {len(composite.layer_map)} rules")


if __name__ == "__main__":
    test_composite()
