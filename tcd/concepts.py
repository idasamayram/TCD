"""
Concept definitions for 1D time series.

The base ChannelConcept from zennit-crp works correctly for 1D:
    .view(*shape[:2], -1) flattens (B, C, T) → (B, C, T) correctly

No custom TimeSeriesConcept needed - ChannelConcept handles it.
"""

from crp.concepts import ChannelConcept as _ChannelConcept
from typing import Optional
import torch


class ChannelConcept(_ChannelConcept):
    """
    Concept extractor for channel-wise analysis of 1D time series.
    
    Wraps zennit-crp's ChannelConcept for clarity and future extensibility.
    The base implementation correctly handles 1D data:
    
    For relevance tensor R with shape (B, C, T):
    - .view(*shape[:2], -1) → (B, C, T) preserves structure
    - .attribute() returns per-channel relevance: (B, C)
    
    Each "concept" = one convolutional filter/channel at the analyzed layer.
    
    Usage:
        cc = ChannelConcept()
        concept_relevances = cc.attribute(
            layer_relevance,  # (B, num_filters, T)
            abs_norm=True
        )
        # Returns: (B, num_filters) - relevance per filter
    """
    
    def __init__(self):
        super().__init__()
    
    def attribute(
        self, 
        relevance: torch.Tensor,
        abs_norm: bool = True,
        signed_norm: bool = False
    ) -> torch.Tensor:
        """
        Compute per-channel concept relevance.
        
        Args:
            relevance: Layer relevance tensor of shape (B, C, T)
            abs_norm: If True, use absolute values before summing (default behavior)
            signed_norm: If True, sum relevance with sign (overrides abs_norm)
                        This preserves positive/negative contributions in signed composites
                        like AlphaBeta+Gamma (CNCValidatedComposite)
            
        Returns:
            Concept relevance of shape (B, C) - one value per channel/filter
        """
        # signed_norm takes precedence over abs_norm
        if signed_norm:
            # Sum relevance preserving sign - for signed composites
            # relevance shape: (B, C, T) -> (B, C)
            shape = relevance.shape
            return relevance.view(*shape[:2], -1).sum(dim=2)
        else:
            # Use parent's abs_norm behavior
            return super().attribute(relevance, abs_norm=abs_norm)


class FilterBankConcept:
    """
    Frequency-band concept extractor for TCD Variant A.
    
    Defines concepts as frequency bands applied to relevance signals.
    Each concept is a temporal filter that isolates specific frequency
    components known to be relevant for fault detection:
    
    - 0-10 Hz: Normal operation signatures
    - 10-50 Hz: Transition band
    - 50-100 Hz: Mid-frequency
    - 100-200 Hz: Fault signatures (~150 Hz)
    
    See tcd/variants/filterbank.py for full implementation.
    """
    
    def __init__(self, bands: list, sample_rate: int = 400):
        """
        Initialize filterbank concept.
        
        Args:
            bands: List of [low_hz, high_hz] frequency bands
            sample_rate: Sampling rate in Hz
        """
        self.bands = bands
        self.sample_rate = sample_rate
        self.n_concepts = len(bands)
    
    def attribute(
        self, 
        relevance: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract frequency-band concept relevances.
        
        Args:
            relevance: Input-level relevance of shape (B, C, T)
            
        Returns:
            Concept relevances of shape (B, n_concepts)
        """
        # Placeholder - full implementation in variants/filterbank.py
        raise NotImplementedError("Use tcd.variants.filterbank.FilterBankTCD")


class WindowConcept:
    """
    Window-based concept extractor for data-driven concept discovery.
    
    Instead of hardcoded frequency bands, this extracts important time windows
    from LRP heatmaps, computes rich feature vectors (RMS, peak frequency,
    crest factor, kurtosis, etc.), and clusters them using GMM.
    
    Each cluster represents a discovered concept (e.g., "high-amplitude impulse",
    "low-frequency oscillation") that the model uses for predictions.
    
    This is more flexible and data-driven than FilterBankConcept.
    
    See tcd/variants/filterbank.WindowConceptTCD for full implementation.
    """
    
    def __init__(
        self,
        n_concepts: int = 6,
        window_size: int = 40,
        n_top_windows: int = 20,
        sample_rate: int = 400
    ):
        """
        Initialize window-based concept.
        
        Args:
            n_concepts: Number of concepts to discover
            window_size: Time window size in timesteps
            n_top_windows: Number of windows to extract per sample
            sample_rate: Sampling rate in Hz
        """
        self.n_concepts = n_concepts
        self.window_size = window_size
        self.n_top_windows = n_top_windows
        self.sample_rate = sample_rate
    
    def attribute(
        self,
        relevance: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract window-based concept relevances.
        
        Args:
            relevance: Input-level relevance of shape (B, C, T)
            
        Returns:
            Concept relevances of shape (B, n_concepts)
        """
        # Placeholder - full implementation in variants/filterbank.py
        raise NotImplementedError("Use tcd.variants.filterbank.WindowConceptTCD")


if __name__ == "__main__":
    # Test ChannelConcept on synthetic 1D data
    cc = ChannelConcept()
    
    # Simulate layer relevance: (batch=2, filters=16, timesteps=500)
    relevance = torch.randn(2, 16, 500)
    
    concept_rel = cc.attribute(relevance, abs_norm=True)
    
    print(f"Input relevance shape: {relevance.shape}")
    print(f"Concept relevance shape: {concept_rel.shape}")
    assert concept_rel.shape == (2, 16), f"Expected (2, 16), got {concept_rel.shape}"
    print("✓ ChannelConcept test passed!")
