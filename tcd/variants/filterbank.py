"""
TCD Variant A: Frequency-Band Filterbank Concepts

FULLY IMPLEMENTED

Defines concepts as frequency bands applied to relevance signals.
Uses physics-informed frequency ranges for fault detection:
- 0-10 Hz: Normal operation signatures
- 10-50 Hz: Transition band
- 50-100 Hz: Mid-frequency
- 100-200 Hz: Fault signatures (~150 Hz)

Soft concept assignment via frequency decomposition of heatmaps.
"""

import torch
import numpy as np
from scipy import signal
from typing import List, Tuple, Optional, Dict


class FilterBankTCD:
    """
    Frequency-band concept extractor for temporal relevance signals.
    
    Decomposes relevance heatmaps into frequency bands using bandpass filters,
    then computes soft concept assignments based on energy in each band.
    
    Algorithm:
    1. For each heatmap channel: apply bandpass filters for each frequency band
    2. Compute energy in each band: E_k = Σ|c_k(t)|²
    3. Soft assignment weights: w_k = E_k / ΣE_j
    4. Concept relevance: R_k = w_k · R_total
    
    Usage:
        tcd = FilterBankTCD(
            bands=[[0, 10], [10, 50], [50, 100], [100, 200]],
            sample_rate=400
        )
        concept_relevances = tcd.extract_concepts(heatmaps)
        # Returns: (batch, n_concepts) relevance per frequency band
    """
    
    def __init__(
        self,
        bands: List[List[float]],
        sample_rate: int = 400,
        filter_order: int = 4
    ):
        """
        Initialize filterbank.
        
        Args:
            bands: List of [low_hz, high_hz] frequency bands
            sample_rate: Sampling rate in Hz
            filter_order: Butterworth filter order
        """
        self.bands = bands
        self.sample_rate = sample_rate
        self.filter_order = filter_order
        self.n_concepts = len(bands)
        
        # Pre-compute filter coefficients
        self.filters = []
        for low, high in bands:
            # Design Butterworth bandpass filter
            nyquist = sample_rate / 2.0
            
            if low == 0:
                # Lowpass filter
                high_norm = high / nyquist
                sos = signal.butter(filter_order, high_norm, btype='low', output='sos')
            elif high >= nyquist:
                # Highpass filter
                low_norm = low / nyquist
                sos = signal.butter(filter_order, low_norm, btype='high', output='sos')
            else:
                # Bandpass filter
                low_norm = low / nyquist
                high_norm = high / nyquist
                sos = signal.butter(filter_order, [low_norm, high_norm], btype='band', output='sos')
            
            self.filters.append(sos)
    
    def _apply_filter(
        self,
        data: np.ndarray,
        sos: np.ndarray
    ) -> np.ndarray:
        """
        Apply bandpass filter to 1D signal.
        
        Args:
            data: Signal of shape (timesteps,)
            sos: Second-order sections filter coefficients
            
        Returns:
            Filtered signal of shape (timesteps,)
        """
        # Use sosfiltfilt for zero-phase filtering
        filtered = signal.sosfiltfilt(sos, data)
        return filtered
    
    def extract_concepts(
        self,
        heatmaps: torch.Tensor,
        aggregate_channels: bool = True
    ) -> torch.Tensor:
        """
        Extract frequency-band concept relevances from heatmaps.
        
        Args:
            heatmaps: Relevance heatmaps of shape (batch, channels, timesteps)
            aggregate_channels: If True, average across channels before filtering
            
        Returns:
            Concept relevances of shape (batch, n_concepts)
        """
        batch_size, n_channels, n_timesteps = heatmaps.shape
        
        # Convert to numpy
        heatmaps_np = heatmaps.detach().cpu().numpy()
        
        # Storage for concept energies
        concept_energies = np.zeros((batch_size, self.n_concepts))
        
        for b in range(batch_size):
            # Get heatmap for this sample
            if aggregate_channels:
                # Average across channels (X, Y, Z → single signal)
                heatmap = heatmaps_np[b].mean(axis=0)
            else:
                # Use first channel or sum
                heatmap = heatmaps_np[b].sum(axis=0)
            
            # Apply each filter and compute energy
            for k, sos in enumerate(self.filters):
                filtered = self._apply_filter(heatmap, sos)
                
                # Energy in this band
                energy = np.sum(np.abs(filtered))
                concept_energies[b, k] = energy
        
        # Soft assignment: normalize energies to sum to 1
        total_energy = concept_energies.sum(axis=1, keepdims=True)
        concept_weights = np.divide(
            concept_energies,
            total_energy,
            out=np.zeros_like(concept_energies),
            where=total_energy > 0
        )
        
        # Compute total relevance per sample
        total_relevance = np.abs(heatmaps_np).sum(axis=(1, 2))
        
        # Concept relevance = weight * total relevance
        concept_relevances = concept_weights * total_relevance[:, None]
        
        return torch.from_numpy(concept_relevances).float()
    
    def get_concept_labels(self) -> List[str]:
        """
        Get human-readable labels for concepts.
        
        Returns:
            List of concept labels (e.g., '0-10 Hz')
        """
        labels = []
        for low, high in self.bands:
            labels.append(f"{low}-{high} Hz")
        return labels
    
    def visualize_concept_decomposition(
        self,
        heatmap: np.ndarray,
        sample_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """
        Decompose a single heatmap into frequency band components.
        
        Args:
            heatmap: Heatmap of shape (channels, timesteps) or (timesteps,)
            sample_idx: Sample index for labeling
            
        Returns:
            Dictionary mapping band label -> filtered signal
        """
        if heatmap.ndim == 2:
            # Average across channels
            heatmap = heatmap.mean(axis=0)
        
        decomposition = {}
        labels = self.get_concept_labels()
        
        for k, (sos, label) in enumerate(zip(self.filters, labels)):
            filtered = self._apply_filter(heatmap, sos)
            decomposition[label] = filtered
        
        return decomposition
    
    def compute_concept_importance(
        self,
        heatmaps: torch.Tensor
    ) -> np.ndarray:
        """
        Compute importance of each concept across dataset.
        
        Args:
            heatmaps: Heatmaps of shape (n_samples, channels, timesteps)
            
        Returns:
            Mean concept relevances of shape (n_concepts,)
        """
        concept_relevances = self.extract_concepts(heatmaps)
        return concept_relevances.mean(dim=0).numpy()


def test_filterbank_tcd():
    """Test FilterBankTCD on synthetic data."""
    # Create synthetic heatmap with multiple frequency components
    sample_rate = 400
    duration = 5  # seconds
    n_samples = sample_rate * duration  # 2000
    
    t = np.linspace(0, duration, n_samples)
    
    # Multi-frequency signal
    signal_10hz = np.sin(2 * np.pi * 10 * t)
    signal_50hz = np.sin(2 * np.pi * 50 * t)
    signal_150hz = np.sin(2 * np.pi * 150 * t)
    
    # Combine into 3-channel heatmap
    heatmap = np.array([
        signal_10hz + 0.5 * signal_150hz,
        signal_50hz,
        signal_150hz + 0.3 * signal_10hz
    ])
    
    # Add batch dimension
    heatmaps = torch.from_numpy(heatmap[None, :, :]).float()
    
    # Test filterbank
    tcd = FilterBankTCD(
        bands=[[0, 10], [10, 50], [50, 100], [100, 200]],
        sample_rate=sample_rate
    )
    
    concept_relevances = tcd.extract_concepts(heatmaps)
    
    print(f"Input heatmap shape: {heatmaps.shape}")
    print(f"Concept relevances shape: {concept_relevances.shape}")
    print(f"Concept relevances: {concept_relevances[0]}")
    print(f"Concept labels: {tcd.get_concept_labels()}")
    
    # Test decomposition
    decomp = tcd.visualize_concept_decomposition(heatmap)
    print(f"Decomposition keys: {list(decomp.keys())}")
    
    # Verify shape
    assert concept_relevances.shape == (1, 4), f"Expected (1, 4), got {concept_relevances.shape}"
    
    print("✓ FilterBankTCD test passed!")


if __name__ == "__main__":
    test_filterbank_tcd()
