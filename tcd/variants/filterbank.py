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


class WindowConceptTCD:
    """
    Window-based concept discovery using data-driven LRP heatmap analysis.
    
    Replaces hardcoded frequency bands with automatic discovery of important
    time windows from LRP heatmaps, then clusters rich feature vectors.
    
    Algorithm:
    1. Extract top-K most important time windows from heatmaps (by absolute relevance)
    2. For each window, compute CNC-relevant features:
       - RMS amplitude (overall vibration energy)
       - Peak frequency (resonance/chatter frequency)
       - Crest factor (peak/RMS - impulsiveness)
       - Kurtosis (spikiness - fault detection)
       - Zero crossing rate (frequency proxy)
       - Spectral energy, centroid (frequency content)
       - Mean, std, max amplitude (statistical features)
    3. Cluster feature vectors using GMM to discover concepts
    4. Each cluster = one concept (e.g., "high-amplitude impulse", "low-freq oscillation")
    
    This approach is data-driven and model-informed, letting the LRP heatmaps
    guide concept discovery rather than using physics-based frequency bands.
    """
    
    def __init__(
        self,
        n_concepts: int = 6,
        window_size: int = 40,  # timesteps (0.1s at 400Hz)
        n_top_windows: int = 20,  # windows to extract per sample
        sample_rate: int = 400,  # Hz
        features: Optional[List[str]] = None,
        gmm_covariance: str = 'full',
        gmm_n_init: int = 10,
        gmm_max_iter: int = 100,
        random_state: int = 42
    ):
        """
        Initialize window-based concept discovery.
        
        Args:
            n_concepts: Number of concepts to discover via GMM
            window_size: Size of time windows in timesteps
            n_top_windows: Number of top windows to extract per sample
            sample_rate: Sampling rate in Hz
            features: List of feature names to extract (default: all)
            gmm_covariance: GMM covariance type
            gmm_n_init: Number of GMM initializations
            gmm_max_iter: Max GMM iterations
            random_state: Random seed
        """
        from sklearn.mixture import GaussianMixture
        
        self.n_concepts = n_concepts
        self.window_size = window_size
        self.n_top_windows = n_top_windows
        self.sample_rate = sample_rate
        self.random_state = random_state
        
        # Default features if not specified
        if features is None:
            features = [
                'rms', 'peak_freq', 'crest_factor', 'kurtosis',
                'zero_crossing_rate', 'spectral_energy', 'spectral_centroid',
                'mean_amplitude', 'std_amplitude', 'max_amplitude'
            ]
        self.features = features
        
        # GMM for clustering
        self.gmm = GaussianMixture(
            n_components=n_concepts,
            covariance_type=gmm_covariance,
            n_init=gmm_n_init,
            max_iter=gmm_max_iter,
            random_state=random_state
        )
        
        self.fitted = False
        self.cluster_centers = None
    
    def _extract_windows(
        self,
        heatmaps: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract top-K most important time windows from heatmaps.
        
        Args:
            heatmaps: Relevance heatmaps (batch, channels, timesteps)
            
        Returns:
            windows: Extracted windows (batch * n_top_windows, channels, window_size)
            importances: Window importance scores (batch * n_top_windows,)
        """
        batch_size, n_channels, n_timesteps = heatmaps.shape
        
        # Average across channels to get overall importance
        importance_signal = heatmaps.abs().mean(dim=1).cpu().numpy()  # (batch, timesteps)
        
        windows_list = []
        importances_list = []
        
        for b in range(batch_size):
            signal = heatmaps[b].cpu().numpy()  # (channels, timesteps)
            importance = importance_signal[b]  # (timesteps,)
            
            # Compute sliding window importance
            n_windows = (n_timesteps - self.window_size) // self.window_size + 1
            window_importances = []
            window_starts = []
            
            for i in range(n_windows):
                start = i * self.window_size
                end = start + self.window_size
                if end > n_timesteps:
                    break
                
                # Sum of absolute relevance in window
                window_imp = importance[start:end].sum()
                window_importances.append(window_imp)
                window_starts.append(start)
            
            # Get top-K windows
            top_k = min(self.n_top_windows, len(window_importances))
            top_indices = np.argsort(window_importances)[-top_k:]
            
            for idx in top_indices:
                start = window_starts[idx]
                end = start + self.window_size
                window = signal[:, start:end]
                windows_list.append(window)
                importances_list.append(window_importances[idx])
        
        windows = np.array(windows_list)  # (batch * n_top_windows, channels, window_size)
        importances = np.array(importances_list)  # (batch * n_top_windows,)
        
        return windows, importances
    
    def _compute_features(
        self,
        windows: np.ndarray
    ) -> np.ndarray:
        """
        Compute feature vector for each window.
        
        Args:
            windows: Time windows (n_windows, channels, window_size)
            
        Returns:
            features: Feature matrix (n_windows, n_features)
        """
        from scipy import stats
        from scipy.fft import rfft, rfftfreq
        
        n_windows, n_channels, window_size = windows.shape
        feature_list = []
        
        for window in windows:
            # Average across channels
            signal = window.mean(axis=0)  # (window_size,)
            
            features = {}
            
            # RMS amplitude (overall vibration energy)
            if 'rms' in self.features:
                features['rms'] = np.sqrt(np.mean(signal**2))
            
            # Mean, std, max amplitude
            if 'mean_amplitude' in self.features:
                features['mean_amplitude'] = np.abs(signal).mean()
            if 'std_amplitude' in self.features:
                features['std_amplitude'] = signal.std()
            if 'max_amplitude' in self.features:
                features['max_amplitude'] = np.abs(signal).max()
            
            # Crest factor (peak/RMS - impulsiveness)
            if 'crest_factor' in self.features:
                rms = np.sqrt(np.mean(signal**2))
                peak = np.abs(signal).max()
                features['crest_factor'] = peak / (rms + 1e-6)
            
            # Kurtosis (spikiness - fault detection)
            if 'kurtosis' in self.features:
                features['kurtosis'] = stats.kurtosis(signal)
            
            # Zero crossing rate
            if 'zero_crossing_rate' in self.features:
                zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
                features['zero_crossing_rate'] = zero_crossings / len(signal)
            
            # Frequency-domain features
            fft_vals = np.abs(rfft(signal))
            freqs = rfftfreq(window_size, d=1/self.sample_rate)
            
            # Peak frequency
            if 'peak_freq' in self.features:
                peak_idx = np.argmax(fft_vals)
                features['peak_freq'] = freqs[peak_idx]
            
            # Spectral energy
            if 'spectral_energy' in self.features:
                features['spectral_energy'] = np.sum(fft_vals**2)
            
            # Spectral centroid
            if 'spectral_centroid' in self.features:
                spectral_centroid = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-6)
                features['spectral_centroid'] = spectral_centroid
            
            # Construct feature vector in consistent order
            feature_vec = [features[f] for f in self.features if f in features]
            feature_list.append(feature_vec)
        
        return np.array(feature_list)
    
    def fit(
        self,
        heatmaps: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Fit GMM to discover concepts from heatmaps.
        
        Args:
            heatmaps: Relevance heatmaps (n_samples, channels, timesteps)
            labels: Optional class labels for per-class normalization
        """
        # Extract windows
        windows, importances = self._extract_windows(heatmaps)
        
        # Compute features
        features = self._compute_features(windows)
        
        # Fit GMM
        self.gmm.fit(features)
        self.cluster_centers = self.gmm.means_
        self.fitted = True
        
        print(f"✓ Fitted GMM with {self.n_concepts} concepts from {len(features)} windows")
    
    def extract_concepts(
        self,
        heatmaps: torch.Tensor,
        raw_signals: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract concept relevances from heatmaps.
        
        Args:
            heatmaps: Relevance heatmaps (batch, channels, timesteps)
            raw_signals: Optional raw signals (not used, for API compatibility)
            
        Returns:
            Concept relevances (batch, n_concepts)
        """
        if not self.fitted:
            raise ValueError("Must call fit() before extract_concepts()")
        
        batch_size = heatmaps.shape[0]
        
        # Extract windows
        windows, importances = self._extract_windows(heatmaps)
        
        # Compute features
        features = self._compute_features(windows)
        
        # Assign to concepts
        assignments = self.gmm.predict(features)
        
        # Aggregate by sample
        concept_relevances = np.zeros((batch_size, self.n_concepts))
        
        for i in range(batch_size):
            start = i * self.n_top_windows
            end = start + self.n_top_windows
            if end > len(assignments):
                end = len(assignments)
            
            sample_assignments = assignments[start:end]
            sample_importances = importances[start:end]
            
            # Sum importance for each concept
            for c in range(self.n_concepts):
                mask = sample_assignments == c
                concept_relevances[i, c] = sample_importances[mask].sum()
        
        return torch.from_numpy(concept_relevances).float()
    
    def get_concept_labels(self) -> List[str]:
        """
        Get human-readable labels for concepts based on cluster centers.
        
        Returns:
            List of concept labels
        """
        if not self.fitted or self.cluster_centers is None:
            return [f"Concept-{i}" for i in range(self.n_concepts)]
        
        labels = []
        for i, center in enumerate(self.cluster_centers):
            # Characterize each concept by its dominant features
            feature_dict = {f: center[j] for j, f in enumerate(self.features) if j < len(center)}
            
            # Create descriptive label based on feature values
            # High RMS -> high energy
            # High crest factor -> impulsive
            # High kurtosis -> spiky/fault-like
            # High peak_freq -> high frequency
            
            descriptors = []
            if 'rms' in feature_dict and feature_dict['rms'] > np.median([c[self.features.index('rms')] for c in self.cluster_centers if 'rms' in self.features]):
                descriptors.append('high-energy')
            if 'crest_factor' in feature_dict and feature_dict['crest_factor'] > 4:
                descriptors.append('impulsive')
            if 'kurtosis' in feature_dict and feature_dict['kurtosis'] > 3:
                descriptors.append('spiky')
            if 'peak_freq' in feature_dict:
                freq = feature_dict['peak_freq']
                if freq < 20:
                    descriptors.append('low-freq')
                elif freq > 100:
                    descriptors.append('high-freq')
            
            if descriptors:
                label = f"Concept-{i}: {'-'.join(descriptors)}"
            else:
                label = f"Concept-{i}"
            
            labels.append(label)
        
        return labels
    
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


if __name__ == "__main__":
    test_filterbank_tcd()
