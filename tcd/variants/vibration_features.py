"""
TCD Variant D: Comprehensive Vibration Feature Concepts

This variant extracts a comprehensive set of vibration-relevant features from
heatmaps and/or raw signals to serve as concepts. Unlike frequency bands only,
this provides a rich feature space for discovering meaningful fault patterns.

Features are organized into three categories:
1. Time-domain features (RMS, Crest Factor, Kurtosis, etc.)
2. Frequency-domain features (Spectral Centroid, Entropy, etc.)
3. Vibration-specific features (Envelope analysis, Inter-axis correlation, etc.)

Each feature becomes a concept dimension. Features are automatically normalized
and can be filtered by discriminative power (mutual information, Fisher score).
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal, stats
from scipy.fft import rfft, rfftfreq
from scipy.signal import hilbert
from sklearn.feature_selection import mutual_info_classif
from sklearn.mixture import GaussianMixture
from typing import List, Optional, Dict, Tuple


class VibrationFeatureTCD:
    """
    Extract comprehensive vibration features as temporal concepts.
    
    This variant automatically extracts a rich set of vibration-relevant
    features from heatmaps and/or raw signals. Each feature becomes a
    concept dimension, enabling automated discovery of fault patterns
    without requiring domain expertise.
    
    Usage:
        tcd = VibrationFeatureTCD(
            sample_rate=400,
            window_size=100,
            n_concepts=20,  # Auto-select top 20 discriminative features
            use_feature_selection=True
        )
        
        # Fit on training data
        tcd.fit(heatmaps, labels=labels, signals=signals)
        
        # Extract concepts
        concept_vectors = tcd.extract_concepts(heatmaps, signals=signals)
        
        # Get concept labels
        labels = tcd.get_concept_labels()
    """
    
    def __init__(
        self,
        sample_rate: int = 400,
        window_size: int = 100,
        n_concepts: Optional[int] = None,
        use_feature_selection: bool = True,
        selection_method: str = 'mutual_info',  # 'mutual_info' or 'fisher'
        n_prototypes: int = 4,
        gmm_covariance: str = 'full',
        gmm_n_init: int = 10,
        gmm_max_iter: int = 100
    ):
        """
        Initialize vibration feature TCD.
        
        Args:
            sample_rate: Sampling rate in Hz
            window_size: Window size for computing features (timesteps)
            n_concepts: Number of top features to select (None = use all)
            use_feature_selection: Whether to filter features by discriminative power
            selection_method: Feature selection method ('mutual_info' or 'fisher')
            n_prototypes: Number of GMM prototypes per class
            gmm_covariance: GMM covariance type
            gmm_n_init: GMM number of initializations
            gmm_max_iter: GMM max iterations
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.n_concepts = n_concepts
        self.use_feature_selection = use_feature_selection
        self.selection_method = selection_method
        
        # GMM parameters
        self.n_prototypes = n_prototypes
        self.gmm_covariance = gmm_covariance
        self.gmm_n_init = gmm_n_init
        self.gmm_max_iter = gmm_max_iter
        
        # Fitted state
        self.feature_names: List[str] = []
        self.selected_features: Optional[List[int]] = None
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.gmms: Dict[int, GaussianMixture] = {}
        self.fitted = False
    
    def extract_time_domain_features(
        self,
        signal: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract time-domain features from a signal segment.
        
        Args:
            signal: 1D signal array
            
        Returns:
            Dictionary of feature_name -> value
        """
        features = {}
        
        # Basic statistics
        signal_abs = np.abs(signal)
        signal_mean = np.mean(signal)
        signal_mean_abs = np.mean(signal_abs)
        
        # RMS (Root Mean Square)
        rms = np.sqrt(np.mean(signal**2))
        features['rms'] = rms
        
        # Peak value
        peak = np.max(signal_abs)
        features['peak'] = peak
        
        # Crest Factor (peak / RMS)
        features['crest_factor'] = peak / (rms + 1e-10)
        
        # Kurtosis (peakedness - high for impulsive faults)
        features['kurtosis'] = stats.kurtosis(signal)
        
        # Skewness (asymmetry)
        features['skewness'] = stats.skew(signal)
        
        # Peak-to-Average Ratio
        features['peak_to_avg'] = peak / (signal_mean_abs + 1e-10)
        
        # Zero-Crossing Rate
        zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
        features['zero_crossing_rate'] = zero_crossings / len(signal)
        
        # Waveform Factor (RMS / mean absolute)
        features['waveform_factor'] = rms / (signal_mean_abs + 1e-10)
        
        # Impulse Factor (peak / mean absolute)
        features['impulse_factor'] = peak / (signal_mean_abs + 1e-10)
        
        # Clearance Factor (peak / square of mean of square root of absolute)
        mean_sqrt_abs = np.mean(np.sqrt(signal_abs))
        features['clearance_factor'] = peak / (mean_sqrt_abs**2 + 1e-10)
        
        # Standard deviation
        features['std'] = np.std(signal)
        
        # Variance
        features['variance'] = np.var(signal)
        
        return features
    
    def extract_frequency_domain_features(
        self,
        signal: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract frequency-domain features from a signal segment.
        
        Args:
            signal: 1D signal array
            
        Returns:
            Dictionary of feature_name -> value
        """
        features = {}
        
        # Compute FFT
        fft_values = np.abs(rfft(signal))
        fft_freqs = rfftfreq(len(signal), 1/self.sample_rate)
        
        # Normalize spectrum
        fft_values_norm = fft_values / (np.sum(fft_values) + 1e-10)
        
        # Spectral Centroid (center of mass)
        features['spectral_centroid'] = np.sum(fft_freqs * fft_values_norm)
        
        # Spectral Entropy (randomness of spectrum)
        entropy = -np.sum(fft_values_norm * np.log(fft_values_norm + 1e-10))
        features['spectral_entropy'] = entropy
        
        # Dominant Frequency
        dominant_idx = np.argmax(fft_values)
        features['dominant_frequency'] = fft_freqs[dominant_idx]
        
        # Spectral Kurtosis
        features['spectral_kurtosis'] = stats.kurtosis(fft_values)
        
        # Spectral Skewness
        features['spectral_skewness'] = stats.skew(fft_values)
        
        # Spectral Rolloff (95% of energy)
        cumsum = np.cumsum(fft_values)
        rolloff_idx = np.where(cumsum >= 0.95 * cumsum[-1])[0]
        if len(rolloff_idx) > 0:
            features['spectral_rolloff'] = fft_freqs[rolloff_idx[0]]
        else:
            features['spectral_rolloff'] = fft_freqs[-1]
        
        # Band Energy Ratios (4 bands)
        # Define frequency bands
        bands = [
            (0, 10, 'band_0_10Hz'),
            (10, 50, 'band_10_50Hz'),
            (50, 100, 'band_50_100Hz'),
            (100, 200, 'band_100_200Hz')
        ]
        
        for low, high, name in bands:
            mask = (fft_freqs >= low) & (fft_freqs < high)
            band_energy = np.sum(fft_values[mask])
            total_energy = np.sum(fft_values) + 1e-10
            features[name] = band_energy / total_energy
        
        # Spectral Flatness (ratio of geometric to arithmetic mean)
        geometric_mean = np.exp(np.mean(np.log(fft_values + 1e-10)))
        arithmetic_mean = np.mean(fft_values)
        features['spectral_flatness'] = geometric_mean / (arithmetic_mean + 1e-10)
        
        return features
    
    def extract_envelope_features(
        self,
        signal: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract envelope analysis features (demodulation for bearing faults).
        
        Args:
            signal: 1D signal array
            
        Returns:
            Dictionary of feature_name -> value
        """
        features = {}
        
        # Compute analytic signal (Hilbert transform)
        analytic_signal = hilbert(signal)
        envelope = np.abs(analytic_signal)
        
        # Envelope statistics
        features['envelope_mean'] = np.mean(envelope)
        features['envelope_std'] = np.std(envelope)
        features['envelope_peak'] = np.max(envelope)
        features['envelope_rms'] = np.sqrt(np.mean(envelope**2))
        
        # Envelope kurtosis (high for bearing faults)
        features['envelope_kurtosis'] = stats.kurtosis(envelope)
        
        return features
    
    def extract_multi_axis_features(
        self,
        signals: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract features from multi-axis correlation.
        
        Args:
            signals: 2D array of shape (n_channels, n_timesteps)
            
        Returns:
            Dictionary of feature_name -> value
        """
        features = {}
        
        if signals.shape[0] < 2:
            # Single channel, skip correlation features
            return features
        
        # Inter-axis correlations
        if signals.shape[0] >= 3:
            # Assuming X, Y, Z axes
            features['corr_xy'] = np.corrcoef(signals[0], signals[1])[0, 1]
            features['corr_xz'] = np.corrcoef(signals[0], signals[2])[0, 1]
            features['corr_yz'] = np.corrcoef(signals[1], signals[2])[0, 1]
        elif signals.shape[0] == 2:
            features['corr_01'] = np.corrcoef(signals[0], signals[1])[0, 1]
        
        # Energy ratio between axes
        energies = np.sum(signals**2, axis=1)
        total_energy = np.sum(energies) + 1e-10
        for i, energy in enumerate(energies):
            features[f'axis_{i}_energy_ratio'] = energy / total_energy
        
        return features
    
    def extract_all_features(
        self,
        heatmap: torch.Tensor,
        signal: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Extract all features from a single sample.
        
        Args:
            heatmap: Heatmap tensor of shape (n_channels, n_timesteps)
            signal: Optional raw signal tensor of shape (n_channels, n_timesteps)
            
        Returns:
            Feature vector as numpy array
        """
        # Convert to numpy
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.cpu().numpy()
        if signal is not None and isinstance(signal, torch.Tensor):
            signal = signal.cpu().numpy()
        
        all_features = {}
        
        # Extract features from each channel of heatmap
        for ch_idx in range(heatmap.shape[0]):
            ch_signal = heatmap[ch_idx]
            
            # Time-domain features
            time_features = self.extract_time_domain_features(ch_signal)
            for name, value in time_features.items():
                all_features[f'heatmap_ch{ch_idx}_{name}'] = value
            
            # Frequency-domain features
            freq_features = self.extract_frequency_domain_features(ch_signal)
            for name, value in freq_features.items():
                all_features[f'heatmap_ch{ch_idx}_{name}'] = value
        
        # Multi-axis features from heatmap
        multi_features = self.extract_multi_axis_features(heatmap)
        for name, value in multi_features.items():
            all_features[f'heatmap_{name}'] = value
        
        # If raw signal provided, extract features from it too
        if signal is not None:
            for ch_idx in range(signal.shape[0]):
                ch_signal = signal[ch_idx]
                
                # Time-domain features
                time_features = self.extract_time_domain_features(ch_signal)
                for name, value in time_features.items():
                    all_features[f'signal_ch{ch_idx}_{name}'] = value
                
                # Frequency-domain features
                freq_features = self.extract_frequency_domain_features(ch_signal)
                for name, value in freq_features.items():
                    all_features[f'signal_ch{ch_idx}_{name}'] = value
                
                # Envelope features
                env_features = self.extract_envelope_features(ch_signal)
                for name, value in env_features.items():
                    all_features[f'signal_ch{ch_idx}_{name}'] = value
            
            # Multi-axis features from signal
            multi_features = self.extract_multi_axis_features(signal)
            for name, value in multi_features.items():
                all_features[f'signal_{name}'] = value
        
        # Convert to array (maintain consistent ordering)
        if not self.feature_names:
            self.feature_names = sorted(all_features.keys())
        
        feature_vector = np.array([all_features[name] for name in self.feature_names])
        
        # Replace NaN/Inf with 0
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        return feature_vector
    
    def fit(
        self,
        heatmaps: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        signals: Optional[torch.Tensor] = None
    ):
        """
        Fit feature extractor and optionally perform feature selection.
        
        Args:
            heatmaps: Heatmap tensors of shape (N, n_channels, n_timesteps)
            labels: Optional labels for feature selection (shape: N)
            signals: Optional raw signals of shape (N, n_channels, n_timesteps)
        """
        print("Extracting features from all samples...")
        
        # Extract features from all samples
        feature_vectors = []
        for i in range(len(heatmaps)):
            heatmap = heatmaps[i]
            signal = signals[i] if signals is not None else None
            features = self.extract_all_features(heatmap, signal)
            feature_vectors.append(features)
        
        feature_matrix = np.array(feature_vectors)
        print(f"Extracted {feature_matrix.shape[1]} features from {feature_matrix.shape[0]} samples")
        
        # Normalize features
        self.feature_mean = np.mean(feature_matrix, axis=0)
        self.feature_std = np.std(feature_matrix, axis=0) + 1e-10
        feature_matrix_norm = (feature_matrix - self.feature_mean) / self.feature_std
        
        # Feature selection
        if self.use_feature_selection and labels is not None:
            print(f"\nPerforming feature selection using {self.selection_method}...")
            
            if isinstance(labels, torch.Tensor):
                labels_np = labels.cpu().numpy()
            else:
                labels_np = labels
            
            if self.selection_method == 'mutual_info':
                # Mutual information
                mi_scores = mutual_info_classif(feature_matrix_norm, labels_np, random_state=0)
                feature_scores = mi_scores
            elif self.selection_method == 'fisher':
                # Fisher score
                feature_scores = self._compute_fisher_scores(feature_matrix_norm, labels_np)
            else:
                raise ValueError(f"Unknown selection method: {self.selection_method}")
            
            # Select top features
            if self.n_concepts is not None:
                n_select = min(self.n_concepts, len(feature_scores))
            else:
                # Select features with score > threshold
                threshold = np.median(feature_scores)
                n_select = np.sum(feature_scores > threshold)
            
            self.selected_features = np.argsort(feature_scores)[-n_select:][::-1]
            
            print(f"Selected top {len(self.selected_features)} features out of {len(self.feature_names)}")
            print(f"Selected features:")
            for idx in self.selected_features[:10]:  # Show top 10
                print(f"  {self.feature_names[idx]}: {feature_scores[idx]:.4f}")
            if len(self.selected_features) > 10:
                print(f"  ... and {len(self.selected_features) - 10} more")
        else:
            # Use all features
            self.selected_features = np.arange(len(self.feature_names))
        
        # Fit GMM on selected features
        if labels is not None:
            print("\nFitting GMM prototypes per class...")
            selected_features = feature_matrix_norm[:, self.selected_features]
            
            if isinstance(labels, torch.Tensor):
                labels_np = labels.cpu().numpy()
            else:
                labels_np = labels
            
            unique_classes = np.unique(labels_np)
            for class_id in unique_classes:
                mask = labels_np == class_id
                class_features = selected_features[mask]
                
                if class_features.shape[0] < self.n_prototypes:
                    print(f"Warning: Class {class_id} has only {class_features.shape[0]} samples")
                    continue
                
                gmm = GaussianMixture(
                    n_components=self.n_prototypes,
                    covariance_type=self.gmm_covariance,
                    n_init=self.gmm_n_init,
                    max_iter=self.gmm_max_iter,
                    random_state=0
                )
                gmm.fit(class_features)
                self.gmms[class_id] = gmm
                
                print(f"  Class {class_id}: Fitted GMM with {self.n_prototypes} prototypes on {class_features.shape[0]} samples")
        
        self.fitted = True
    
    def _compute_fisher_scores(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """
        Compute Fisher score for each feature.
        
        Fisher score = (between-class variance) / (within-class variance)
        
        Args:
            features: Feature matrix of shape (N, n_features)
            labels: Labels of shape (N,)
            
        Returns:
            Fisher scores of shape (n_features,)
        """
        n_features = features.shape[1]
        fisher_scores = np.zeros(n_features)
        
        unique_classes = np.unique(labels)
        overall_mean = np.mean(features, axis=0)
        
        for feat_idx in range(n_features):
            feat = features[:, feat_idx]
            
            # Between-class variance
            between_var = 0
            for class_id in unique_classes:
                mask = labels == class_id
                class_mean = np.mean(feat[mask])
                n_samples = np.sum(mask)
                between_var += n_samples * (class_mean - overall_mean[feat_idx])**2
            
            # Within-class variance
            within_var = 0
            for class_id in unique_classes:
                mask = labels == class_id
                class_features = feat[mask]
                class_mean = np.mean(class_features)
                within_var += np.sum((class_features - class_mean)**2)
            
            # Fisher score
            fisher_scores[feat_idx] = between_var / (within_var + 1e-10)
        
        return fisher_scores
    
    def extract_concepts(
        self,
        heatmaps: torch.Tensor,
        signals: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract concept vectors from heatmaps/signals.
        
        Args:
            heatmaps: Heatmap tensors of shape (N, n_channels, n_timesteps)
            signals: Optional raw signals of shape (N, n_channels, n_timesteps)
            
        Returns:
            Concept vectors of shape (N, n_concepts)
        """
        if not self.fitted:
            raise ValueError("Must call fit() before extract_concepts()")
        
        # Extract features from all samples
        feature_vectors = []
        for i in range(len(heatmaps)):
            heatmap = heatmaps[i]
            signal = signals[i] if signals is not None else None
            features = self.extract_all_features(heatmap, signal)
            feature_vectors.append(features)
        
        feature_matrix = np.array(feature_vectors)
        
        # Normalize
        feature_matrix_norm = (feature_matrix - self.feature_mean) / self.feature_std
        
        # Select features
        selected_features = feature_matrix_norm[:, self.selected_features]
        
        return torch.from_numpy(selected_features).float()
    
    def get_concept_labels(self) -> List[str]:
        """
        Get labels for selected concepts.
        
        Returns:
            List of concept names
        """
        if not self.fitted:
            return []
        
        return [self.feature_names[idx] for idx in self.selected_features]
    
    def compute_concept_importance(
        self,
        heatmaps: torch.Tensor,
        signals: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Compute importance of each concept (mean absolute value).
        
        Args:
            heatmaps: Heatmap tensors
            signals: Optional raw signals
            
        Returns:
            Importance scores for each concept
        """
        concept_vectors = self.extract_concepts(heatmaps, signals)
        importance = torch.mean(torch.abs(concept_vectors), dim=0).numpy()
        return importance


if __name__ == "__main__":
    # Test vibration feature extraction
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Synthetic data
    n_samples = 100
    n_channels = 3
    n_timesteps = 2000
    
    heatmaps = torch.randn(n_samples, n_channels, n_timesteps)
    signals = torch.randn(n_samples, n_channels, n_timesteps)
    labels = torch.cat([torch.zeros(50), torch.ones(50)]).long()
    
    # Initialize and fit
    tcd = VibrationFeatureTCD(
        sample_rate=400,
        n_concepts=20,
        use_feature_selection=True
    )
    
    tcd.fit(heatmaps, labels=labels, signals=signals)
    
    # Extract concepts
    concept_vectors = tcd.extract_concepts(heatmaps, signals=signals)
    print(f"\nConcept vectors shape: {concept_vectors.shape}")
    
    # Get labels
    concept_labels = tcd.get_concept_labels()
    print(f"\nSelected {len(concept_labels)} concepts")
    
    # Compute importance
    importance = tcd.compute_concept_importance(heatmaps, signals=signals)
    print(f"\nTop 5 most important concepts:")
    top_indices = np.argsort(importance)[-5:][::-1]
    for idx in top_indices:
        print(f"  {concept_labels[idx]}: {importance[idx]:.4f}")
    
    print("\n✓ Vibration feature TCD test passed!")
