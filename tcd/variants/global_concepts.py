"""
Global window analysis for temporal concept discovery.

Finds important temporal window positions GLOBALLY across all samples,
rather than extracting top-K windows per sample.

This addresses the fundamental issue that per-sample window extraction
is circular - it finds "types of windows" not model concepts.

Algorithm:
1. Divide all timesteps into non-overlapping windows
2. For each window position, compute MEAN absolute heatmap relevance across all samples
3. Identify globally-important positions (adaptive threshold or top-K)
4. These positions are where the model consistently looks across the dataset
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class GlobalWindowAnalysis:
    """
    Find globally important temporal window positions across all samples.
    
    Unlike per-sample window extraction (which is circular), this finds
    the temporal positions where the model consistently attends, regardless
    of individual sample variation.
    
    Usage:
        analyzer = GlobalWindowAnalysis(window_size=40, n_top_positions=10)
        important_positions = analyzer.find_important_windows(heatmaps, labels)
        # Returns: {class_id: [(start, end, importance_score), ...]}
    """
    
    def __init__(
        self,
        window_size: int = 40,
        n_top_positions: Optional[int] = 10,
        threshold_factor: float = 1.5,
        per_class: bool = True,
        use_signed_relevance: bool = False
    ):
        """
        Initialize global window analyzer.
        
        Args:
            window_size: Window size in timesteps (e.g., 40 timesteps = 0.1s at 400Hz)
            n_top_positions: Number of top window positions to select (None = use threshold)
            threshold_factor: For threshold mode: keep windows with importance > mean + factor*std
            per_class: If True, find important windows per class; else across all data
            use_signed_relevance: If True, use mean signed relevance instead of mean absolute
                                 This is useful for signed composites (AlphaBeta+Gamma)
                                 where positive = supports prediction, negative = argues against
        """
        self.window_size = window_size
        self.n_top_positions = n_top_positions
        self.threshold_factor = threshold_factor
        self.per_class = per_class
        self.use_signed_relevance = use_signed_relevance
        
        self.important_windows: Dict[int, List[Tuple[int, int, float]]] = {}
    
    def find_important_windows(
        self,
        heatmaps: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[int, List[Tuple[int, int, float]]]:
        """
        Find globally important window positions.
        
        Args:
            heatmaps: Heatmap relevances of shape (N, C, T) where:
                      N = number of samples
                      C = number of channels (e.g., 3 for X/Y/Z accelerometer)
                      T = number of timesteps (e.g., 2000)
            labels: Optional class labels of shape (N,) for per-class analysis
            
        Returns:
            Dictionary mapping class_id -> list of (start, end, importance_score) tuples
            If per_class=False, uses class_id=-1 for all samples
        """
        if heatmaps.dim() != 3:
            raise ValueError(f"Expected heatmaps of shape (N, C, T), got {heatmaps.shape}")
        
        n_samples, n_channels, n_timesteps = heatmaps.shape
        n_windows = n_timesteps // self.window_size
        
        if n_windows == 0:
            raise ValueError(f"Window size {self.window_size} is too large for {n_timesteps} timesteps")
        
        # Determine classes to analyze
        if self.per_class and labels is not None:
            unique_classes = torch.unique(labels).cpu().numpy()
        else:
            unique_classes = [-1]  # Analyze all samples together
            labels = torch.full((n_samples,), -1, dtype=torch.long)
        
        results = {}
        
        for class_id in unique_classes:
            # Filter to this class (or all if class_id=-1)
            if class_id == -1:
                class_mask = torch.ones(n_samples, dtype=torch.bool)
            else:
                class_mask = labels == class_id
            
            class_heatmaps = heatmaps[class_mask]  # (N_class, C, T)
            
            if class_heatmaps.shape[0] == 0:
                continue
            
            # Compute global importance profile
            window_importances = []
            window_positions = []
            
            for win_idx in range(n_windows):
                start_pos = win_idx * self.window_size
                end_pos = min(start_pos + self.window_size, n_timesteps)
                
                # Extract window from all samples
                window_heatmaps = class_heatmaps[:, :, start_pos:end_pos]  # (N_class, C, win_size)
                
                # Compute mean importance across samples and channels
                # This gives us: "how important is this temporal position on average?"
                if self.use_signed_relevance:
                    # Use signed relevance - preserves positive/negative contributions
                    mean_importance = window_heatmaps.mean().item()
                else:
                    # Use absolute relevance (default) - only magnitude matters
                    mean_importance = torch.abs(window_heatmaps).mean().item()
                
                window_importances.append(mean_importance)
                window_positions.append((start_pos, end_pos))
            
            window_importances = np.array(window_importances)
            
            # Select important windows
            if self.n_top_positions is not None:
                # Top-K selection
                k = min(self.n_top_positions, len(window_importances))
                top_indices = np.argsort(window_importances)[-k:][::-1]
            else:
                # Threshold-based selection
                threshold = window_importances.mean() + self.threshold_factor * window_importances.std()
                top_indices = np.where(window_importances >= threshold)[0]
            
            # Store results: (start, end, importance)
            important_windows = [
                (window_positions[idx][0], window_positions[idx][1], window_importances[idx])
                for idx in top_indices
            ]
            
            # Sort by importance (descending)
            important_windows.sort(key=lambda x: x[2], reverse=True)
            
            results[int(class_id)] = important_windows
            
            # Print summary
            class_name = f"Class {class_id}" if class_id != -1 else "All samples"
            print(f"\n{class_name}:")
            print(f"  Total windows: {n_windows}")
            print(f"  Selected windows: {len(important_windows)}")
            print(f"  Mean importance: {window_importances.mean():.6f}")
            print(f"  Std importance: {window_importances.std():.6f}")
            if len(important_windows) > 0:
                print(f"  Top window importance: {important_windows[0][2]:.6f}")
                print(f"  Top 5 window positions (timesteps):")
                for i, (start, end, imp) in enumerate(important_windows[:5]):
                    print(f"    {i+1}. [{start:4d}-{end:4d}]: {imp:.6f}")
        
        self.important_windows = results
        return results
    
    def extract_window_features(
        self,
        signals: torch.Tensor,
        window_positions: List[Tuple[int, int, float]],
        feature_names: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Extract features from raw signals at globally-important window positions.
        
        This is for INTERPRETATION of CRP concepts, not for concept definition.
        
        Args:
            signals: Raw signal data of shape (N, C, T)
            window_positions: List of (start, end, importance) tuples from find_important_windows
            feature_names: List of feature names to extract (None = extract all)
            
        Returns:
            features: Feature tensor of shape (N, n_windows * n_features)
            feature_labels: List of feature labels like "window_0_rms", "window_1_peak_freq", etc.
        """
        if signals.dim() != 3:
            raise ValueError(f"Expected signals of shape (N, C, T), got {signals.shape}")
        
        n_samples = signals.shape[0]
        
        # Define available features
        available_features = [
            'rms', 'crest_factor', 'kurtosis', 'skewness',
            'peak_freq', 'spectral_energy', 'zero_crossing_rate',
            'envelope_rms', 'inter_axis_corr', 'band_energy_ratio',
            'spectral_flatness'
        ]
        
        if feature_names is None:
            feature_names = available_features
        else:
            # Validate feature names
            invalid = set(feature_names) - set(available_features)
            if invalid:
                raise ValueError(f"Unknown features: {invalid}")
        
        all_features = []
        feature_labels = []
        
        for win_idx, (start, end, importance) in enumerate(window_positions):
            window_signals = signals[:, :, start:end]  # (N, C, win_size)
            
            # Extract features for this window
            window_features = self._extract_features(window_signals, feature_names)
            all_features.append(window_features)
            
            # Create labels
            for feat_name in feature_names:
                feature_labels.append(f"window_{win_idx}_{feat_name}")
        
        # Concatenate all window features
        features = torch.cat(all_features, dim=1)  # (N, n_windows * n_features)
        
        return features, feature_labels
    
    def _extract_features(
        self,
        window: torch.Tensor,
        feature_names: List[str]
    ) -> torch.Tensor:
        """
        Extract specified features from a window.
        
        Args:
            window: Signal window of shape (N, C, win_size)
            feature_names: List of feature names to extract
            
        Returns:
            features: Feature tensor of shape (N, len(feature_names))
        """
        n_samples = window.shape[0]
        features = []
        
        for feat_name in feature_names:
            if feat_name == 'rms':
                # RMS per channel, then mean across channels
                feat = torch.sqrt((window ** 2).mean(dim=2)).mean(dim=1, keepdim=True)
            elif feat_name == 'crest_factor':
                # Peak-to-RMS ratio
                peak = torch.abs(window).max(dim=2)[0].mean(dim=1, keepdim=True)
                rms = torch.sqrt((window ** 2).mean(dim=2)).mean(dim=1, keepdim=True)
                feat = peak / (rms + 1e-8)
            elif feat_name == 'kurtosis':
                # Fourth moment (excess kurtosis)
                mean = window.mean(dim=2, keepdim=True)
                std = window.std(dim=2, keepdim=True) + 1e-8
                kurt = ((window - mean) ** 4).mean(dim=2) / (std.squeeze(2) ** 4) - 3
                feat = kurt.mean(dim=1, keepdim=True)
            elif feat_name == 'skewness':
                # Third moment
                mean = window.mean(dim=2, keepdim=True)
                std = window.std(dim=2, keepdim=True) + 1e-8
                skew = ((window - mean) ** 3).mean(dim=2) / (std.squeeze(2) ** 3)
                feat = skew.mean(dim=1, keepdim=True)
            elif feat_name == 'zero_crossing_rate':
                # Zero crossing rate per channel, then mean
                signs = torch.sign(window)
                # Detect sign changes
                sign_changes = (signs[:, :, 1:] != signs[:, :, :-1]).float()
                # Calculate rate per channel
                zcr_per_channel = sign_changes.sum(dim=2) / window.shape[2]
                # Mean across channels
                feat = zcr_per_channel.mean(dim=1, keepdim=True)
            elif feat_name == 'inter_axis_corr':
                # Correlation between channels (for 3-axis accelerometer)
                if window.shape[1] >= 2:
                    # Correlation between first two channels
                    ch0 = window[:, 0, :]  # (N, T)
                    ch1 = window[:, 1, :]
                    ch0_centered = ch0 - ch0.mean(dim=1, keepdim=True)
                    ch1_centered = ch1 - ch1.mean(dim=1, keepdim=True)
                    corr = (ch0_centered * ch1_centered).sum(dim=1) / (
                        torch.sqrt((ch0_centered ** 2).sum(dim=1) * (ch1_centered ** 2).sum(dim=1)) + 1e-8
                    )
                    feat = corr.unsqueeze(1)
                else:
                    feat = torch.zeros(n_samples, 1)
            else:
                # Placeholder for other features (would need FFT for frequency-domain)
                # For simplicity, return zeros
                feat = torch.zeros(n_samples, 1)
            
            features.append(feat)
        
        return torch.cat(features, dim=1)
    
    def get_window_coverage_per_sample(
        self,
        heatmaps: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[int, torch.Tensor]:
        """
        For each sample, compute what fraction of its relevance falls within the globally-important windows.
        
        This helps understand if the global windows are representative of individual samples.
        
        Args:
            heatmaps: Heatmap relevances of shape (N, C, T)
            labels: Optional class labels of shape (N,)
            
        Returns:
            Dictionary mapping class_id -> coverage tensor of shape (N_class,)
            Coverage = sum(relevance in important windows) / sum(total relevance)
        """
        if not self.important_windows:
            raise ValueError("Must call find_important_windows() first")
        
        n_samples = heatmaps.shape[0]
        
        if labels is None:
            labels = torch.full((n_samples,), -1, dtype=torch.long)
        
        results = {}
        
        for class_id, window_list in self.important_windows.items():
            # Get samples for this class
            if class_id == -1:
                class_mask = torch.ones(n_samples, dtype=torch.bool)
            else:
                class_mask = labels == class_id
            
            class_heatmaps = heatmaps[class_mask]  # (N_class, C, T)
            
            # Compute total relevance per sample
            total_relevance = torch.abs(class_heatmaps).sum(dim=(1, 2))  # (N_class,)
            
            # Compute relevance in important windows
            window_relevance = torch.zeros(class_heatmaps.shape[0])
            for start, end, _ in window_list:
                window_relevance += torch.abs(class_heatmaps[:, :, start:end]).sum(dim=(1, 2))
            
            # Compute coverage
            coverage = window_relevance / (total_relevance + 1e-8)
            results[class_id] = coverage
        
        return results
    
    def extract_important_windows_per_sample(
        self,
        heatmaps: torch.Tensor,
        signals: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        n_top_windows: int = 10,
        sample_rate: int = 400
    ) -> Dict:
        """
        CNC-style per-sample important window extraction.
        
        Unlike the global averaging approach, this:
        1. For each sample, finds top-K most important windows
        2. Extracts per-window features from RAW signal
        3. Aggregates across samples per class
        4. Performs statistical tests between classes
        
        This is the proven approach from idasamayram/CNC evaluation/time_faithfulness.py
        
        Args:
            heatmaps: Heatmap relevances of shape (N, C, T)
            signals: Optional raw signals of shape (N, C, T)
            labels: Optional class labels of shape (N,)
            n_top_windows: Number of top windows to extract per sample
            sample_rate: Sampling rate in Hz
            
        Returns:
            Dictionary with:
                - 'windows': List of dicts with per-window features
                - 'per_class_stats': Aggregate statistics per class
                - 'statistical_tests': T-test results between classes
        """
        if heatmaps.dim() != 3:
            raise ValueError(f"Expected heatmaps of shape (N, C, T), got {heatmaps.shape}")
        
        n_samples, n_channels, n_timesteps = heatmaps.shape
        n_windows_per_timestep = n_timesteps // self.window_size
        
        if labels is None:
            labels = torch.zeros(n_samples, dtype=torch.long)
        
        # Collect all windows across all samples
        all_windows = []
        
        print(f"\nExtracting top-{n_top_windows} windows per sample...")
        print(f"  Window size: {self.window_size} timesteps ({self.window_size/sample_rate:.3f}s at {sample_rate}Hz)")
        print(f"  Total windows per sample: {n_windows_per_timestep}")
        
        for sample_idx in range(n_samples):
            sample_heatmap = heatmaps[sample_idx]  # (C, T)
            sample_signal = signals[sample_idx] if signals is not None else None
            sample_label = labels[sample_idx].item()
            
            # Divide into non-overlapping windows
            window_relevances = []
            window_positions = []
            
            for win_idx in range(n_windows_per_timestep):
                start_pos = win_idx * self.window_size
                end_pos = min(start_pos + self.window_size, n_timesteps)
                
                # Compute window relevance (mean absolute relevance across channels and timesteps)
                window_heat = sample_heatmap[:, start_pos:end_pos]
                window_rel = torch.abs(window_heat).mean().item()
                
                window_relevances.append(window_rel)
                window_positions.append((start_pos, end_pos))
            
            # Get top-K windows by relevance
            window_relevances = np.array(window_relevances)
            top_k = min(n_top_windows, len(window_relevances))
            top_indices = np.argsort(window_relevances)[-top_k:][::-1]
            
            # Extract features for each top window
            for rank, win_idx in enumerate(top_indices):
                start_pos, end_pos = window_positions[win_idx]
                window_rel = window_relevances[win_idx]
                
                # Extract window from heatmap
                window_heat = sample_heatmap[:, start_pos:end_pos]
                
                # Extract features from signal if available
                if sample_signal is not None:
                    window_sig = sample_signal[:, start_pos:end_pos]
                    features = self._extract_window_features_from_signal(
                        window_sig, sample_rate
                    )
                else:
                    # Extract from heatmap as fallback
                    features = self._extract_window_features_from_heatmap(window_heat)
                
                # Add metadata
                window_info = {
                    'sample_idx': sample_idx,
                    'class_id': sample_label,
                    'window_idx': win_idx,
                    'rank': rank,
                    'start': start_pos,
                    'end': end_pos,
                    'position': start_pos / n_timesteps,  # Normalized position [0, 1]
                    'relevance': window_rel,
                    **features
                }
                
                all_windows.append(window_info)
        
        # Aggregate per-class statistics
        per_class_stats = self._compute_per_class_statistics(all_windows, labels)
        
        # Perform statistical tests between classes
        statistical_tests = self._perform_statistical_tests(all_windows, labels)
        
        print(f"\nExtracted {len(all_windows)} windows total")
        for class_id in torch.unique(labels):
            n_class_windows = sum(1 for w in all_windows if w['class_id'] == class_id.item())
            class_name = "OK" if class_id == 0 else "NOK"
            print(f"  Class {class_id} ({class_name}): {n_class_windows} windows")
        
        return {
            'windows': all_windows,
            'per_class_stats': per_class_stats,
            'statistical_tests': statistical_tests
        }
    
    def _extract_window_features_from_signal(
        self,
        window_signal: torch.Tensor,
        sample_rate: int = 400
    ) -> Dict[str, float]:
        """
        Extract vibration features from a signal window.
        
        Features match CNC thesis evaluation/time_faithfulness.py
        
        Args:
            window_signal: Signal window of shape (C, T)
            sample_rate: Sampling rate
            
        Returns:
            Dictionary of features
        """
        import scipy.stats as stats
        from scipy import signal as scipy_signal
        
        # Convert to numpy
        sig = window_signal.cpu().numpy()
        
        # Average across channels for simplicity (or could extract per-channel)
        sig_avg = sig.mean(axis=0)
        
        # Time-domain features
        avg_amplitude = np.abs(sig_avg).mean()
        max_amplitude = np.abs(sig_avg).max()
        std_amplitude = sig_avg.std()
        rms = np.sqrt(np.mean(sig_avg**2))
        
        # Statistical features
        skewness = float(stats.skew(sig_avg))
        kurtosis = float(stats.kurtosis(sig_avg))
        
        # Frequency-domain features
        if len(sig_avg) >= 4:
            freqs, psd = scipy_signal.welch(sig_avg, fs=sample_rate, nperseg=min(len(sig_avg), 256))
            peak_freq_idx = np.argmax(psd)
            peak_freq = freqs[peak_freq_idx]
            spectral_energy = np.sum(psd)
            
            # Spectral centroid
            spectral_centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
        else:
            peak_freq = 0.0
            spectral_energy = 0.0
            spectral_centroid = 0.0
        
        # Zero-crossing rate
        zero_crossings = np.sum(np.diff(np.sign(sig_avg)) != 0)
        zero_crossing_rate = zero_crossings / len(sig_avg)
        
        return {
            'avg_amplitude': float(avg_amplitude),
            'max_amplitude': float(max_amplitude),
            'std_amplitude': float(std_amplitude),
            'rms': float(rms),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'peak_freq': float(peak_freq),
            'spectral_energy': float(spectral_energy),
            'spectral_centroid': float(spectral_centroid),
            'zero_crossing_rate': float(zero_crossing_rate)
        }
    
    def _extract_window_features_from_heatmap(
        self,
        window_heatmap: torch.Tensor
    ) -> Dict[str, float]:
        """
        Extract basic features from heatmap window (fallback when no signal available).
        
        Args:
            window_heatmap: Heatmap window of shape (C, T)
            
        Returns:
            Dictionary of features
        """
        heat = window_heatmap.cpu().numpy()
        heat_avg = np.abs(heat).mean(axis=0)
        
        return {
            'avg_amplitude': float(heat_avg.mean()),
            'max_amplitude': float(heat_avg.max()),
            'std_amplitude': float(heat_avg.std()),
            'rms': float(np.sqrt(np.mean(heat_avg**2))),
            'skewness': float('nan'),   # Not computable from heatmap
            'kurtosis': float('nan'),
            'peak_freq': float('nan'),
            'spectral_energy': float('nan'),
            'spectral_centroid': float('nan'),
            'zero_crossing_rate': float('nan')
        }
    
    def _compute_per_class_statistics(
        self,
        all_windows: List[Dict],
        labels: torch.Tensor
    ) -> Dict[int, Dict]:
        """
        Compute aggregate statistics per class.
        
        Similar to CNC thesis Table 5.2
        """
        unique_classes = torch.unique(labels).cpu().numpy()
        per_class_stats = {}
        
        for class_id in unique_classes:
            class_windows = [w for w in all_windows if w['class_id'] == class_id]
            
            if len(class_windows) == 0:
                continue
            
            # Extract feature arrays
            feature_names = ['avg_amplitude', 'max_amplitude', 'std_amplitude', 'rms',
                           'skewness', 'kurtosis', 'peak_freq', 'spectral_energy',
                           'spectral_centroid', 'zero_crossing_rate', 'relevance']
            
            stats_dict = {}
            for feat in feature_names:
                if feat in class_windows[0]:
                    values = np.array([w[feat] for w in class_windows])
                    stats_dict[feat] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'median': float(np.median(values))
                    }
            
            per_class_stats[int(class_id)] = stats_dict
        
        return per_class_stats
    
    def _perform_statistical_tests(
        self,
        all_windows: List[Dict],
        labels: torch.Tensor
    ) -> Dict[str, Dict]:
        """
        Perform t-tests between classes for each feature.
        
        Returns p-values and effect sizes.
        """
        from scipy import stats
        
        unique_classes = torch.unique(labels).cpu().numpy()
        
        if len(unique_classes) < 2:
            return {}
        
        # Get windows for each class
        class_0_windows = [w for w in all_windows if w['class_id'] == unique_classes[0]]
        class_1_windows = [w for w in all_windows if w['class_id'] == unique_classes[1]]
        
        if len(class_0_windows) == 0 or len(class_1_windows) == 0:
            return {}
        
        # Test each feature
        feature_names = ['avg_amplitude', 'max_amplitude', 'std_amplitude', 'rms',
                        'skewness', 'kurtosis', 'peak_freq', 'spectral_energy',
                        'spectral_centroid', 'zero_crossing_rate', 'relevance']
        
        test_results = {}
        for feat in feature_names:
            if feat not in class_0_windows[0]:
                continue
            
            values_0 = np.array([w[feat] for w in class_0_windows], dtype=float)
            values_1 = np.array([w[feat] for w in class_1_windows], dtype=float)
            
            # Skip features that are entirely NaN (heatmap-only fallback)
            if np.all(np.isnan(values_0)) or np.all(np.isnan(values_1)):
                test_results[feat] = {
                    't_statistic': float('nan'),
                    'p_value': float('nan'),
                    'cohens_d': float('nan'),
                    'significant': False,
                    'class_0_mean': float('nan'),
                    'class_1_mean': float('nan')
                }
                continue
            
            # Remove NaN values before testing
            values_0 = values_0[~np.isnan(values_0)]
            values_1 = values_1[~np.isnan(values_1)]
            
            if len(values_0) < 2 or len(values_1) < 2:
                test_results[feat] = {
                    't_statistic': float('nan'),
                    'p_value': float('nan'),
                    'cohens_d': float('nan'),
                    'significant': False,
                    'class_0_mean': float(values_0.mean()) if len(values_0) else float('nan'),
                    'class_1_mean': float(values_1.mean()) if len(values_1) else float('nan')
                }
                continue
            
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(values_0, values_1)
            
            # Cohen's d effect size
            pooled_std = np.sqrt(((len(values_0) - 1) * values_0.std()**2 + 
                                  (len(values_1) - 1) * values_1.std()**2) / 
                                 (len(values_0) + len(values_1) - 2))
            cohens_d = (values_0.mean() - values_1.mean()) / (pooled_std + 1e-10)
            
            test_results[feat] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'cohens_d': float(cohens_d),
                'significant': bool(p_value < 0.05),
                'class_0_mean': float(values_0.mean()),
                'class_1_mean': float(values_1.mean())
            }
        
        return test_results


if __name__ == "__main__":
    # Test global window analysis on synthetic data
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Synthetic heatmaps: 100 samples, 3 channels, 2000 timesteps
    # Create two classes with different important regions
    n_samples = 100
    heatmaps_class0 = torch.randn(50, 3, 2000) * 0.1
    heatmaps_class1 = torch.randn(50, 3, 2000) * 0.1
    
    # Add important regions
    # Class 0: high importance at timesteps 400-480 and 1200-1280
    heatmaps_class0[:, :, 400:480] += torch.randn(50, 3, 80) * 2.0
    heatmaps_class0[:, :, 1200:1280] += torch.randn(50, 3, 80) * 1.5
    
    # Class 1: high importance at timesteps 800-880 and 1600-1680
    heatmaps_class1[:, :, 800:880] += torch.randn(50, 3, 80) * 2.0
    heatmaps_class1[:, :, 1600:1680] += torch.randn(50, 3, 80) * 1.5
    
    heatmaps = torch.cat([heatmaps_class0, heatmaps_class1])
    labels = torch.cat([torch.zeros(50), torch.ones(50)]).long()
    
    # Test global window analysis
    analyzer = GlobalWindowAnalysis(window_size=40, n_top_positions=5)
    important_windows = analyzer.find_important_windows(heatmaps, labels)
    
    print("\n" + "="*60)
    print("GLOBAL WINDOW ANALYSIS TEST")
    print("="*60)
    
    for class_id, windows in important_windows.items():
        print(f"\nClass {class_id} - Top {len(windows)} important windows:")
        for i, (start, end, importance) in enumerate(windows):
            print(f"  {i+1}. Timesteps {start:4d}-{end:4d}: importance={importance:.6f}")
    
    # Test coverage
    coverage = analyzer.get_window_coverage_per_sample(heatmaps, labels)
    for class_id, cov in coverage.items():
        print(f"\nClass {class_id} - Window coverage per sample:")
        print(f"  Mean: {cov.mean():.4f}")
        print(f"  Std:  {cov.std():.4f}")
        print(f"  Min:  {cov.min():.4f}")
        print(f"  Max:  {cov.max():.4f}")
    
    print("\n✓ Global window analysis test passed!")
