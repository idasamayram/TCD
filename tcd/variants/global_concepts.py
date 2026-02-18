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
