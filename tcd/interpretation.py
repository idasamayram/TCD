"""
Concept interpretation for CRP-native prototypes.

After GMM prototypes are discovered in the CRP filter space,
this module answers: "What does each prototype mean in human-understandable terms?"

The interpretation pipeline:
1. Identify top-k most important filters for each prototype (from prototype center μ)
2. Find globally-important time windows from GlobalWindowAnalysis
3. Extract vibration features at those positions
4. Compare features between OK/NOK samples assigned to this prototype
5. Generate human-readable descriptions

This uses vibration features for INTERPRETATION of CRP concepts,
not as concept definitions themselves.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.mixture import GaussianMixture
import scipy.stats as stats
from scipy import signal as scipy_signal
import warnings


# Class name mapping (can be customized per dataset)
DEFAULT_CLASS_NAMES = {0: "OK", 1: "NOK"}

# Constants for feature extraction
EPSILON_SPECTRAL = 1e-10  # Small epsilon to prevent division by zero in spectral features


def get_class_name(class_id: int, class_names: Optional[Dict[int, str]] = None) -> str:
    """
    Get human-readable class name for a class ID.
    
    Args:
        class_id: Class ID
        class_names: Optional custom class name mapping
        
    Returns:
        Class name string
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES
    return class_names.get(class_id, f"Class_{class_id}")


def extract_sample_features(
    heatmap: torch.Tensor,
    signal: torch.Tensor,
    window_size: int = 40,
    n_top_windows: int = 10,
    sample_rate: int = 400
) -> List[Dict[str, Any]]:
    """
    Extract features from a single sample's high-relevance windows.
    
    CNC-style per-sample extraction matching idasamayram/CNC/evaluation/time_faithfulness.py.
    
    Algorithm:
    1. For each sample, compute per-channel, per-window mean absolute relevance
    2. Flatten to (n_channels × n_windows_per_channel) and sort descending
    3. Take top-K windows (with channel info preserved)
    4. Extract features per window: avg_amplitude, max_amplitude, std_amplitude,
       skewness, kurtosis, peak_freq, spectral_energy, etc.
    
    Args:
        heatmap: Heatmap relevance of shape (C, T) for a single sample
        signal: Raw signal of shape (C, T) for a single sample
        window_size: Window size in timesteps
        n_top_windows: Number of top windows to extract
        sample_rate: Sampling rate in Hz
        
    Returns:
        List of dictionaries, each containing:
        {
            'window_idx': int,
            'channel_idx': int,
            'start': int,
            'end': int,
            'relevance': float,
            'avg_amplitude': float,
            'max_amplitude': float,
            'std_amplitude': float,
            'skewness': float,
            'kurtosis': float,
            'peak_freq': float,
            'spectral_energy': float,
            'spectral_centroid': float,
            'zero_crossing_rate': float
        }
    """
    n_channels, n_timesteps = heatmap.shape
    n_windows = n_timesteps // window_size
    
    # Compute relevance per window per channel
    window_relevances = []
    window_info = []
    
    for channel_idx in range(n_channels):
        for win_idx in range(n_windows):
            start_pos = win_idx * window_size
            end_pos = min(start_pos + window_size, n_timesteps)
            
            # Compute window relevance (mean absolute relevance)
            window_heat = heatmap[channel_idx, start_pos:end_pos]
            window_rel = torch.abs(window_heat).mean().item()
            
            window_relevances.append(window_rel)
            window_info.append({
                'channel_idx': channel_idx,
                'window_idx': win_idx,
                'start': start_pos,
                'end': end_pos
            })
    
    # Get top-K windows
    window_relevances = np.array(window_relevances)
    top_k = min(n_top_windows, len(window_relevances))
    top_indices = np.argsort(window_relevances)[-top_k:][::-1]
    
    # Extract features for each top window
    window_features = []
    
    for idx in top_indices:
        info = window_info[idx]
        channel_idx = info['channel_idx']
        start_pos = info['start']
        end_pos = info['end']
        relevance = window_relevances[idx]
        
        # Extract signal window
        window_sig = signal[channel_idx, start_pos:end_pos].cpu().numpy()
        
        # Time-domain features
        avg_amplitude = np.abs(window_sig).mean()
        max_amplitude = np.abs(window_sig).max()
        std_amplitude = window_sig.std()
        rms = np.sqrt(np.mean(window_sig**2))
        
        # Statistical features
        skewness = float(stats.skew(window_sig)) if len(window_sig) > 2 else 0.0
        kurtosis = float(stats.kurtosis(window_sig)) if len(window_sig) > 2 else 0.0
        
        # Frequency-domain features
        if len(window_sig) >= 4:
            try:
                freqs, psd = scipy_signal.welch(
                    window_sig, 
                    fs=sample_rate, 
                    nperseg=min(len(window_sig), 256)
                )
                peak_freq_idx = np.argmax(psd)
                peak_freq = freqs[peak_freq_idx]
                spectral_energy = np.sum(psd)
                spectral_centroid = np.sum(freqs * psd) / (np.sum(psd) + EPSILON_SPECTRAL)
            except (ValueError, ZeroDivisionError) as e:
                # Welch computation can fail on certain signal properties
                peak_freq = 0.0
                spectral_energy = 0.0
                spectral_centroid = 0.0
        else:
            peak_freq = 0.0
            spectral_energy = 0.0
            spectral_centroid = 0.0
        
        # Zero-crossing rate
        zero_crossings = np.sum(np.diff(np.sign(window_sig)) != 0)
        zero_crossing_rate = zero_crossings / len(window_sig) if len(window_sig) > 1 else 0.0
        
        window_features.append({
            'window_idx': info['window_idx'],
            'channel_idx': channel_idx,
            'start': start_pos,
            'end': end_pos,
            'relevance': float(relevance),
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
        })
    
    return window_features


class ConceptInterpreter:
    """
    Interpret CRP concept prototypes by analyzing what they respond to.
    
    Takes GMM prototypes fitted in CRP filter space and generates
    human-readable interpretations using global window analysis
    and vibration features.
    
    Usage:
        interpreter = ConceptInterpreter(gmms, features, labels)
        interpretations = interpreter.interpret_prototypes(
            global_windows=important_window_positions,
            signals=raw_signals
        )
    """
    
    def __init__(
        self,
        gmms: Dict[int, GaussianMixture],
        features: torch.Tensor,
        labels: torch.Tensor,
        layer_name: str = "conv3",
        class_names: Optional[Dict[int, str]] = None
    ):
        """
        Initialize concept interpreter.
        
        Args:
            gmms: Dictionary mapping class_id -> fitted GaussianMixture
            features: CRP concept relevances of shape (N, n_filters)
            labels: Class labels of shape (N,)
            layer_name: Name of the layer (for reporting)
            class_names: Optional custom class name mapping (default: {0: "OK", 1: "NOK"})
        """
        self.gmms = gmms
        self.features = features
        self.labels = labels
        self.layer_name = layer_name
        self.n_filters = features.shape[1]
        self.class_names = class_names or DEFAULT_CLASS_NAMES
    
    def interpret_prototypes(
        self,
        heatmaps: torch.Tensor,
        signals: torch.Tensor,
        window_size: int = 40,
        n_top_windows: int = 10,
        sample_rate: int = 400,
        top_k_filters: int = 10,
        global_windows: Optional[Dict[int, List[Tuple[int, int, float]]]] = None
    ) -> Dict[int, Dict[int, Dict[str, Any]]]:
        """
        Generate interpretations for all prototypes.
        
        Uses CNC-style per-sample feature extraction from high-relevance windows.
        The global_windows parameter is now optional and ignored - feature extraction
        is done per-sample as proven effective in idasamayram/CNC.
        
        Args:
            heatmaps: Heatmap relevances of shape (N, C, T)
            signals: Raw signals of shape (N, C, T)
            window_size: Window size in timesteps for per-sample extraction
            n_top_windows: Number of top windows to extract per sample
            sample_rate: Sampling rate in Hz
            top_k_filters: Number of top filters to analyze per prototype
            global_windows: Optional (deprecated) - kept for backward compatibility
            
        Returns:
            Dictionary structure:
            {
                class_id: {
                    prototype_idx: {
                        'top_filters': [(filter_idx, weight), ...],
                        'n_samples': int,
                        'coverage': float,
                        'description': str,
                        'filter_summary': str,
                        'window_features': Dict with aggregated per-sample features
                    }
                }
            }
        """
        # Warn if deprecated parameter is used
        if global_windows is not None:
            warnings.warn(
                "The 'global_windows' parameter is deprecated and will be removed in a future version. "
                "Feature extraction now uses per-sample high-relevance windows instead of global positions.",
                DeprecationWarning,
                stacklevel=2
            )
        
        interpretations = {}
        
        for class_id, gmm in self.gmms.items():
            class_mask = self.labels == class_id
            class_features = self.features[class_mask]  # (N_class, n_filters)
            class_heatmaps = heatmaps[class_mask]  # (N_class, C, T)
            class_signals = signals[class_mask]  # (N_class, C, T)
            n_class_samples = class_features.shape[0]
            
            # Assign samples to prototypes
            assignments = gmm.predict(class_features.cpu().numpy())
            
            class_interpretations = {}
            
            for proto_idx in range(gmm.n_components):
                # Get samples assigned to this prototype
                proto_mask = assignments == proto_idx
                n_proto_samples = proto_mask.sum()
                coverage = n_proto_samples / n_class_samples
                
                # Get prototype mean (filter weights)
                proto_mean = gmm.means_[proto_idx]  # (n_filters,)
                
                # Identify top filters
                top_filter_indices = np.argsort(np.abs(proto_mean))[-top_k_filters:][::-1]
                top_filters = [
                    (int(idx), float(proto_mean[idx]))
                    for idx in top_filter_indices
                ]
                
                # Generate filter summary
                filter_summary = self._generate_filter_summary(top_filters, self.layer_name)
                
                # Analyze window features using per-sample extraction
                window_features = None
                if n_proto_samples > 0:
                    proto_heatmaps = class_heatmaps[proto_mask]
                    proto_signals = class_signals[proto_mask]
                    
                    window_features = self._analyze_window_features(
                        proto_heatmaps,
                        proto_signals,
                        window_size,
                        n_top_windows,
                        sample_rate
                    )
                
                # Generate human-readable description
                description = self._generate_description(
                    class_id=class_id,
                    proto_idx=proto_idx,
                    top_filters=top_filters,
                    n_samples=n_proto_samples,
                    coverage=coverage,
                    window_features=window_features
                )
                
                class_interpretations[proto_idx] = {
                    'top_filters': top_filters,
                    'n_samples': int(n_proto_samples),
                    'coverage': float(coverage),
                    'description': description,
                    'filter_summary': filter_summary,
                    'window_features': window_features
                }
            
            interpretations[class_id] = class_interpretations
        
        return interpretations
    
    def _generate_filter_summary(
        self,
        top_filters: List[Tuple[int, float]],
        layer_name: str
    ) -> str:
        """
        Generate a textual summary of top filters.
        
        Args:
            top_filters: List of (filter_idx, weight) tuples
            layer_name: Name of the layer
            
        Returns:
            Summary string
        """
        filter_strs = []
        for idx, weight in top_filters[:5]:  # Top 5 for brevity
            filter_strs.append(f"{layer_name}[{idx}]={weight:.3f}")
        
        return ", ".join(filter_strs)
    
    def _analyze_window_features(
        self,
        proto_heatmaps: torch.Tensor,
        proto_signals: torch.Tensor,
        window_size: int,
        n_top_windows: int,
        sample_rate: int
    ) -> Dict[str, Any]:
        """
        Analyze vibration features using CNC-style per-sample extraction.
        
        For each sample assigned to this prototype:
        1. Extract features from its own top-K high-relevance windows
        2. Aggregate statistics across all samples in the prototype
        
        This matches the proven approach from idasamayram/CNC/evaluation/time_faithfulness.py
        
        Args:
            proto_heatmaps: Heatmaps for samples assigned to this prototype (N_proto, C, T)
            proto_signals: Raw signals for samples assigned to this prototype (N_proto, C, T)
            window_size: Window size in timesteps
            n_top_windows: Number of top windows to extract per sample
            sample_rate: Sampling rate in Hz
            
        Returns:
            Dictionary with aggregated feature statistics:
            {
                'n_samples': int,
                'n_windows_per_sample': int,
                'feature_stats': {
                    'avg_amplitude': {'mean': float, 'std': float},
                    'max_amplitude': {...},
                    ...
                }
            }
        """
        if proto_heatmaps.shape[0] == 0:
            return None
        
        n_samples = proto_heatmaps.shape[0]
        
        # Collect features from all samples
        all_window_features = []
        
        for sample_idx in range(n_samples):
            sample_heatmap = proto_heatmaps[sample_idx]  # (C, T)
            sample_signal = proto_signals[sample_idx]  # (C, T)
            
            # Extract features from this sample's high-relevance windows
            sample_windows = extract_sample_features(
                sample_heatmap,
                sample_signal,
                window_size,
                n_top_windows,
                sample_rate
            )
            
            all_window_features.extend(sample_windows)
        
        # Aggregate statistics across all windows
        if len(all_window_features) == 0:
            return None
        
        # Compute mean and std for each feature
        feature_names = [
            'avg_amplitude', 'max_amplitude', 'std_amplitude', 'rms',
            'skewness', 'kurtosis', 'peak_freq', 'spectral_energy',
            'spectral_centroid', 'zero_crossing_rate', 'relevance'
        ]
        
        feature_stats = {}
        for feature_name in feature_names:
            values = [w[feature_name] for w in all_window_features]
            feature_stats[feature_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        return {
            'n_samples': n_samples,
            'n_windows_per_sample': n_top_windows,
            'total_windows_analyzed': len(all_window_features),
            'feature_stats': feature_stats
        }
    
    def _generate_description(
        self,
        class_id: int,
        proto_idx: int,
        top_filters: List[Tuple[int, float]],
        n_samples: int,
        coverage: float,
        window_features: Optional[Dict[str, Any]]
    ) -> str:
        """
        Generate human-readable description of a prototype.
        
        Args:
            class_id: Class ID
            proto_idx: Prototype index within class
            top_filters: Top filter contributions
            n_samples: Number of samples assigned to this prototype
            coverage: Fraction of class samples assigned to this prototype
            window_features: Optional per-sample window feature analysis
            
        Returns:
            Human-readable description string
        """
        class_name = get_class_name(class_id, self.class_names)
        
        description_parts = [
            f"Prototype {proto_idx} for {class_name} (Class {class_id})",
            f"- Covers {coverage*100:.1f}% of {class_name} samples ({n_samples} samples)",
            f"- Top contributing filters: {', '.join([f'filter_{idx}' for idx, _ in top_filters[:5]])}",
        ]
        
        # Add filter weight information
        pos_filters = [idx for idx, w in top_filters if w > 0]
        neg_filters = [idx for idx, w in top_filters if w < 0]
        
        if pos_filters:
            description_parts.append(f"- Positive contributions from filters: {pos_filters[:3]}")
        if neg_filters:
            description_parts.append(f"- Negative contributions from filters: {neg_filters[:3]}")
        
        # Add per-sample window feature information if available
        if window_features:
            n_samples_analyzed = window_features['n_samples']
            n_windows = window_features['total_windows_analyzed']
            feature_stats = window_features['feature_stats']
            
            description_parts.append(
                f"- Analyzed {n_windows} high-relevance windows from {n_samples_analyzed} samples"
            )
            
            # Add key signal characteristics
            if 'rms' in feature_stats:
                rms_mean = feature_stats['rms']['mean']
                peak_mean = feature_stats['max_amplitude']['mean']
                description_parts.append(
                    f"- Signal characteristics: RMS={rms_mean:.4f}, peak={peak_mean:.4f}"
                )
            
            if 'peak_freq' in feature_stats:
                freq_mean = feature_stats['peak_freq']['mean']
                freq_std = feature_stats['peak_freq']['std']
                description_parts.append(
                    f"- Dominant frequency: {freq_mean:.1f} ± {freq_std:.1f} Hz"
                )
            
            if 'kurtosis' in feature_stats:
                kurt_mean = feature_stats['kurtosis']['mean']
                description_parts.append(
                    f"- Kurtosis: {kurt_mean:.2f} (>3 indicates impulse-like vibration)"
                )
        
        return "\n".join(description_parts)
    
    def print_interpretations(
        self,
        interpretations: Dict[int, Dict[int, Dict[str, Any]]],
        verbose: bool = True
    ):
        """
        Pretty-print prototype interpretations.
        
        Args:
            interpretations: Output from interpret_prototypes()
            verbose: If True, print detailed information
        """
        print("\n" + "="*80)
        print("PROTOTYPE INTERPRETATIONS")
        print("="*80)
        
        for class_id in sorted(interpretations.keys()):
            class_name = get_class_name(class_id, self.class_names)
            print(f"\n{'='*80}")
            print(f"CLASS {class_id} ({class_name})")
            print(f"{'='*80}")
            
            class_interp = interpretations[class_id]
            
            for proto_idx in sorted(class_interp.keys()):
                proto = class_interp[proto_idx]
                
                print(f"\n{proto['description']}")
                
                if verbose:
                    print(f"\n  Filter Contributions:")
                    for i, (filter_idx, weight) in enumerate(proto['top_filters'][:5]):
                        print(f"    {i+1}. Filter {filter_idx:3d}: {weight:+.4f}")
                    
                    if proto.get('window_features'):
                        wf = proto['window_features']
                        print(f"\n  Per-Sample Window Analysis:")
                        print(f"    Analyzed {wf['total_windows_analyzed']} windows from {wf['n_samples']} samples")
                        print(f"    ({wf['n_windows_per_sample']} top windows per sample)")
                        
                        # Show key feature statistics
                        feature_stats = wf['feature_stats']
                        print(f"\n  Signal Statistics (across all high-relevance windows):")
                        
                        for feature_name in ['avg_amplitude', 'max_amplitude', 'rms', 
                                            'peak_freq', 'kurtosis', 'skewness']:
                            if feature_name in feature_stats:
                                stats = feature_stats[feature_name]
                                print(f"    {feature_name}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                                     f"(range: [{stats['min']:.4f}, {stats['max']:.4f}])")
                
                print()
        
        print("="*80 + "\n")
    
    def compare_prototypes_between_classes(
        self,
        interpretations: Dict[int, Dict[int, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Compare prototypes between OK and NOK classes.
        
        Args:
            interpretations: Output from interpret_prototypes()
            
        Returns:
            Comparison statistics
        """
        if 0 not in interpretations or 1 not in interpretations:
            return {}
        
        ok_prototypes = interpretations[0]
        nok_prototypes = interpretations[1]
        
        # Compare filter usage
        ok_filters = set()
        for proto in ok_prototypes.values():
            ok_filters.update([idx for idx, _ in proto['top_filters']])
        
        nok_filters = set()
        for proto in nok_prototypes.values():
            nok_filters.update([idx for idx, _ in proto['top_filters']])
        
        shared_filters = ok_filters & nok_filters
        ok_only_filters = ok_filters - nok_filters
        nok_only_filters = nok_filters - ok_filters
        
        comparison = {
            'ok_n_prototypes': len(ok_prototypes),
            'nok_n_prototypes': len(nok_prototypes),
            'ok_unique_filters': len(ok_filters),
            'nok_unique_filters': len(nok_filters),
            'shared_filters': list(shared_filters),
            'ok_only_filters': list(ok_only_filters),
            'nok_only_filters': list(nok_only_filters),
            'filter_overlap_ratio': len(shared_filters) / max(len(ok_filters | nok_filters), 1)
        }
        
        return comparison
    
    def export_to_dict(
        self,
        interpretations: Dict[int, Dict[int, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Export interpretations to a serializable dictionary.
        
        Args:
            interpretations: Output from interpret_prototypes()
            
        Returns:
            Dictionary suitable for JSON/pickle serialization
        """
        return {
            'layer_name': self.layer_name,
            'n_filters': self.n_filters,
            'interpretations': interpretations,
            'comparison': self.compare_prototypes_between_classes(interpretations)
        }


if __name__ == "__main__":
    # Test concept interpreter on synthetic data
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Synthetic CRP features: 100 samples, 64 filters (like conv3)
    n_filters = 64
    features_class0 = torch.randn(50, n_filters) + torch.randn(n_filters) * 2
    features_class1 = torch.randn(50, n_filters) + torch.randn(n_filters) * 2
    features = torch.cat([features_class0, features_class1])
    labels = torch.cat([torch.zeros(50), torch.ones(50)]).long()
    
    # Synthetic heatmaps and signals
    n_samples = 100
    n_channels = 3
    n_timesteps = 2000
    heatmaps = torch.randn(n_samples, n_channels, n_timesteps)
    signals = torch.randn(n_samples, n_channels, n_timesteps)
    
    # Fit GMMs
    gmms = {}
    for class_id in [0, 1]:
        class_mask = labels == class_id
        class_features = features[class_mask].cpu().numpy()
        
        gmm = GaussianMixture(
            n_components=2,
            covariance_type='diag',
            n_init=3,
            max_iter=100,
            random_state=42
        )
        gmm.fit(class_features)
        gmms[class_id] = gmm
    
    # Test interpreter with new per-sample extraction API
    interpreter = ConceptInterpreter(gmms, features, labels, layer_name="conv3")
    interpretations = interpreter.interpret_prototypes(
        heatmaps=heatmaps,
        signals=signals,
        window_size=40,
        n_top_windows=10,
        sample_rate=400,
        top_k_filters=10
    )
    
    # Print results
    interpreter.print_interpretations(interpretations, verbose=True)
    
    # Compare classes
    comparison = interpreter.compare_prototypes_between_classes(interpretations)
    print("\nCLASS COMPARISON:")
    print(f"  OK prototypes: {comparison['ok_n_prototypes']}")
    print(f"  NOK prototypes: {comparison['nok_n_prototypes']}")
    print(f"  Shared filters: {len(comparison['shared_filters'])}")
    print(f"  OK-only filters: {len(comparison['ok_only_filters'])}")
    print(f"  NOK-only filters: {len(comparison['nok_only_filters'])}")
    print(f"  Filter overlap: {comparison['filter_overlap_ratio']:.2%}")
    
    print("\n✓ Concept interpreter test passed!")
