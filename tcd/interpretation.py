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


# Class name mapping (can be customized per dataset)
DEFAULT_CLASS_NAMES = {0: "OK", 1: "NOK"}


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
        global_windows: Dict[int, List[Tuple[int, int, float]]],
        heatmaps: Optional[torch.Tensor] = None,
        signals: Optional[torch.Tensor] = None,
        top_k_filters: int = 10
    ) -> Dict[int, Dict[int, Dict[str, Any]]]:
        """
        Generate interpretations for all prototypes.
        
        Args:
            global_windows: Important window positions from GlobalWindowAnalysis
                           Format: {class_id: [(start, end, importance), ...]}
            heatmaps: Optional heatmap relevances of shape (N, C, T)
            signals: Optional raw signals of shape (N, C, T)
            top_k_filters: Number of top filters to analyze per prototype
            
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
                        'window_features': Optional[Dict]
                    }
                }
            }
        """
        interpretations = {}
        
        for class_id, gmm in self.gmms.items():
            class_mask = self.labels == class_id
            class_features = self.features[class_mask]  # (N_class, n_filters)
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
                
                # Analyze window features if provided
                window_features = None
                if heatmaps is not None and class_id in global_windows:
                    window_features = self._analyze_window_features(
                        heatmaps[class_mask][proto_mask],
                        global_windows[class_id],
                        signals[class_mask][proto_mask] if signals is not None else None
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
        window_positions: List[Tuple[int, int, float]],
        proto_signals: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Analyze vibration features at globally-important windows for this prototype.
        
        Args:
            proto_heatmaps: Heatmaps for samples assigned to this prototype (N_proto, C, T)
            window_positions: Global window positions [(start, end, importance), ...]
            proto_signals: Optional raw signals (N_proto, C, T)
            
        Returns:
            Dictionary with feature statistics
        """
        if proto_heatmaps.shape[0] == 0:
            return None
        
        # Use top 3 windows for analysis
        top_windows = window_positions[:3]
        
        window_stats = []
        for win_idx, (start, end, importance) in enumerate(top_windows):
            window_heatmaps = proto_heatmaps[:, :, start:end]
            
            # Compute simple statistics
            stats = {
                'window_idx': win_idx,
                'start': start,
                'end': end,
                'global_importance': importance,
                'mean_relevance': torch.abs(window_heatmaps).mean().item(),
                'max_relevance': torch.abs(window_heatmaps).max().item(),
                'std_relevance': torch.abs(window_heatmaps).std().item()
            }
            
            # If signals provided, extract basic features
            if proto_signals is not None:
                window_signals = proto_signals[:, :, start:end]
                
                # RMS
                rms = torch.sqrt((window_signals ** 2).mean(dim=(1, 2))).mean().item()
                stats['rms'] = rms
                
                # Peak amplitude
                peak = torch.abs(window_signals).max(dim=2)[0].mean().item()
                stats['peak_amplitude'] = peak
                
                # Crest factor
                stats['crest_factor'] = peak / (rms + 1e-8)
            
            window_stats.append(stats)
        
        return {
            'n_windows_analyzed': len(top_windows),
            'window_stats': window_stats
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
            window_features: Optional window feature analysis
            
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
        
        # Add window feature information if available
        if window_features and window_features.get('window_stats'):
            window_stats = window_features['window_stats'][0]  # Use first window
            description_parts.append(
                f"- Primary temporal region: timesteps {window_stats['start']}-{window_stats['end']} "
                f"(mean relevance: {window_stats['mean_relevance']:.4f})"
            )
            
            if 'rms' in window_stats:
                description_parts.append(
                    f"- Signal characteristics: RMS={window_stats['rms']:.4f}, "
                    f"peak={window_stats.get('peak_amplitude', 0):.4f}, "
                    f"crest_factor={window_stats.get('crest_factor', 0):.2f}"
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
                        print(f"\n  Window Analysis ({wf['n_windows_analyzed']} windows):")
                        for ws in wf['window_stats'][:2]:  # Show top 2 windows
                            print(f"    Window {ws['window_idx']} (timesteps {ws['start']}-{ws['end']}):")
                            print(f"      Global importance: {ws['global_importance']:.6f}")
                            print(f"      Mean relevance: {ws['mean_relevance']:.6f}")
                            if 'rms' in ws:
                                print(f"      RMS: {ws['rms']:.4f}, Crest factor: {ws.get('crest_factor', 0):.2f}")
                
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
    
    # Create synthetic global windows
    global_windows = {
        0: [(400, 440, 0.85), (1200, 1240, 0.72), (800, 840, 0.65)],
        1: [(800, 840, 0.91), (1600, 1640, 0.78), (400, 440, 0.60)]
    }
    
    # Test interpreter
    interpreter = ConceptInterpreter(gmms, features, labels, layer_name="conv3")
    interpretations = interpreter.interpret_prototypes(
        global_windows=global_windows,
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
