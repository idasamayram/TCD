#!/usr/bin/env python3
"""
Step 2: Discover Concepts - Run TCD variant pipeline.

Loads pre-computed CRP features and applies TCD variant (A/B/C)
to discover temporal concepts.

Usage:
    # Variant A: Filterbank
    python scripts/discover_concepts.py --variant A --features results/crp_features --output results/concepts_A
    
    # Variant C: Learned clusters (requires features)
    python scripts/discover_concepts.py --variant C --features results/crp_features --output results/concepts_C
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os
import yaml
import torch
import h5py
import numpy as np
import pickle
from pathlib import Path

# from models.cnn1d_model import CNN1D_Wide
from tcd.variants.filterbank import FilterBankTCD, WindowConceptTCD
from tcd.variants.temporal_descriptors import TemporalDescriptorTCD
from tcd.variants.learned_clusters import LearnedClusterTCD
from tcd.variants.vibration_features import VibrationFeatureTCD
from tcd.visualization import plot_prototype_grid, plot_concept_relevance


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_variant_a(
    features_path: str,
    output_path: str,
    config: dict,
    use_window_based: bool = False
):
    """
    Run Variant A: Filterbank or window-based concepts.
    
    Args:
        features_path: Path to CRP features directory
        output_path: Output directory
        config: Configuration dict
        use_window_based: If True, use WindowConceptTCD; else use FilterBankTCD
    """
    print("\n" + "="*60)
    if use_window_based:
        print("VARIANT A: Window-Based Concept Discovery (Data-Driven)")
    else:
        print("VARIANT A: Frequency-Band Filterbank Concepts")
    print("="*60)
    
    # Load heatmaps
    print("Loading heatmaps...")
    heatmaps_list = []
    labels_list = []
    
    for class_id in [0, 1]:
        heatmaps_path = os.path.join(features_path, f"heatmaps_class_{class_id}.hdf5")
        if not os.path.exists(heatmaps_path):
            print(f"Warning: Heatmaps not found at {heatmaps_path}")
            continue
        
        with h5py.File(heatmaps_path, 'r') as f:
            heatmaps = np.array(f['heatmaps'])
            heatmaps_list.append(heatmaps)
            labels_list.extend([class_id] * len(heatmaps))
            print(f"  Class {class_id}: {heatmaps.shape}")
    
    if not heatmaps_list:
        print("Error: No heatmaps found. Run run_analysis.py first.")
        return
    
    heatmaps = torch.from_numpy(np.concatenate(heatmaps_list)).float()
    labels = np.array(labels_list)
    
    # Initialize TCD variant
    sample_rate = config['data']['sample_rate']
    
    if use_window_based:
        # Window-based concept discovery
        window_config = config['tcd'].get('window_concept', {})
        n_concepts = config['tcd'].get('n_concepts', 6)
        
        tcd = WindowConceptTCD(
            n_concepts=n_concepts,
            window_size=window_config.get('window_size', 40),
            n_top_windows=window_config.get('n_top_windows', None),  # None = adaptive
            threshold_factor=window_config.get('threshold_factor', 1.0),
            sample_rate=sample_rate,
            features=window_config.get('features', None),  # None = use all features
            use_raw_signal=window_config.get('use_raw_signal', False),
            gmm_covariance=window_config.get('gmm_covariance', 'full'),
            gmm_n_init=window_config.get('gmm_n_init', 10),
            gmm_max_iter=window_config.get('gmm_max_iter', 100)
        )
        
        print(f"\nWindow-based concept discovery with {n_concepts} concepts")
        print(f"  Window size: {tcd.window_size} timesteps")
        if tcd.n_top_windows is None:
            print(f"  Adaptive threshold mode (threshold_factor={tcd.threshold_factor})")
        else:
            print(f"  Top windows per sample: {tcd.n_top_windows}")
        print(f"  Use raw signal: {tcd.use_raw_signal}")
        print(f"  Number of features: {len(tcd.features)}")
        print(f"  Features: {', '.join(tcd.features[:5])}{'...' if len(tcd.features) > 5 else ''}")
        
        # Fit GMM on training data
        print("\nFitting GMM to discover concepts...")
        tcd.fit(heatmaps, labels=torch.from_numpy(labels))
        
    else:
        # Filterbank-based concepts (legacy)
        bands = config['tcd']['filterbank_bands']
        tcd = FilterBankTCD(bands=bands, sample_rate=sample_rate)
        print(f"\nFilterbank with {len(bands)} bands:")
        for i, label in enumerate(tcd.get_concept_labels()):
            print(f"  Concept {i}: {label}")
    
    # Extract concepts
    print("\nExtracting concepts...")
    concept_relevances = tcd.extract_concepts(heatmaps)
    print(f"Concept relevances shape: {concept_relevances.shape}")
    
    # Get concept labels
    concept_labels = tcd.get_concept_labels()
    print(f"\nDiscovered concepts:")
    for i, label in enumerate(concept_labels):
        print(f"  {label}")
    
    # Compute importance per concept (overall and per-class)
    importance = tcd.compute_concept_importance(heatmaps)
    
    # Separate by class
    class_0_mask = np.array(labels) == 0
    class_1_mask = np.array(labels) == 1
    heatmaps_class_0 = heatmaps[class_0_mask]
    heatmaps_class_1 = heatmaps[class_1_mask]
    
    importance_class_0 = tcd.compute_concept_importance(heatmaps_class_0)
    importance_class_1 = tcd.compute_concept_importance(heatmaps_class_1)
    
    print("\n" + "="*60)
    print("CONCEPT IMPORTANCE (Overall and Per-Class)")
    print("="*60)
    
    # Dynamic column width based on max label length
    max_label_len = max(len(label) for label in concept_labels)
    col_width = max(40, max_label_len + 2)  # At least 40, or longer if needed
    
    print(f"{'Concept':<{col_width}} {'Overall':<15} {'OK (Class 0)':<15} {'NOK (Class 1)':<15} {'Ratio (NOK/OK)':<15}")
    print("-" * (col_width + 60))
    for i, label in enumerate(concept_labels):
        if importance_class_0[i] > 0:
            ratio = importance_class_1[i] / importance_class_0[i]
            ratio_str = f"{ratio:>14.2f}x"
        else:
            ratio_str = "N/A".rjust(15)
        print(f"{label:<{col_width}} {importance[i]:>14.4f} {importance_class_0[i]:>14.4f} {importance_class_1[i]:>14.4f} {ratio_str}")
    print("=" * (col_width + 60) + "\n")
    
    # Save results
    os.makedirs(output_path, exist_ok=True)
    
    results = {
        'variant': 'A',
        'method': 'window_based' if use_window_based else 'filterbank',
        'concept_labels': concept_labels,
        'concept_relevances': concept_relevances.numpy(),
        'labels': labels,
        'importance': importance,
        'importance_class_0': importance_class_0,
        'importance_class_1': importance_class_1
    }
    
    if not use_window_based:
        results['bands'] = config['tcd']['filterbank_bands']
    
    with open(os.path.join(output_path, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✓ Results saved to {output_path}")


def run_variant_c(
    features_path: str,
    output_path: str,
    config: dict,
    layer_name: str = None,  # None = use config, otherwise override
    data_path: str = None  # Path to dataset for loading class weights
):
    """
    Run Variant C: CRP-native concepts with GMM prototypes (PRIMARY METHOD).
    
    This is the correct approach:
    1. CRP filter relevances ARE the concept space
    2. GMM clustering in CRP space finds prototypes
    3. Global window analysis identifies important temporal regions
    4. Interpretation pipeline explains what prototypes mean
    
    Args:
        features_path: Path to CRP features directory
        output_path: Output directory
        config: Configuration dict
        layer_name: Layer to use for concepts (None = use config['tcd']['primary_layer'])
        data_path: Path to dataset for loading class weights (optional)
    """
    print("\n" + "="*80)
    print("VARIANT C: CRP-Native Concepts with GMM Prototypes (PRIMARY METHOD)")
    print("="*80)
    
    # Get layer name from config if not specified
    if layer_name is None:
        layer_name = config['tcd'].get('primary_layer', 'conv3')
    
    # Load concept features (CRP filter relevances)
    print(f"\nStep 1: Loading CRP concept relevances for layer {layer_name}...")
    features_list = []
    labels_list = []
    outputs_list = []
    heatmaps_list = []
    
    for class_id in [0, 1]:
        # Load concept relevances
        h5_path = os.path.join(features_path, f"eps_relevances_class_{class_id}.hdf5")
        if not os.path.exists(h5_path):
            print(f"Warning: Features not found at {h5_path}")
            continue
        
        with h5py.File(h5_path, 'r') as f:
            if layer_name not in f:
                print(f"Warning: Layer {layer_name} not in features file")
                print(f"Available layers: {list(f.keys())}")
                continue
            features = np.array(f[layer_name])
            features_list.append(features)
            labels_list.extend([class_id] * len(features))
        
        # Load outputs
        outputs_path = os.path.join(features_path, f"outputs_class_{class_id}.pt")
        outputs = torch.load(outputs_path)
        outputs_list.extend(outputs)
        
        # Load heatmaps for global window analysis
        heatmaps_path = os.path.join(features_path, f"heatmaps_class_{class_id}.hdf5")
        if os.path.exists(heatmaps_path):
            with h5py.File(heatmaps_path, 'r') as f:
                heatmaps = np.array(f['heatmaps'])
                heatmaps_list.append(heatmaps)
        
        print(f"  Class {class_id}: {features.shape}")
    
    if not features_list:
        print("Error: No features found. Run run_analysis.py first.")
        return
    
    features = torch.from_numpy(np.concatenate(features_list)).float()
    labels = torch.tensor(labels_list).long()
    outputs = torch.stack(outputs_list)
    
    if heatmaps_list:
        heatmaps = torch.from_numpy(np.concatenate(heatmaps_list)).float()
    else:
        heatmaps = None
    
    print(f"\nCRP features loaded:")
    print(f"  Shape: {features.shape}")
    print(f"  Feature dimension (n_filters): {features.shape[1]}")
    print(f"  These {features.shape[1]} filters ARE the concept space")
    
    # Load class weights if data path provided and use_class_weights is enabled
    class_weights = None
    if data_path and config['analysis'].get('use_class_weights', False):
        from models.cnn1d_model import VibrationDataset
        if os.path.exists(data_path):
            temp_dataset = VibrationDataset(data_path)
            class_weights = temp_dataset.weights
            print(f"\nClass weights loaded: {class_weights.numpy()}")
        else:
            print(f"\nWarning: Data path {data_path} not found, cannot load class weights")
    
    # Step 2: Fit GMM prototypes in CRP space
    print("\n" + "="*80)
    print("Step 2: Fit GMM Prototypes in CRP Filter Space")
    print("="*80)
    
    # Check if we should auto-select n_prototypes
    use_bic = config['tcd'].get('use_bic_selection', False)
    
    if use_bic:
        print("\nAuto-selecting optimal n_prototypes using BIC...")
        from tcd.prototypes import TemporalPrototypeDiscovery
        
        # Get BIC range from config
        bic_range = config['tcd'].get('bic_range', [1, 2, 3, 4, 5, 6])
        min_n = min(bic_range)
        max_n = max(bic_range)
        
        # Create temporary proto_discovery instance for BIC selection
        proto_discovery = TemporalPrototypeDiscovery(
            n_prototypes=1,  # temporary value
            covariance_type=config['tcd'].get('gmm_covariance', 'diag'),
            n_init=config['tcd'].get('gmm_n_init', 5),
            max_iter=config['tcd'].get('gmm_max_iter', 200),
            balance_method=config['tcd'].get('balance_method', 'downsample')
        )
        
        # Build features_dict for select_optimal_n_prototypes
        features_dict = {}
        for class_id in [0, 1]:
            class_mask = (labels == class_id) & (outputs.argmax(dim=1) == class_id)
            features_dict[class_id] = features[class_mask]
        
        # Select optimal n using BIC
        print(f"\nTesting range: {bic_range}")
        optimal_n_dict = {}
        
        for class_id in [0, 1]:
            class_features = features_dict[class_id]
            class_name = "OK" if class_id == 0 else "NOK"
            
            if class_features.shape[0] < min_n:
                print(f"\nClass {class_id} ({class_name}): Only {class_features.shape[0]} samples, using n={min_n}")
                optimal_n_dict[class_id] = min_n
                continue
            
            print(f"\nClass {class_id} ({class_name}) with {class_features.shape[0]} samples:")
            optimal_n, scores = TemporalPrototypeDiscovery.select_optimal_n_prototypes(
                class_features,
                min_prototypes=min_n,
                max_prototypes=min(max_n, class_features.shape[0] // 10),
                covariance_type=config['tcd'].get('gmm_covariance', 'diag'),
                n_init=config['tcd'].get('gmm_n_init', 5),
                max_iter=config['tcd'].get('gmm_max_iter', 200),
                criterion='bic'
            )
            optimal_n_dict[class_id] = optimal_n
            
            # Print BIC scores for transparency
            print(f"  BIC scores: {scores}")
            print(f"  Selected: n_prototypes={optimal_n}")
        
        # Use the average (rounded) for simplicity across both classes
        n_prototypes = int(np.mean(list(optimal_n_dict.values())))
        print(f"\n{'='*60}")
        print(f"BIC-optimal prototypes: {n_prototypes} (averaged across classes)")
        print(f"  Per-class optimal: OK={optimal_n_dict.get(0, 'N/A')}, NOK={optimal_n_dict.get(1, 'N/A')}")
        print(f"  If BIC says 1 per class, that's valid - means one strategy per class")
        print(f"{'='*60}")
    else:
        n_prototypes = config['tcd']['n_prototypes']
        print(f"\nUsing configured n_prototypes={n_prototypes}")
    
    # Initialize LearnedClusterTCD with updated defaults
    tcd = LearnedClusterTCD(
        n_prototypes=n_prototypes,
        layer_name=layer_name,
        covariance_type=config['tcd'].get('gmm_covariance', 'diag'),
        n_init=config['tcd'].get('gmm_n_init', 5),
        max_iter=config['tcd'].get('gmm_max_iter', 200),
        balance_method=config['tcd'].get('balance_method', 'downsample')
    )
    
    # Fit GMM prototypes
    print(f"\nFitting {n_prototypes} prototypes per class with improved convergence settings...")
    print(f"  Covariance type: {config['tcd'].get('gmm_covariance', 'diag')}")
    print(f"  n_init: {config['tcd'].get('gmm_n_init', 5)}")
    print(f"  max_iter: {config['tcd'].get('gmm_max_iter', 200)}")
    tcd.fit(features, labels, outputs, class_weights=class_weights)
    
    # Step 3: Global window analysis
    print("\n" + "="*80)
    print("Step 3: Window Analysis")
    print("="*80)
    
    global_windows = None
    window_analysis_results = None
    
    if heatmaps is not None:
        from tcd.variants.global_concepts import GlobalWindowAnalysis
        
        gw_config = config['tcd'].get('global_windows', {})
        iw_config = config['tcd'].get('important_windows', {})
        
        analyzer = GlobalWindowAnalysis(
            window_size=gw_config.get('window_size', 40),
            n_top_positions=gw_config.get('n_top_positions', 10),
            threshold_factor=gw_config.get('threshold_factor', 1.5),
            per_class=gw_config.get('per_class', True)
        )
        
        # Choose method based on config
        analysis_method = iw_config.get('method', 'global')
        
        if analysis_method == 'per_sample':
            print(f"\nUsing CNC-style per-sample window extraction...")
            print(f"  Window size: {iw_config.get('window_size', 40)} timesteps")
            print(f"  Top windows per sample: {iw_config.get('n_top_windows', 10)}")
            
            # Note: We don't have raw signals here, will extract from heatmaps
            window_analysis_results = analyzer.extract_important_windows_per_sample(
                heatmaps=heatmaps,
                signals=None,  # Could load from dataset if needed
                labels=labels,
                n_top_windows=iw_config.get('n_top_windows', 10),
                sample_rate=config['data']['sample_rate']
            )
            
            # Print per-class statistics
            print("\n" + "="*60)
            print("PER-CLASS WINDOW FEATURE STATISTICS")
            print("="*60)
            
            for class_id, stats in window_analysis_results['per_class_stats'].items():
                class_name = "OK" if class_id == 0 else "NOK"
                print(f"\nClass {class_id} ({class_name}):")
                
                for feat_name, feat_stats in list(stats.items())[:5]:  # Show first 5 features
                    print(f"  {feat_name}: mean={feat_stats['mean']:.4f}, std={feat_stats['std']:.4f}")
            
            # Print statistical test results
            if window_analysis_results['statistical_tests']:
                print("\n" + "="*60)
                print("STATISTICAL TESTS (OK vs NOK)")
                print("="*60)
                print(f"{'Feature':<25} {'p-value':<12} {'Cohen\'s d':<12} {'Significant':<12}")
                print("-"*60)
                
                for feat_name, test in window_analysis_results['statistical_tests'].items():
                    sig_str = "Yes" if test['significant'] else "No"
                    print(f"{feat_name:<25} {test['p_value']:<12.6f} {test['cohens_d']:<12.3f} {sig_str:<12}")
        else:
            # Use original global averaging method
            print(f"\nUsing global averaging method...")
            print(f"  Window size: {gw_config.get('window_size', 40)} timesteps")
            print(f"  Top positions: {gw_config.get('n_top_positions', 10)}")
            
            global_windows = analyzer.find_important_windows(heatmaps, labels)
            
            # Compute coverage
            coverage = analyzer.get_window_coverage_per_sample(heatmaps, labels)
            print(f"\nWindow coverage statistics:")
            for class_id, cov in coverage.items():
                class_name = "OK" if class_id == 0 else "NOK"
                print(f"  Class {class_id} ({class_name}): mean={cov.mean():.2%}, std={cov.std():.2%}")
    else:
        print("\nWarning: Heatmaps not found, skipping window analysis")
    
    # Step 4: Interpret prototypes
    print("\n" + "="*80)
    print("Step 4: Interpret CRP Prototypes")
    print("="*80)
    
    from tcd.interpretation import ConceptInterpreter
    
    interpreter = ConceptInterpreter(
        gmms=tcd.prototype_discovery.gmms,
        features=features,
        labels=labels,
        layer_name=layer_name
    )
    
    interpretations = interpreter.interpret_prototypes(
        global_windows=global_windows or {},
        heatmaps=heatmaps,
        signals=None,  # Could load raw signals if available
        top_k_filters=10
    )
    
    # Print interpretations
    interpreter.print_interpretations(interpretations, verbose=True)
    
    # Analyze prototypes per class (detailed statistics)
    print("\n" + "="*80)
    print("Step 5: Prototype Statistics")
    print("="*80)
    
    for class_id in [0, 1]:
        class_name = "OK" if class_id == 0 else "NOK"
        print(f"\n{class_name} (Class {class_id}) Prototypes:")
        
        # Get prototype samples
        top_k = config['tcd'].get('top_k_samples', 6)
        proto_samples = tcd.find_prototypes(class_id, top_k=top_k)
        print(f"  Top-{top_k} representative samples identified per prototype")
        
        # Get coverage
        coverage = tcd.get_coverage(class_id)
        print(f"  Coverage per prototype: {coverage}")
        
        # Get assignments
        class_mask = labels == class_id
        class_features = features[class_mask]
        assignments = tcd.assign_prototype(class_features, class_id)
        print(f"  Assignment distribution: {np.bincount(assignments)}")
    
    # Save results
    print("\n" + "="*80)
    print("Step 6: Save Results")
    print("="*80)
    
    os.makedirs(output_path, exist_ok=True)
    
    # Save TCD model
    with open(os.path.join(output_path, 'tcd_model.pkl'), 'wb') as f:
        pickle.dump(tcd, f)
    print(f"  Saved TCD model")
    
    # Save interpretations
    interpretation_export = interpreter.export_to_dict(interpretations)
    with open(os.path.join(output_path, 'interpretations.pkl'), 'wb') as f:
        pickle.dump(interpretation_export, f)
    print(f"  Saved interpretations")
    
    # Save global windows if available
    if global_windows:
        with open(os.path.join(output_path, 'global_windows.pkl'), 'wb') as f:
            pickle.dump(global_windows, f)
        print(f"  Saved global window analysis")
    
    # Save window analysis results if available
    if window_analysis_results:
        with open(os.path.join(output_path, 'window_analysis.pkl'), 'wb') as f:
            pickle.dump(window_analysis_results, f)
        print(f"  Saved per-sample window analysis")
    
    # Save main results
    results = {
        'variant': 'C',
        'layer_name': layer_name,
        'n_prototypes': n_prototypes,
        'features': features.numpy(),
        'labels': labels.numpy(),
        'outputs': outputs.numpy(),
        'global_windows': global_windows,
        'window_analysis': window_analysis_results,
        'interpretations': interpretation_export
    }
    
    with open(os.path.join(output_path, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f"  Saved results")
    
    print(f"\n✓ All results saved to {output_path}")
    print("="*80 + "\n")


def run_variant_b(
    features_path: str,
    output_path: str,
    config: dict
):
    """
    Run Variant B: Temporal descriptor concepts (SKELETON).
    
    Args:
        features_path: Path to CRP features directory
        output_path: Output directory
        config: Configuration dict
    """
    print("\n" + "="*60)
    print("VARIANT B: Temporal Descriptor Concepts (SKELETON)")
    print("="*60)
    
    print("\nVariant B is not fully implemented yet.")
    print("See tcd/variants/temporal_descriptors.py for TODO items.")
    print("\nKey steps to implement:")
    print("  1. Extract temporal descriptors (slope, peak, autocorr, spectral)")
    print("  2. Cluster descriptors with k-means or GMM")
    print("  3. Assign segments to concepts")
    
    # Create placeholder output
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'README.txt'), 'w') as f:
        f.write("Variant B is not yet implemented.\n")
        f.write("See tcd/variants/temporal_descriptors.py for skeleton.\n")


def run_variant_d(
    features_path: str,
    output_path: str,
    config: dict,
    data_path: str = None
):
    """
    Run Variant D: Comprehensive vibration feature concepts.
    
    Args:
        features_path: Path to CRP features directory
        output_path: Output directory
        config: Configuration dict
        data_path: Path to raw data directory (optional, for loading signals)
    """
    print("\n" + "="*60)
    print("VARIANT D: Comprehensive Vibration Feature Concepts")
    print("="*60)
    
    # Load heatmaps
    print("Loading heatmaps...")
    heatmaps_list = []
    labels_list = []
    
    for class_id in [0, 1]:
        heatmaps_path = os.path.join(features_path, f"heatmaps_class_{class_id}.hdf5")
        if not os.path.exists(heatmaps_path):
            print(f"Warning: Heatmaps not found at {heatmaps_path}")
            continue
        
        with h5py.File(heatmaps_path, 'r') as f:
            heatmaps = np.array(f['heatmaps'])
            heatmaps_list.append(heatmaps)
            labels_list.extend([class_id] * len(heatmaps))
            print(f"  Class {class_id}: {heatmaps.shape}")
    
    if not heatmaps_list:
        print("Error: No heatmaps found. Run run_analysis.py first.")
        return
    
    heatmaps = torch.from_numpy(np.concatenate(heatmaps_list)).float()
    labels = torch.tensor(labels_list).long()
    
    # Optionally load raw signals if data path provided
    signals = None
    if data_path and os.path.exists(data_path):
        print("\nNote: Raw signal loading not implemented yet.")
        print("Using heatmaps only for feature extraction.")
    
    # Get config
    vf_config = config['tcd'].get('vibration_features', {})
    sample_rate = config['data']['sample_rate']
    
    # Initialize VibrationFeatureTCD
    tcd = VibrationFeatureTCD(
        sample_rate=sample_rate,
        window_size=vf_config.get('window_size', 100),
        n_concepts=vf_config.get('n_concepts', None),
        use_feature_selection=vf_config.get('use_feature_selection', True),
        selection_method=vf_config.get('selection_method', 'mutual_info'),
        n_prototypes=vf_config.get('n_prototypes', 4),
        gmm_covariance=vf_config.get('gmm_covariance', 'full'),
        gmm_n_init=vf_config.get('gmm_n_init', 10),
        gmm_max_iter=vf_config.get('gmm_max_iter', 100)
    )
    
    # Fit on training data
    print("\nFitting vibration feature extractor...")
    tcd.fit(heatmaps, labels=labels, signals=signals)
    
    # Extract concepts
    print("\nExtracting concepts...")
    concept_relevances = tcd.extract_concepts(heatmaps, signals=signals)
    print(f"Concept relevances shape: {concept_relevances.shape}")
    
    # Get concept labels
    concept_labels = tcd.get_concept_labels()
    print(f"\nDiscovered {len(concept_labels)} concepts:")
    for i, label in enumerate(concept_labels[:20]):  # Show first 20
        print(f"  {i+1}. {label}")
    if len(concept_labels) > 20:
        print(f"  ... and {len(concept_labels) - 20} more")
    
    # Compute importance per concept (overall and per-class)
    importance = tcd.compute_concept_importance(heatmaps, signals=signals)
    
    # Separate by class
    class_0_mask = labels == 0
    class_1_mask = labels == 1
    heatmaps_class_0 = heatmaps[class_0_mask]
    heatmaps_class_1 = heatmaps[class_1_mask]
    
    importance_class_0 = tcd.compute_concept_importance(heatmaps_class_0, signals=None)
    importance_class_1 = tcd.compute_concept_importance(heatmaps_class_1, signals=None)
    
    print("\n" + "="*60)
    print("TOP 15 MOST IMPORTANT CONCEPTS (Overall and Per-Class)")
    print("="*60)
    
    # Get top 15 concepts by overall importance
    top_indices = np.argsort(importance)[-15:][::-1]
    
    # Dynamic column width
    max_label_len = max(len(concept_labels[i]) for i in top_indices)
    col_width = max(40, max_label_len + 2)
    
    print(f"{'Concept':<{col_width}} {'Overall':<15} {'OK (Class 0)':<15} {'NOK (Class 1)':<15} {'Ratio (NOK/OK)':<15}")
    print("-" * (col_width + 60))
    for idx in top_indices:
        label = concept_labels[idx]
        if importance_class_0[idx] > 0:
            ratio = importance_class_1[idx] / importance_class_0[idx]
            ratio_str = f"{ratio:>14.2f}x"
        else:
            ratio_str = "N/A".rjust(15)
        print(f"{label:<{col_width}} {importance[idx]:>14.4f} {importance_class_0[idx]:>14.4f} {importance_class_1[idx]:>14.4f} {ratio_str}")
    print("=" * (col_width + 60) + "\n")
    
    # Save results
    os.makedirs(output_path, exist_ok=True)
    
    results = {
        'variant': 'D',
        'method': 'vibration_features',
        'concept_labels': concept_labels,
        'concept_relevances': concept_relevances.numpy(),
        'labels': labels.numpy(),
        'importance': importance,
        'importance_class_0': importance_class_0,
        'importance_class_1': importance_class_1,
        'config': vf_config
    }
    
    with open(os.path.join(output_path, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Save TCD model
    with open(os.path.join(output_path, 'tcd_model.pkl'), 'wb') as f:
        pickle.dump(tcd, f)
    
    print(f"\n✓ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Discover temporal concepts using TCD variants")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--variant', type=str, required=True, choices=['A', 'B', 'C', 'D'],
                       help='TCD variant: A (filterbank/window), B (temporal descriptors), C (learned clusters), D (vibration features)')
    parser.add_argument('--features', type=str, required=True,
                       help='Path to CRP features directory from run_analysis.py')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for concept results')
    parser.add_argument('--layer', type=str, default='conv3',
                       help='Layer to use for Variant C (default: conv3)')
    parser.add_argument('--window-based', action='store_true',
                       help='Use window-based concept discovery for Variant A (default: filterbank)')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to data directory (for loading class weights in Variant C)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get data path from args or config
    data_path = args.data or config.get('data', {}).get('path', None)
    
    # Run appropriate variant
    if args.variant == 'A':
        run_variant_a(args.features, args.output, config, use_window_based=args.window_based)
    elif args.variant == 'B':
        run_variant_b(args.features, args.output, config)
    elif args.variant == 'C':
        run_variant_c(args.features, args.output, config, layer_name=args.layer, data_path=data_path)
    elif args.variant == 'D':
        run_variant_d(args.features, args.output, config, data_path=data_path)


if __name__ == "__main__":
    main()
