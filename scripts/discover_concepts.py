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
import matplotlib.pyplot as plt

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
    data_path: str = None,  # Path to dataset for loading class weights
    joint_gmm: bool = False  # If True, fit one GMM across both classes
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

    os.makedirs(output_path, exist_ok=True)

    # Get layer name from config if not specified
    if layer_name is None:
        layer_name = config['tcd'].get('primary_layer', 'conv4')
    
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

    # === CRITICAL: Normalize like PCX does ===
    # abs_norm: each sample's concept vector sums to 1.0 in absolute value
    abs_sums = features.abs().sum(dim=1, keepdim=True)
    abs_sums = abs_sums.clamp(min=1e-10)  # avoid division by zero
    features = features / abs_sums

    print(f"  Applied abs_norm normalization (sum of |features| = 1.0 per sample)")

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
    use_bic = config['tcd'].get('use_bic_selection', True)
    
    if use_bic:
        print("\nAuto-selecting optimal n_prototypes using BIC...")
        from tcd.prototypes import TemporalPrototypeDiscovery
        
        # Get BIC range from config
        bic_range = config['tcd'].get('bic_range', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        min_n = min(bic_range)
        max_n = max(bic_range)
        
        # Validate BIC range against sample sizes
        for class_id in [0, 1]:
            class_mask = (labels == class_id) & (outputs.argmax(dim=1) == class_id)
            n_class_samples = class_mask.sum().item()
            max_reasonable = n_class_samples // 10
            
            if max_n > max_reasonable:
                print(f"  Warning: BIC max ({max_n}) may be too large for class {class_id} "
                      f"with {n_class_samples} samples. Recommended max: {max_reasonable}")
        
        # Create temporary proto_discovery instance for BIC selection
        proto_discovery = TemporalPrototypeDiscovery(
            n_prototypes=1,  # temporary value
            covariance_type=config['tcd'].get('gmm_covariance', 'diag'),
            n_init=config['tcd'].get('gmm_n_init', 5),
            max_iter=config['tcd'].get('gmm_max_iter', 200),
            balance_method=config['tcd'].get('balance_method', 'none')
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
        
        # Use per-class n_prototypes directly instead of averaging
        n_prototypes = optimal_n_dict
        print(f"\n{'='*60}")
        print(f"BIC-optimal prototypes (per class): OK={optimal_n_dict.get(0, 'N/A')}, NOK={optimal_n_dict.get(1, 'N/A')}")
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
    joint_gmm_result = None
    if joint_gmm:
        print("\nFitting JOINT GMM (class-agnostic) across both classes...")
        use_bic = config['tcd'].get('use_bic_selection', False)
        joint_gmm_result = tcd.prototype_discovery.fit_joint(
            features, labels, use_bic_selection=use_bic
        )
        fitted_gmm, component_labels, purity_scores = joint_gmm_result
        print("\nJoint GMM component summary:")
        for i, (lbl, purity) in enumerate(zip(component_labels, purity_scores)):
            print(f"  Component {i}: {lbl}  (purity={purity*100:.1f}%)")
    else:
        proto_desc = (f"per-class ({', '.join(f'class {k}: {v}' for k, v in n_prototypes.items())})"
                      if isinstance(n_prototypes, dict) else f"{n_prototypes} per class")
        print(f"\nFitting {proto_desc} prototypes with improved convergence settings...")
        print(f"  Covariance type: {config['tcd'].get('gmm_covariance', 'diag')}")
        print(f"  n_init: {config['tcd'].get('gmm_n_init', 5)}")
        print(f"  max_iter: {config['tcd'].get('gmm_max_iter', 200)}")
        tcd.fit(features, labels, outputs, class_weights=class_weights)
    
    # Step 3: Global window analysis (optional — disabled by default for Variant C)
    print("\n" + "="*80)
    print("Step 3: Window Analysis")
    print("="*80)

    global_windows = None
    window_analysis_results = None

    iw_config = config['tcd'].get('important_windows', {})
    window_analysis_enabled = iw_config.get('enabled', False)

    if not window_analysis_enabled:
        print("\nWindow analysis disabled (tcd.important_windows.enabled: false).")
        print("Skipping Step 3.")
    elif heatmaps is not None:
        from tcd.variants.global_concepts import GlobalWindowAnalysis
        
        gw_config = config['tcd'].get('global_windows', {})
        
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
                print(f"{'Feature':<25} {'p-value':<12} {'Cohens_d':<12} {'Significant':<12}")
                print("-"*60)
                
                for feat_name, test in window_analysis_results['statistical_tests'].items():
                    sig_str = "Yes" if test['significant'] else "No"
                    p_val = test['p_value']
                    cohens_d = test['cohens_d']
                    p_str = f"{p_val:<12.6f}" if not np.isnan(p_val) else f"{'nan':<12}"
                    d_str = f"{cohens_d:<12.3f}" if not np.isnan(cohens_d) else f"{'nan':<12}"
                    print(f"{feat_name:<25} {p_str} {d_str} {sig_str:<12}")
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

    interpreter = None
    interpretations = None
    interpretation_export = {}

    if joint_gmm:
        print("  (Skipped — joint GMM mode does not use per-class interpreter)")
    else:
        from tcd.interpretation import ConceptInterpreter

        interpreter = ConceptInterpreter(
            gmms=tcd.prototype_discovery.gmms,
            features=features,
            labels=labels,
            layer_name=layer_name
        )

        # Add around line 475, before interpreter.interpret_prototypes()
        from models.cnn1d_model import VibrationDataset
        signals = None
        if data_path and os.path.exists(data_path):
            print("Loading raw signals for interpretation...")
            temp_dataset = VibrationDataset(data_path)
            # Stack all signals in same order as features (class 0 first, then class 1)
            signals_list = []
            for class_id in [0, 1]:
                class_indices = [i for i, (_, label) in enumerate(temp_dataset) if label == class_id]
                for idx in class_indices:
                    signals_list.append(temp_dataset[idx][0])
            signals = torch.stack(signals_list)
            print(f"  Loaded signals: {signals.shape}")

        interpretations = interpreter.interpret_prototypes(
            global_windows=global_windows or {},
            heatmaps=heatmaps,
            signals=signals,  # Could load raw signals if available,  but Now it actually has data
            top_k_filters=10
        )

        # Print interpretations
        interpreter.print_interpretations(interpretations, verbose=True)
    
    # Analyze prototypes per class (detailed statistics)
    print("\n" + "="*80)
    print("Step 5: Prototype Statistics")
    print("="*80)

    if joint_gmm:
        print("  (Skipped — joint GMM mode; see component summary above)")
    else:
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
    
    # Step 7: Generate Visualizations
    print("\n" + "="*80)
    print("Step 7: Generate Visualizations")
    print("="*80)
    
    # Import visualization functions
    from tcd.visualization import plot_prototype_gallery, plot_prototype_comparison
    
    # For visualizations, we need signals. For now we'll use heatmaps as proxy
    # In a full implementation, we'd load actual signals from dataset
    os.makedirs(output_path, exist_ok=True)

    if joint_gmm:
        print("  (Per-class prototype comparison skipped in joint GMM mode)")
    else:
        # Generate prototype comparison (OK vs NOK)
        print("\nGenerating prototype comparison plot...")
        try:
            ok_prototypes = []
            nok_prototypes = []
            
            for class_id in [0, 1]:
                if class_id in tcd.prototype_discovery.gmms:
                    gmm = tcd.prototype_discovery.gmms[class_id]
                    for proto_idx in range(gmm.n_components):
                        prototype_mean = gmm.means_[proto_idx]
                        if class_id == 0:
                            ok_prototypes.append(prototype_mean)
                        else:
                            nok_prototypes.append(prototype_mean)
            
            if ok_prototypes and nok_prototypes:
                fig = plot_prototype_comparison(
                    ok_prototypes=ok_prototypes,
                    nok_prototypes=nok_prototypes,
                    filter_names=[f"F{i}" for i in range(features.shape[1])],
                    top_k=10
                )

                comparison_dir = Path(output_path)
                comparison_dir.mkdir(parents=True, exist_ok=True)
                comparison_path = str(comparison_dir / 'prototype_comparison.png')

                # comparison_path = os.path.join(output_path, 'prototype_comparison.png')
                fig.savefig(comparison_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved prototype comparison to {comparison_path}")
            else:
                print("  Warning: Could not generate prototype comparison (missing prototypes)")
        except Exception as e:
            print(f"  Warning: Could not generate prototype comparison: {e}")
    
    # Generate prototype galleries for each class
    # Note: This requires loading actual signals/heatmaps for closest samples
    # For now we'll skip the full gallery but leave the structure
    print("\nPrototype gallery generation:")
    print("  Note: Full gallery generation requires loading sample signals")
    print("  This would be added in a complete implementation")

    # UMAP visualization of CRV space
    print("\nGenerating UMAP/PCA visualization of CRV space...")
    try:
        from tcd.visualization import plot_umap_prototypes

        features_np = features.numpy() if hasattr(features, 'numpy') else np.array(features)
        labels_np = labels.numpy() if hasattr(labels, 'numpy') else np.array(labels)

        # Build per-sample prototype assignments across all classes
        proto_assignments = np.full(len(features_np), -1, dtype=int)
        gmm_means_dict = {}
        for class_id in [0, 1]:
            if class_id not in tcd.prototype_discovery.gmms:
                continue
            gmm = tcd.prototype_discovery.gmms[class_id]
            class_mask = labels_np == class_id
            class_feat = features_np[class_mask]
            if len(class_feat) > 0:
                assignments = tcd.assign_prototype(
                    features[labels == class_id], class_id
                )
                proto_assignments[class_mask] = assignments + class_id * gmm.n_components
            gmm_means_dict[class_id] = gmm.means_

        fig_umap = plot_umap_prototypes(
            features=features_np,
            labels=labels_np,
            prototype_assignments=proto_assignments,
            gmm_means=gmm_means_dict
        )
        umap_path = os.path.join(output_path, 'umap_crv_space.png')
        fig_umap.savefig(umap_path, dpi=150, bbox_inches='tight')
        plt.close(fig_umap)
        print(f"  Saved UMAP/PCA plot to {umap_path}")
    except Exception as e:
        print(f"  Warning: Could not generate UMAP visualization: {e}")

    # Joint-GMM UMAP (only when --joint-gmm was used)
    if joint_gmm and joint_gmm_result is not None:
        print("\nGenerating joint-GMM UMAP visualization...")
        try:
            from tcd.visualization import plot_umap_prototypes

            fitted_gmm, component_labels, purity_scores = joint_gmm_result
            features_np = features.numpy() if hasattr(features, 'numpy') else np.array(features)
            labels_np = labels.numpy() if hasattr(labels, 'numpy') else np.array(labels)
            joint_assignments = fitted_gmm.predict(features_np)

            # Build gmm_means_dict keyed by component index (use dummy class 0)
            gmm_means_joint = {0: fitted_gmm.means_}

            fig_j = plot_umap_prototypes(
                features=features_np,
                labels=labels_np,
                prototype_assignments=joint_assignments,
                gmm_means=gmm_means_joint
            )
            j_path = os.path.join(output_path, 'umap_joint_gmm.png')
            fig_j.savefig(j_path, dpi=150, bbox_inches='tight')
            plt.close(fig_j)
            print(f"  Saved joint-GMM UMAP to {j_path}")
        except Exception as e:
            print(f"  Warning: Could not generate joint-GMM UMAP: {e}")


    # Concept-Prototype Matrix (PCX Figure 5 equivalent)
    if not joint_gmm:
        print("\nGenerating concept-prototype matrices...")
        try:
            from tcd.visualization import plot_concept_prototype_matrix

            for class_id in [0, 1]:
                if class_id not in tcd.prototype_discovery.gmms:
                    continue
                gmm = tcd.prototype_discovery.gmms[class_id]
                cov_pct = None
                try:
                    class_mask = labels == class_id
                    assignments = tcd.assign_prototype(features[class_mask], class_id)
                    counts = np.bincount(assignments, minlength=gmm.n_components)
                    cov_pct = 100.0 * counts / (counts.sum() + 1e-9)
                except Exception:
                    pass

                fig_mat = plot_concept_prototype_matrix(
                    gmm_means=gmm.means_,
                    class_id=class_id,
                    n_top_concepts=config['tcd'].get('top_k_samples', 5),
                    coverage_pct=cov_pct,
                    filter_names=[f"F{i}" for i in range(features.shape[1])]
                )
                mat_path = os.path.join(output_path, f'concept_prototype_matrix_class{class_id}.png')
                fig_mat.savefig(mat_path, dpi=150, bbox_inches='tight')
                plt.close(fig_mat)
                print(f"  Saved concept-prototype matrix to {mat_path}")
        except Exception as e:
            print(f"  Warning: Could not generate concept-prototype matrix: {e}")

    # Attribution graph (one per prototype per class)
    if not joint_gmm:
        print("\nGenerating attribution graphs...")
        try:
            from tcd.visualization import plot_attribution_graph

            top_k_attr = config['tcd'].get('top_k_samples', 5)
            for class_id in [0, 1]:
                if class_id not in tcd.prototype_discovery.gmms:
                    continue
                gmm = tcd.prototype_discovery.gmms[class_id]
                for proto_idx in range(gmm.n_components):
                    fig_attr = plot_attribution_graph(
                        prototype_mean=gmm.means_[proto_idx],
                        class_id=class_id,
                        top_k=top_k_attr
                    )
                    attr_path = os.path.join(
                        output_path,
                        f'attribution_graph_class{class_id}_proto{proto_idx}.png'
                    )
                    fig_attr.savefig(attr_path, dpi=150, bbox_inches='tight')
                    plt.close(fig_attr)
                    print(f"  Saved attribution graph to {attr_path}")
        except Exception as e:
            print(f"  Warning: Could not generate attribution graphs: {e}")
    
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
    if interpreter is not None and interpretations is not None:
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
        'interpretations': interpretation_export,
        'joint_gmm': joint_gmm,
    }

    if joint_gmm and joint_gmm_result is not None:
        fitted_gmm, component_labels, purity_scores = joint_gmm_result
        results['joint_gmm_component_labels'] = component_labels
        results['joint_gmm_purity_scores'] = purity_scores
    
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
    Run Variant B: Temporal descriptor concepts.
    
    Args:
        features_path: Path to CRP features directory
        output_path: Output directory
        config: Configuration dict
    """
    print("\n" + "="*60)
    print("VARIANT B: Temporal Descriptor Concepts")
    print("="*60)
    
    # Load heatmaps
    print("\nLoading heatmaps...")
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
    
    # Get config for Variant B
    n_concepts = config['tcd'].get('n_concepts', 5)
    descriptor_types = config['tcd'].get('descriptor_types', ['slope', 'peak', 'autocorr', 'spectral'])
    
    print(f"\nInitializing TemporalDescriptorTCD with {n_concepts} concepts")
    print(f"  Descriptor types: {descriptor_types}")
    
    # Create TemporalDescriptorTCD instance
    tcd = TemporalDescriptorTCD(
        n_concepts=n_concepts,
        descriptor_types=descriptor_types,
        clustering_method='kmeans'
    )
    
    # Fit: extract segments, compute descriptors, cluster them
    print("\nFitting temporal descriptor clustering...")
    tcd.fit(heatmaps)
    
    # Extract concepts: assign segments to clusters
    print("\nExtracting concept assignments...")
    concept_assignments = tcd.extract_concepts(heatmaps)
    print(f"Concept assignments shape: {concept_assignments.shape}")
    
    # Print discovered temporal concepts
    print("\n" + "="*60)
    print("DISCOVERED TEMPORAL CONCEPTS")
    print("="*60)
    
    concept_labels = tcd.get_concept_labels()
    for i, label in enumerate(concept_labels):
        print(f"  Concept {i}: {label}")
    
    # Compute per-class concept importance
    print("\n" + "="*60)
    print("PER-CLASS CONCEPT IMPORTANCE")
    print("="*60)

    for class_id in [0, 1]:
        class_mask = labels == class_id
        class_assignments = concept_assignments[class_mask]

        # Convert to numpy if it's a torch tensor
        if isinstance(class_assignments, torch.Tensor):
            class_assignments = class_assignments.numpy()

        # Validate that assignments are integers
        if not np.issubdtype(class_assignments.dtype, np.integer):
            print(f"  Warning: Converting non-integer assignments to int for class {class_id}")
            class_assignments = class_assignments.astype(np.int64)

        # Count concept occurrences
        concept_counts = np.bincount(class_assignments.flatten().astype(int),
                                     minlength=n_concepts)

        total_segments = concept_counts.sum()
        
        class_name = "OK" if class_id == 0 else "NOK"
        print(f"\nClass {class_id} ({class_name}):")
        for i in range(n_concepts):
            pct = 100.0 * concept_counts[i] / (total_segments + 1e-10)
            print(f"  Concept {i}: {concept_counts[i]} segments ({pct:.1f}%)")
    
    # Print cluster statistics if available
    if hasattr(tcd, 'cluster_centers_'):
        print("\n" + "="*60)
        print("CLUSTER CENTER STATISTICS")
        print("="*60)
        
        centers = tcd.cluster_centers_
        print(f"Cluster centers shape: {centers.shape}")
        
        for i in range(n_concepts):
            center = centers[i]
            print(f"\nConcept {i}:")
            print(f"  Mean: {center.mean():.4f}")
            print(f"  Std: {center.std():.4f}")
            print(f"  Min: {center.min():.4f}")
            print(f"  Max: {center.max():.4f}")
    
    # Save results
    os.makedirs(output_path, exist_ok=True)
    
    results = {
        'variant': 'B',
        'n_concepts': n_concepts,
        'descriptor_types': descriptor_types,
        'concept_labels': concept_labels,
        'concept_assignments': concept_assignments.numpy(),
        'labels': labels.numpy()
    }
    
    if hasattr(tcd, 'cluster_centers_'):
        results['cluster_centers'] = tcd.cluster_centers_
    
    with open(os.path.join(output_path, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Save TCD model
    with open(os.path.join(output_path, 'tcd_model.pkl'), 'wb') as f:
        pickle.dump(tcd, f)
    
    print(f"\n✓ Results saved to {output_path}")
    print("="*60 + "\n")


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
    parser.add_argument('--layer', type=str, default='conv4',
                       help='Layer to use for Variant C (default: conv4)')
    parser.add_argument('--window-based', action='store_true',
                       help='Use window-based concept discovery for Variant A (default: filterbank)')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to data directory (for loading class weights in Variant C)')
    parser.add_argument('--joint-gmm', action='store_true',
                       help='Fit one GMM across both classes instead of per-class (Variant C only)')
    
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
        run_variant_c(
            args.features, args.output, config,
            layer_name=args.layer, data_path=data_path,
            joint_gmm=args.joint_gmm,
        )
    elif args.variant == 'D':
        run_variant_d(args.features, args.output, config, data_path=data_path)


if __name__ == "__main__":
    main()
