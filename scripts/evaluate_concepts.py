#!/usr/bin/env python3
"""
Step 3: Evaluate Concepts - Intervention and validation.

Tests discovered concepts via:
- Faithfulness: causal intervention (suppress concepts, measure effect)
- Stability: consistency across similar samples
- Concept purity: distinctiveness of concepts
- Coverage: prototype assignment statistics

Usage:
    python scripts/evaluate_concepts.py --concepts results/concepts_A --model path/to/model.ckpt --data path/to/data
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os
import yaml
import torch
import pickle
import numpy as np
import traceback
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.cnn1d_model import CNN1D_Wide, VibrationDataset
from tcd.intervention import measure_concept_importance, compute_intervention_effect
from tcd.evaluation import evaluate_concept_quality, print_evaluation_report
from tcd.variants.filterbank import FilterBankTCD
from tcd.variants.learned_clusters import LearnedClusterTCD


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_variant_a(
    concepts_path: str,
    model,
    dataset,
    config: dict,
    output_path: str
):
    """
    Evaluate Variant A: Filterbank concepts.
    
    Tests importance of frequency-band concepts via intervention.
    
    Args:
        concepts_path: Path to concept results
        model: PyTorch model
        dataset: Dataset
        config: Configuration dict
        output_path: Output directory
    """
    print("\n" + "="*60)
    print("EVALUATING VARIANT A: Filterbank Concepts")
    print("="*60)
    
    # Load results
    with open(os.path.join(concepts_path, 'results.pkl'), 'rb') as f:
        results = pickle.load(f)
    
    concept_relevances = torch.from_numpy(results['concept_relevances'])
    labels = results['labels']
    importance = results['importance']
    
    print(f"\nConcept importance scores:")
    for i, (label, imp) in enumerate(zip(results['concept_labels'], importance)):
        print(f"  {label}: {imp:.4f}")
    
    # For filterbank, concepts are derived from heatmaps, not layer activations
    # So intervention needs to work at input level
    print("\nNote: Intervention for Variant A requires modifying input-level relevance.")
    print("This is conceptually different from suppressing layer activations.")
    print("Skipping intervention test for Variant A.")
    
    # Compute stability and purity
    print("\nComputing stability and purity...")
    from tcd.evaluation import compute_stability, compute_concept_purity
    
    # Overall metrics
    stability = compute_stability(concept_relevances.numpy(), labels)
    purity = compute_concept_purity(concept_relevances.numpy())
    
    print(f"Overall Stability: {stability:.3f}")
    print(f"Overall Purity: {purity:.3f}")
    
    # Per-class metrics
    class_0_mask = labels == 0
    class_1_mask = labels == 1
    concept_relevances_class_0 = concept_relevances[class_0_mask]
    concept_relevances_class_1 = concept_relevances[class_1_mask]
    
    # Compute per-class stability and purity for class 0 (OK)
    if len(concept_relevances_class_0) > 0:
        stability_class_0 = compute_stability(concept_relevances_class_0.numpy(), np.zeros(len(concept_relevances_class_0)))
        purity_class_0 = compute_concept_purity(concept_relevances_class_0.numpy())
    else:
        stability_class_0 = 0.0
        purity_class_0 = 0.0
    
    # Compute per-class stability and purity for class 1 (NOK)
    if len(concept_relevances_class_1) > 0:
        stability_class_1 = compute_stability(concept_relevances_class_1.numpy(), np.zeros(len(concept_relevances_class_1)))
        purity_class_1 = compute_concept_purity(concept_relevances_class_1.numpy())
    else:
        stability_class_1 = 0.0
        purity_class_1 = 0.0
    
    # Print per-class report
    print("\n" + "="*60)
    print("PER-CLASS EVALUATION REPORT")
    print("="*60)
    print(f"\n{'Metric':<20} {'OK (Class 0)':<20} {'NOK (Class 1)':<20}")
    print("-"*60)
    print(f"{'Stability:':<20} {stability_class_0:>19.3f} {stability_class_1:>19.3f}")
    print(f"{'Purity:':<20} {purity_class_0:>19.3f} {purity_class_1:>19.3f}")
    print("="*60 + "\n")
    
    # Show per-class concept importance breakdown
    print("\nPer-Class Concept Importance Breakdown:")
    print(f"{'Concept':<20} {'Overall':<15} {'OK (Class 0)':<15} {'NOK (Class 1)':<15}")
    print("-"*65)
    
    # Compute per-class importance (handle empty tensors)
    if len(concept_relevances_class_0) > 0:
        importance_class_0 = concept_relevances_class_0.mean(dim=0).numpy()
    else:
        importance_class_0 = np.zeros_like(importance)
    
    if len(concept_relevances_class_1) > 0:
        importance_class_1 = concept_relevances_class_1.mean(dim=0).numpy()
    else:
        importance_class_1 = np.zeros_like(importance)
    
    for i, label in enumerate(results['concept_labels']):
        print(f"{label:<20} {importance[i]:>14.4f} {importance_class_0[i]:>14.4f} {importance_class_1[i]:>14.4f}")
    print("-"*65 + "\n")
    
    # Compute class-weighted metrics if class weights are available
    if hasattr(dataset, 'weights'):
        class_weights = dataset.weights.cpu().numpy()
        print(f"\nComputing class-weighted averages with weights: {class_weights}")
        
        # Weighted stability
        stability_weighted = (stability_class_0 * class_weights[0] + 
                             stability_class_1 * class_weights[1]) / (class_weights[0] + class_weights[1])
        
        # Weighted purity
        purity_weighted = (purity_class_0 * class_weights[0] + 
                          purity_class_1 * class_weights[1]) / (class_weights[0] + class_weights[1])
        
        print(f"\nClass-Weighted Metrics:")
        print(f"  Stability (weighted): {stability_weighted:.3f}")
        print(f"  Purity (weighted):    {purity_weighted:.3f}")
    else:
        stability_weighted = None
        purity_weighted = None
    
    # Save evaluation
    os.makedirs(output_path, exist_ok=True)
    evaluation = {
        'variant': 'A',
        'stability': stability,
        'purity': purity,
        'importance': importance,
        'stability_class_0': stability_class_0,
        'stability_class_1': stability_class_1,
        'purity_class_0': purity_class_0,
        'purity_class_1': purity_class_1,
        'importance_class_0': importance_class_0,
        'importance_class_1': importance_class_1,
        'stability_weighted': stability_weighted,
        'purity_weighted': purity_weighted
    }
    
    with open(os.path.join(output_path, 'evaluation.pkl'), 'wb') as f:
        pickle.dump(evaluation, f)
    
    print(f"\n✓ Evaluation saved to {output_path}")


def evaluate_variant_c(
    concepts_path: str,
    model,
    dataset,
    config: dict,
    output_path: str,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Evaluate Variant C: Learned cluster concepts.
    
    Tests prototype concepts via layer-level intervention.
    
    Args:
        concepts_path: Path to concept results
        model: PyTorch model
        dataset: Dataset
        config: Configuration dict
        output_path: Output directory
        device: Device to use
    """
    print("\n" + "="*60)
    print("EVALUATING VARIANT C: Learned Cluster Concepts")
    print("="*60)
    
    # Load results
    with open(os.path.join(concepts_path, 'results.pkl'), 'rb') as f:
        results = pickle.load(f)
    
    with open(os.path.join(concepts_path, 'tcd_model.pkl'), 'rb') as f:
        tcd = pickle.load(f)
    
    features = torch.from_numpy(results['features'])
    labels = torch.from_numpy(results['labels'])
    layer_name = results['layer_name']
    
    print(f"Layer: {layer_name}")
    print(f"Features shape: {features.shape}")
    
    # Get number of concepts (filters at layer)
    n_concepts = features.shape[1]
    
    # Measure concept importance via intervention for both classes
    print(f"\nMeasuring importance of {n_concepts} concepts via intervention...")
    
    model.to(device)
    model.eval()
    
    # Create subsets for both classes
    class_0_indices = [i for i, (_, label) in enumerate(dataset) if label == 0]
    class_1_indices = [i for i, (_, label) in enumerate(dataset) if label == 1]
    
    importance_scores_class_0 = np.zeros(n_concepts)
    importance_scores_class_1 = np.zeros(n_concepts)
    
    # Test on class 0 (OK)
    print(f"Testing on class 0 (OK) samples ({len(class_0_indices)} available)...")
    class_0_subset = torch.utils.data.Subset(dataset, class_0_indices[:min(100, len(class_0_indices))])
    
    if len(class_0_subset) > 0:
        importance_scores_class_0 = measure_concept_importance(
            model=model,
            dataset=class_0_subset,
            layer_name=layer_name,
            n_concepts=n_concepts,
            target_class=0,
            method='suppress',
            batch_size=config['analysis']['batch_size']
        )
        
        print(f"\nTop 5 most important concepts for OK (class 0):")
        top_indices = np.argsort(importance_scores_class_0)[-5:][::-1]
        for idx in top_indices:
            print(f"  Concept {idx}: {importance_scores_class_0[idx]:.4f}")
    else:
        print("Warning: No class 0 samples found for intervention test")
    
    # Test on class 1 (NOK)
    print(f"\nTesting on class 1 (NOK) samples ({len(class_1_indices)} available)...")
    class_1_subset = torch.utils.data.Subset(dataset, class_1_indices[:min(100, len(class_1_indices))])
    
    if len(class_1_subset) > 0:
        importance_scores_class_1 = measure_concept_importance(
            model=model,
            dataset=class_1_subset,
            layer_name=layer_name,
            n_concepts=n_concepts,
            target_class=1,
            method='suppress',
            batch_size=config['analysis']['batch_size']
        )
        
        print(f"\nTop 5 most important concepts for NOK (class 1):")
        top_indices = np.argsort(importance_scores_class_1)[-5:][::-1]
        for idx in top_indices:
            print(f"  Concept {idx}: {importance_scores_class_1[idx]:.4f}")
    else:
        print("Warning: No class 1 samples found for intervention test")
    
    # Overall importance (averaged across both classes)
    importance_scores = (importance_scores_class_0 + importance_scores_class_1) / 2
    
    # Add prototype-level intervention if enabled
    proto_results = None
    if config.get('evaluation', {}).get('prototype_intervention', True):
        print("\n" + "="*60)
        print("PROTOTYPE-LEVEL MULTI-FILTER INTERVENTION")
        print("="*60)
        
        from tcd.intervention import prototype_intervention_analysis
        
        # Prepare data for intervention
        # We need full dataset to run interventions
        loader = DataLoader(dataset, batch_size=config['analysis']['batch_size'], shuffle=False)
        
        all_data = []
        all_labels = []
        for batch_data, batch_labels in loader:
            all_data.append(batch_data)
            all_labels.append(batch_labels)
        
        data_tensor = torch.cat(all_data).to(device)
        labels_tensor = torch.cat(all_labels).to(device)
        
        top_k = config.get('evaluation', {}).get('prototype_top_k', 5)
        
        print(f"\nRunning prototype intervention with top-{top_k} filters per prototype...")
        
        proto_results = prototype_intervention_analysis(
            model=model,
            data=data_tensor,
            labels=labels_tensor,
            prototype_discovery=tcd.prototype_discovery,
            features=features.to(device),
            layer_name=layer_name,
            top_k=top_k,
            method='suppress'
        )
        
        # Print formatted results
        print("\n" + "="*60)
        print("PROTOTYPE INTERVENTION RESULTS")
        print("="*60)
        
        for class_id, class_results in proto_results.items():
            class_name = "OK" if class_id == 0 else "NOK"
            print(f"\nClass {class_id} ({class_name}):")
            
            for result in class_results:
                proto_idx = result['prototype_idx']
                top_filters = result['top_filters']
                mean_prob_change = result['mean_prob_change']
                flip_rate = result['flip_rate']
                
                print(f"  Prototype {proto_idx}:")
                print(f"    Top-{top_k} filters: {top_filters}")
                print(f"    Mean probability change: {mean_prob_change:.4f}")
                print(f"    Prediction flip rate: {flip_rate:.2%}")
        
        # --- Prototype diversity diagnostics ---
        print("\n" + "="*60)
        print("PROTOTYPE DIVERSITY DIAGNOSTICS")
        print("="*60)
        
        from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
        
        for class_id, class_results in proto_results.items():
            class_name = "OK" if class_id == 0 else "NOK"
            if class_id not in tcd.prototype_discovery.gmms:
                continue
            gmm = tcd.prototype_discovery.gmms[class_id]
            means = gmm.means_  # (n_prototypes, n_filters)
            
            # Inter-prototype cosine similarity
            if len(means) > 1:
                sim_matrix = sk_cosine_similarity(means)
                # Off-diagonal mean
                n_p = len(means)
                off_diag = [sim_matrix[i, j] for i in range(n_p) for j in range(n_p) if i != j]
                mean_inter_sim = float(np.mean(off_diag))
                print(f"\nClass {class_id} ({class_name}):")
                print(f"  Mean inter-prototype cosine similarity: {mean_inter_sim:.4f}")
                if mean_inter_sim > 0.95:
                    print(f"  ⚠ WARNING: Prototypes are nearly identical (sim={mean_inter_sim:.3f}). "
                          "Consider increasing n_prototypes or checking GMM convergence.")
            
            # Check top-k filter overlap
            all_top_filters = []
            for result in class_results:
                all_top_filters.append(set(result['top_filters']))
            if len(all_top_filters) > 1:
                common = all_top_filters[0].copy()
                for s in all_top_filters[1:]:
                    common &= s
                if len(common) == top_k:
                    print(f"  ⚠ WARNING: All {len(all_top_filters)} prototypes share the same "
                          f"top-{top_k} filters {sorted(common)}. Prototype diversity is low.")
    
    # Compute evaluation metrics
    print("\n" + "="*60)
    
    from tcd.evaluation import (compute_faithfulness, compute_stability, compute_concept_purity,
                             compute_prototype_coverage, compute_faithfulness_prototype_level,
                             compute_incremental_faithfulness)
    
    # Get relevance scores (mean of features)
    relevance_scores = features.abs().mean(dim=0).numpy()
    
    # Separate features by class
    class_0_mask = labels == 0
    class_1_mask = labels == 1
    class_0_features = features[class_0_mask]
    class_1_features = features[class_1_mask]
    
    # Overall metrics
    intervention_effects = importance_scores
    faithfulness = compute_faithfulness(relevance_scores, intervention_effects)
    stability = compute_stability(features.numpy(), labels.numpy())
    purity = compute_concept_purity(features.numpy())
    
    # Per-class metrics for class 0 (OK)
    relevance_scores_class_0 = class_0_features.abs().mean(dim=0).numpy()
    faithfulness_class_0 = compute_faithfulness(relevance_scores_class_0, importance_scores_class_0)
    stability_class_0 = compute_stability(class_0_features.numpy(), np.zeros(len(class_0_features)))
    purity_class_0 = compute_concept_purity(class_0_features.numpy())
    
    # Prototype coverage for class 0
    if len(class_0_features) > 0:
        assignments_class_0 = tcd.assign_prototype(class_0_features, class_id=0)
        n_proto_0 = tcd.prototype_discovery.gmms[0].n_components
        coverage_metrics_class_0 = compute_prototype_coverage(assignments_class_0, n_proto_0)
    else:
        coverage_metrics_class_0 = {'coverage': 0, 'balance': 0, 'max_coverage': 0}
    
    # Per-class metrics for class 1 (NOK)
    if len(class_1_features) > 0:
        relevance_scores_class_1 = class_1_features.abs().mean(dim=0).numpy()
        faithfulness_class_1 = compute_faithfulness(relevance_scores_class_1, importance_scores_class_1)
        stability_class_1 = compute_stability(class_1_features.numpy(), np.zeros(len(class_1_features)))
        purity_class_1 = compute_concept_purity(class_1_features.numpy())
        
        # Prototype coverage for class 1
        assignments_class_1 = tcd.assign_prototype(class_1_features, class_id=1)
        n_proto_1 = tcd.prototype_discovery.gmms[1].n_components
        coverage_metrics_class_1 = compute_prototype_coverage(assignments_class_1, n_proto_1)
    else:
        faithfulness_class_1 = 0.0
        stability_class_1 = 0.0
        purity_class_1 = 0.0
        coverage_metrics_class_1 = {'coverage': 0, 'balance': 0, 'max_coverage': 0}
    
    # Print overall report
    metrics = {
        'faithfulness': faithfulness,
        'stability': stability,
        'purity': purity
    }
    
    print_evaluation_report(metrics)
    
    # Print per-class report
    metrics_class_0 = {
        'faithfulness': faithfulness_class_0,
        'stability': stability_class_0,
        'purity': purity_class_0,
        **coverage_metrics_class_0
    }
    
    metrics_class_1 = {
        'faithfulness': faithfulness_class_1,
        'stability': stability_class_1,
        'purity': purity_class_1,
        **coverage_metrics_class_1
    }
    
    # Use the print_evaluation_report function with per-class metrics
    print_evaluation_report(metrics, per_class_metrics={'class_0': metrics_class_0, 'class_1': metrics_class_1})

    # Faithfulness warning for minority class
    if faithfulness_class_1 < 0.1:
        print(f"\n⚠ WARNING: NOK (class 1) faithfulness is {faithfulness_class_1:.3f} (< 0.1).")
        print("  The concept relevance scores may not correlate with causal effects for the")
        print("  minority class. Consider: (1) more NOK samples, (2) checking intervention")
        print("  method, or (3) verifying the GMM has converged properly for class 1.")
    
    # Prototype-level faithfulness
    faithfulness_proto = None
    if proto_results is not None:
        print("\n" + "="*60)
        print("PROTOTYPE-LEVEL FAITHFULNESS")
        print("="*60)
        top_k = config.get('evaluation', {}).get('prototype_top_k', 5)
        faithfulness_proto = compute_faithfulness_prototype_level(
            proto_results=proto_results,
            prototype_discovery=tcd.prototype_discovery,
            importance_scores_per_class={0: importance_scores_class_0, 1: importance_scores_class_1},
            top_k=top_k
        )
        print(f"\nPrototype-level Spearman faithfulness (top-{top_k} filters per prototype):")
        for class_id, class_mean in faithfulness_proto['per_class'].items():
            class_name = "OK" if class_id == 0 else "NOK"
            print(f"  Class {class_id} ({class_name}): {class_mean:.3f}")
            for proto_idx, corr in faithfulness_proto['per_prototype'].get(class_id, {}).items():
                print(f"    Prototype {proto_idx}: {corr:.3f}")
        print(f"  Overall mean: {faithfulness_proto['mean']:.3f}")
    
    # PCX-style incremental faithfulness with AUC
    incremental_faithfulness_results = {}
    inc_faith_config = config.get('evaluation', {}).get('incremental_faithfulness', {})
    if inc_faith_config.get('enabled', True):
        print("\n" + "="*60)
        print("INCREMENTAL FAITHFULNESS (PCX-style AUC)")
        print("="*60)
        inc_n_samples = inc_faith_config.get('n_samples', 100)
        inc_n_steps = inc_faith_config.get('n_steps', 30)
        
        for target_cls in [0, 1]:
            class_name = "OK" if target_cls == 0 else "NOK"
            # Use per-class relevance vectors (convert to numpy if needed)
            labels_np = labels.numpy() if hasattr(labels, 'numpy') else np.asarray(labels)
            features_np = features.numpy() if hasattr(features, 'numpy') else np.asarray(features)
            cls_mask = labels_np == target_cls
            cls_relevance = features_np[cls_mask]
            
            if cls_relevance.shape[0] == 0:
                print(f"  Class {target_cls} ({class_name}): no samples, skipping")
                continue
            
            try:
                result = compute_incremental_faithfulness(
                    model=model,
                    dataset=dataset,
                    layer_name=layer_name,
                    concept_relevance_vectors=features_np,  #all samples
                    labels=labels_np,
                    n_samples=inc_n_samples,
                    n_steps=inc_n_steps,
                    target_class=target_cls,
                    batch_size=config['analysis']['batch_size'],
                    device=device
                )
                incremental_faithfulness_results[target_cls] = result
                print(f"\n  Class {target_cls} ({class_name}):")
                print(f"    AUC: {result['auc']:.4f}")
                print(f"    Samples used: {result['n_samples_used']}")
                print(f"    Steps: {result['steps'][:5]}... (showing first 5)")
                
                # Save incremental faithfulness plot
                try:
                    import matplotlib.pyplot as plt
                    os.makedirs(output_path, exist_ok=True)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(result['steps'], result['mean_logit_changes'],
                            marker='o', linewidth=2, label=f'Relevance-ordered (AUC={result["auc"]:.4f})')
                    ax.plot(result['steps'], result['perturbation_curve_random'],
                            marker='s', linewidth=2, linestyle='--',
                            label=f'Random baseline (AUC={result["auc_random"]:.4f})')
                    ax.set_xlabel('Number of suppressed concepts')
                    ax.set_ylabel('Mean logit change')
                    ax.set_title(
                        f'Incremental Faithfulness — {class_name} (Class {target_cls})\n'
                        f'AUC relevance={result["auc"]:.4f}, AUC random={result["auc_random"]:.4f}'
                    )
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plot_path = os.path.join(output_path, f'incremental_faithfulness_class_{target_cls}.png')
                    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f"    Plot saved to {plot_path}")
                except Exception as plot_err:
                    print(f"    Warning: could not save plot: {plot_err}")
            except Exception as e:
                print(f"  Class {target_cls} ({class_name}): error — {e}")
                traceback.print_exc()
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    os.makedirs(output_path, exist_ok=True)

    # Concept conditional heatmaps if enabled
    if config.get('evaluation', {}).get('concept_heatmaps', True):
        from tcd.visualization import generate_concept_heatmaps
        from tcd.composites import CNCValidatedComposite
        
        print("\nGenerating concept conditional heatmaps...")
        
        # Create composite for CRP
        composite = CNCValidatedComposite()
        
        concept_heatmap_top_k = config.get('evaluation', {}).get('concept_heatmap_top_k', 5)
        
        try:
            concept_heatmap_figs = generate_concept_heatmaps(
                model=model,
                dataset=dataset,
                prototype_discovery=tcd.prototype_discovery,
                layer_name=layer_name,
                composite=composite,
                output_dir=os.path.join(output_path, 'concept_heatmaps'),
                top_k=concept_heatmap_top_k,
                device=device,
                features=features,
                labels=labels
            )
            print(f"  Generated {len(concept_heatmap_figs)} concept heatmap figures")
        except Exception as e:
            print(f"  Warning: Could not generate concept heatmaps: {e}")

    # UMAP visualization of CRV space
    print("\nGenerating UMAP/PCA visualization of CRV space...")
    try:
        from tcd.visualization import plot_umap_prototypes

        features_np = features.numpy() if hasattr(features, 'numpy') else np.array(features)
        labels_np = labels.numpy() if hasattr(labels, 'numpy') else np.array(labels)

        proto_assignments = np.full(len(features_np), -1, dtype=int)
        gmm_means_dict = {}
        for cid in [0, 1]:
            if cid not in tcd.prototype_discovery.gmms:
                continue
            gmm = tcd.prototype_discovery.gmms[cid]
            class_mask = labels_np == cid
            class_feat = features[labels == cid]
            if len(class_feat) > 0:
                assignments = tcd.assign_prototype(class_feat, cid)
                proto_assignments[class_mask] = assignments + cid * gmm.n_components
            gmm_means_dict[cid] = gmm.means_

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

    # Concept-Prototype Matrix (PCX Figure 5 equivalent)
    print("\nGenerating concept-prototype matrices...")
    try:
        from tcd.visualization import plot_concept_prototype_matrix

        for cid in [0, 1]:
            if cid not in tcd.prototype_discovery.gmms:
                continue
            gmm = tcd.prototype_discovery.gmms[cid]
            cov_pct = None
            try:
                class_mask = labels == cid
                assignments = tcd.assign_prototype(features[class_mask], cid)
                counts = np.bincount(assignments, minlength=gmm.n_components)
                cov_pct = 100.0 * counts / (counts.sum() + 1e-9)
            except Exception:
                pass

            top_k_eval = config.get('evaluation', {}).get('prototype_top_k', 5)
            fig_mat = plot_concept_prototype_matrix(
                gmm_means=gmm.means_,
                class_id=cid,
                n_top_concepts=top_k_eval,
                coverage_pct=cov_pct,
                filter_names=[f"F{i}" for i in range(features.shape[1])]
            )
            mat_path = os.path.join(output_path, f'concept_prototype_matrix_class{cid}.png')
            fig_mat.savefig(mat_path, dpi=150, bbox_inches='tight')
            plt.close(fig_mat)
            print(f"  Saved concept-prototype matrix to {mat_path}")
    except Exception as e:
        print(f"  Warning: Could not generate concept-prototype matrix: {e}")

    # Attribution graph (one per prototype per class)
    print("\nGenerating attribution graphs...")
    try:
        from tcd.visualization import plot_attribution_graph

        top_k_attr = config.get('evaluation', {}).get('prototype_top_k', 5)
        for cid in [0, 1]:
            if cid not in tcd.prototype_discovery.gmms:
                continue
            gmm = tcd.prototype_discovery.gmms[cid]
            for proto_idx in range(gmm.n_components):
                fig_attr = plot_attribution_graph(
                    prototype_mean=gmm.means_[proto_idx],
                    class_id=cid,
                    top_k=top_k_attr
                )
                attr_path = os.path.join(
                    output_path,
                    f'attribution_graph_class_{cid}_proto_{proto_idx}.png'
                )
                fig_attr.savefig(attr_path, dpi=150, bbox_inches='tight')
                plt.close(fig_attr)
                print(f"  Saved attribution graph to {attr_path}")
    except Exception as e:
        print(f"  Warning: Could not generate attribution graphs: {e}")
    
    # Save evaluation
    os.makedirs(output_path, exist_ok=True)
    evaluation = {
        'variant': 'C',
        'layer_name': layer_name,
        'metrics': metrics,
        'importance_scores': importance_scores,
        'importance_scores_class_0': importance_scores_class_0,
        'importance_scores_class_1': importance_scores_class_1,
        'metrics_class_0': metrics_class_0,
        'metrics_class_1': metrics_class_1,
        'prototype_intervention': proto_results,
        'faithfulness_prototype_level': faithfulness_proto,
        'incremental_faithfulness': incremental_faithfulness_results
    }




    # Robustness deviation analysis if enabled
    if config.get('evaluation', {}).get('robustness_analysis', True):
        from tcd.robustness import robustness_deviation_analysis
        from tcd.visualization import plot_deviation_matrix
        
        print("\nPerforming robustness deviation analysis...")
        
        deviation_threshold = config.get('evaluation', {}).get('deviation_threshold', 2.0)
        
        try:
            robustness_results = robustness_deviation_analysis(
                features=features,
                labels=labels,
                prototype_discovery=tcd.prototype_discovery,
                deviation_threshold=deviation_threshold
            )
            
            # Print summary
            print("\nRobustness Summary:")
            for class_id, stats in robustness_results['class_statistics'].items():
                class_name = "OK" if class_id == 0 else "NOK"
                print(f"  {class_name} (Class {class_id}): {stats['pct_flagged']:.1f}% samples flagged as unusual")
            
            # Generate deviation matrix visualization
            print("\nGenerating deviation matrix visualizations...")
            
            for class_id in [0, 1]:
                if class_id not in robustness_results['class_statistics']:
                    continue
                
                class_mask = labels == class_id
                class_features = features[class_mask]
                class_deviations_all = robustness_results['per_sample_deviations'][class_mask.cpu().numpy()]
                class_assignments = robustness_results['per_sample_assignments'][class_mask.cpu().numpy()]
                
                # For visualization, show deviations for top 50 samples (or fewer if not available)
                n_samples_to_show = min(50, len(class_features))
                
                # Compute deviations per concept for visualization
                gmm = tcd.prototype_discovery.gmms[class_id]
                
                # Get samples sorted by deviation magnitude
                sorted_indices = np.argsort(class_deviations_all)[-n_samples_to_show:][::-1]
                
                deviations_matrix = []
                for idx in sorted_indices:
                    sample_features = class_features[idx]
                    proto_idx = class_assignments[idx]
                    prototype_mean = gmm.means_[proto_idx]
                    deviation = sample_features.cpu().numpy() - prototype_mean
                    deviations_matrix.append(deviation)
                
                deviations_matrix = np.array(deviations_matrix)
                
                # Plot deviation matrix
                concept_labels = [f"F{i}" for i in range(features.shape[1])]
                sample_labels = [f"S{i}" for i in sorted_indices]
                
                class_name = "OK" if class_id == 0 else "NOK"
                fig = plot_deviation_matrix(
                    deviations=deviations_matrix,
                    concept_labels=concept_labels,
                    sample_labels=sample_labels,
                    title=f'Deviation Matrix - {class_name} (Top {n_samples_to_show} by deviation magnitude)'
                )
                
                deviation_path = os.path.join(output_path, f'deviation_matrix_class_{class_id}.png')
                fig.savefig(deviation_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved deviation matrix to {deviation_path}")
            
            # Add robustness results to evaluation
            evaluation['robustness'] = robustness_results
            
        except Exception as e:
            print(f"  Warning: Could not perform robustness analysis: {e}")
            traceback.print_exc()
    
    # Prototype sample visualization
    if config.get('evaluation', {}).get('prototype_sample_vis', True):
        print("\n" + "="*60)
        print("PROTOTYPE SAMPLE VISUALIZATION")
        print("="*60)
        try:
            from tcd.visualization import plot_prototype_samples, plot_prototype_gallery, plot_pcx_prototype_concept_grid
            from tcd.composites import CNCValidatedComposite
            from tcd.attribution import TimeSeriesCondAttribution

            composite = CNCValidatedComposite()
            attributor = TimeSeriesCondAttribution(model, no_param_grad=True)
            proto_vis_dir = os.path.join(output_path, 'prototype_samples')
            os.makedirs(proto_vis_dir, exist_ok=True)

            # Collect dataset indices per class (class 0 first, then class 1 — same order
            # as features were extracted in discover_concepts.py)
            class_dataset_indices = {0: [], 1: []}
            for idx, (_, lbl) in enumerate(dataset):
                lbl_int = int(lbl)
                if lbl_int in class_dataset_indices:
                    class_dataset_indices[lbl_int].append(idx)

            for class_id in [0, 1]:
                class_name = "OK" if class_id == 0 else "NOK"
                if class_id not in tcd.prototype_discovery.gmms:
                    continue

                gmm = tcd.prototype_discovery.gmms[class_id]
                n_proto = gmm.n_components
                top_k_proto = config.get('evaluation', {}).get('prototype_top_k', 6)

                proto_samples = tcd.prototype_discovery.find_prototypes(
                    class_id=class_id, top_k=top_k_proto
                )

                ds_indices_for_class = class_dataset_indices[class_id]
                class_features_np = features[labels == class_id].cpu().numpy()

                all_signals = {}
                all_heatmaps = {}
                all_distances = {}

                for proto_idx, sample_positions in proto_samples.items():
                    signals_proto = []
                    heatmaps_proto = []
                    distances_proto = []
                    proto_mean = gmm.means_[proto_idx]

                    for pos in sample_positions:
                        if pos >= len(ds_indices_for_class):
                            continue
                        ds_idx = ds_indices_for_class[pos]
                        signal_tensor, _ = dataset[ds_idx]
                        signal_np = signal_tensor.cpu().numpy() if hasattr(signal_tensor, 'cpu') else np.array(signal_tensor)

                        # Compute CRP heatmap (full, not conditional on specific filter)
                        data_p = signal_tensor.unsqueeze(0).to(device)
                        conditions = [{"y": class_id}]
                        try:
                            cond_heatmap, _, _, _ = attributor(
                                data_p, conditions, composite, record_layer=[]
                            )
                            heat_np = cond_heatmap.detach().cpu().numpy()[0]
                        except Exception:
                            heat_np = np.zeros_like(signal_np)

                        # Distance to prototype mean
                        if pos < len(class_features_np):
                            dist = float(np.linalg.norm(class_features_np[pos] - proto_mean))
                        else:
                            dist = 0.0

                        signals_proto.append(signal_np)
                        heatmaps_proto.append(heat_np)
                        distances_proto.append(dist)

                    if not signals_proto:
                        continue

                    all_signals[proto_idx] = signals_proto
                    all_heatmaps[proto_idx] = heatmaps_proto
                    all_distances[proto_idx] = distances_proto

                    # Individual prototype figure
                    fig = plot_prototype_samples(
                        signals=signals_proto,
                        heatmaps=heatmaps_proto,
                        prototype_idx=proto_idx,
                        sample_distances=distances_proto,
                        title=f'{class_name} Prototype {proto_idx}: Representative Samples'
                    )
                    fig_path = os.path.join(proto_vis_dir, f'proto_samples_class{class_id}_proto{proto_idx}.png')
                    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f"  Saved {fig_path}")

                    # PCX-style prototype+concept grid
                    try:
                        grid_fig = plot_pcx_prototype_concept_grid(
                            model=model,
                            dataset=dataset,
                            prototype_idx=proto_idx,
                            proto_mean=proto_mean,
                            closest_sample_signal=signals_proto[0],
                            closest_sample_heatmap=heatmaps_proto[0],
                            layer_name=layer_name,
                            composite=composite,
                            top_k=top_k_proto,
                            class_id=class_id,
                            device=device
                        )
                        grid_path = os.path.join(proto_vis_dir, f'pcx_grid_class{class_id}_proto{proto_idx}.png')
                        grid_fig.savefig(grid_path, dpi=150, bbox_inches='tight')
                        plt.close(grid_fig)
                        print(f"  Saved PCX grid {grid_path}")
                    except Exception as grid_err:
                        print(f"  Warning: Could not generate PCX grid for class {class_id} proto {proto_idx}: {grid_err}")

                # Gallery figure for the class
                if all_signals:
                    gallery_fig = plot_prototype_gallery(
                        all_signals=all_signals,
                        all_heatmaps=all_heatmaps,
                        all_distances=all_distances,
                        class_id=class_id
                    )
                    gallery_path = os.path.join(proto_vis_dir, f'proto_gallery_class{class_id}.png')
                    gallery_fig.savefig(gallery_path, dpi=150, bbox_inches='tight')
                    plt.close(gallery_fig)
                    print(f"  Saved gallery {gallery_path}")

        except Exception as e:
            print(f"  Warning: Could not generate prototype sample visualizations: {e}")
            traceback.print_exc()
    
    # Save evaluation
    '''
    # moved the save before robustness analysis to ensure we have something saved even if robustness fails
    os.makedirs(output_path, exist_ok=True)
    evaluation = {
        'variant': 'C',
        'layer_name': layer_name,
        'metrics': metrics,
        'importance_scores': importance_scores,
        'importance_scores_class_0': importance_scores_class_0,
        'importance_scores_class_1': importance_scores_class_1,
        'metrics_class_0': metrics_class_0,
        'metrics_class_1': metrics_class_1,
        'prototype_intervention': proto_results
    }
    '''
    
    with open(os.path.join(output_path, 'evaluation.pkl'), 'wb') as f:
        pickle.dump(evaluation, f)


    
    print(f"\n✓ Evaluation saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate discovered concepts")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--concepts', type=str, required=True,
                       help='Path to concept results from discover_concepts.py')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model checkpoint (overrides config)')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to data directory (overrides config)')
    parser.add_argument('--output', type=str, default='results/evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup
    model_path = args.model or config['model']['path']
    data_path = args.data or config['data']['path']
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = CNN1D_Wide()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model checkpoint not found at {model_path}")
        print("Using randomly initialized model (evaluation may not be meaningful)")
    
    # Load dataset
    print(f"Loading dataset from {data_path}...")
    if os.path.exists(data_path):
        dataset = VibrationDataset(data_path)
        print(f"Loaded {len(dataset)} samples")
    else:
        print(f"Warning: Data path not found at {data_path}")
        print("Cannot proceed without data.")
        return
    
    # Detect variant from results
    with open(os.path.join(args.concepts, 'results.pkl'), 'rb') as f:
        results = pickle.load(f)
    
    variant = results['variant']
    print(f"\nDetected variant: {variant}")
    
    # Run evaluation
    if variant == 'A':
        evaluate_variant_a(args.concepts, model, dataset, config, args.output)
    elif variant == 'C':
        evaluate_variant_c(args.concepts, model, dataset, config, args.output, device)
    else:
        print(f"Evaluation not implemented for variant {variant}")


if __name__ == "__main__":
    main()
