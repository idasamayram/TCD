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
    
    # Compute evaluation metrics
    print("\n" + "="*60)
    
    from tcd.evaluation import compute_faithfulness, compute_stability, compute_concept_purity, compute_prototype_coverage
    
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
        coverage_metrics_class_0 = compute_prototype_coverage(assignments_class_0, tcd.n_prototypes)
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
        coverage_metrics_class_1 = compute_prototype_coverage(assignments_class_1, tcd.n_prototypes)
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
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Note: plot_prototype_samples requires loading actual signals from samples
    # For a complete implementation, we would load the closest samples to each prototype
    # and generate visualizations. For now we note this requirement.
    print("\nPrototype sample visualization:")
    print("  Note: Requires loading sample signals for closest prototypes")
    print("  This would be added in a complete implementation with dataset access")
    
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
        'prototype_intervention': proto_results
    }
    
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
