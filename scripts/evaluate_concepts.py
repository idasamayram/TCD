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

import argparse
import os
import yaml
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader

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
    
    stability = compute_stability(concept_relevances.numpy(), labels)
    purity = compute_concept_purity(concept_relevances.numpy())
    
    print(f"Stability: {stability:.3f}")
    print(f"Purity: {purity:.3f}")
    
    # Save evaluation
    os.makedirs(output_path, exist_ok=True)
    evaluation = {
        'variant': 'A',
        'stability': stability,
        'purity': purity,
        'importance': importance
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
    
    # Measure concept importance via intervention
    print(f"\nMeasuring importance of {n_concepts} concepts via intervention...")
    print("Testing on class 0 samples...")
    
    model.to(device)
    model.eval()
    
    # Create subset of dataset for class 0
    class_0_indices = [i for i, (_, label) in enumerate(dataset) if label == 0]
    class_0_subset = torch.utils.data.Subset(dataset, class_0_indices[:min(100, len(class_0_indices))])
    
    if len(class_0_subset) > 0:
        importance_scores = measure_concept_importance(
            model=model,
            dataset=class_0_subset,
            layer_name=layer_name,
            n_concepts=n_concepts,
            target_class=0,
            method='suppress',
            batch_size=config['analysis']['batch_size']
        )
        
        print(f"\nTop 5 most important concepts:")
        top_indices = np.argsort(importance_scores)[-5:][::-1]
        for idx in top_indices:
            print(f"  Concept {idx}: {importance_scores[idx]:.4f}")
    else:
        print("Warning: No class 0 samples found for intervention test")
        importance_scores = np.zeros(n_concepts)
    
    # Compute evaluation metrics
    print("\nComputing evaluation metrics...")
    
    # For faithfulness, we need intervention effects
    # Use importance scores as proxy for intervention effects
    intervention_effects = importance_scores
    
    # Get relevance scores (mean of features)
    relevance_scores = features.abs().mean(dim=0).numpy()
    
    from tcd.evaluation import compute_faithfulness, compute_stability, compute_concept_purity
    
    faithfulness = compute_faithfulness(relevance_scores, intervention_effects)
    stability = compute_stability(features.numpy(), labels.numpy())
    purity = compute_concept_purity(features.numpy())
    
    # Get prototype coverage
    class_0_features = features[labels == 0]
    if len(class_0_features) > 0:
        assignments = tcd.assign_prototype(class_0_features, class_id=0)
        from tcd.evaluation import compute_prototype_coverage
        coverage_metrics = compute_prototype_coverage(assignments, tcd.n_prototypes)
    else:
        coverage_metrics = {'coverage': 0, 'balance': 0, 'max_coverage': 0}
    
    # Print report
    metrics = {
        'faithfulness': faithfulness,
        'stability': stability,
        'purity': purity,
        **coverage_metrics
    }
    
    print_evaluation_report(metrics)
    
    # Save evaluation
    os.makedirs(output_path, exist_ok=True)
    evaluation = {
        'variant': 'C',
        'layer_name': layer_name,
        'metrics': metrics,
        'importance_scores': importance_scores
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
        dataset = VibrationDataset(data_path, split='test')
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
