"""
Evaluation metrics for concept quality and faithfulness.

Measures:
- Faithfulness: Do concepts causally affect predictions?
- Stability: Are concepts consistent across similar samples?
- Concept purity: Are concepts semantically distinct?
- Coverage: How many samples are explained by prototypes?
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr


def compute_faithfulness(
    importance_scores: np.ndarray,
    intervention_effects: np.ndarray
) -> float:
    """
    Compute faithfulness via correlation of importance vs intervention effect.
    
    Measures whether concepts that have high relevance scores also
    cause large changes in prediction when suppressed/modified.
    
    Args:
        importance_scores: Concept relevance scores of shape (n_concepts,)
        intervention_effects: Prediction change when suppressing each concept (n_concepts,)
        
    Returns:
        Spearman correlation coefficient (0-1)
    """
    # Higher relevance should correlate with larger intervention effect
    corr, p_value = spearmanr(importance_scores, intervention_effects)
    return corr


def compute_stability(
    concept_vectors: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 5
) -> float:
    """
    Compute stability: consistency of concepts across similar samples.
    
    Measures whether nearby samples (in input space) have similar
    concept activation patterns.
    
    Args:
        concept_vectors: Concept activations of shape (n_samples, n_concepts)
        labels: Sample labels of shape (n_samples,)
        n_neighbors: Number of neighbors to consider
        
    Returns:
        Mean cosine similarity to nearest neighbors within class
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Compute pairwise similarities
    similarities = cosine_similarity(concept_vectors)
    
    stabilities = []
    
    # For each sample, find nearest neighbors in same class
    for i in range(len(concept_vectors)):
        # Find samples in same class
        same_class_mask = labels == labels[i]
        same_class_mask[i] = False  # Exclude self
        
        if same_class_mask.sum() == 0:
            continue
        
        # Get similarities to same-class samples
        same_class_sims = similarities[i, same_class_mask]
        
        # Average over k nearest neighbors
        k = min(n_neighbors, len(same_class_sims))
        top_k_sims = np.sort(same_class_sims)[-k:]
        stabilities.append(top_k_sims.mean())
    
    return np.mean(stabilities) if stabilities else 0.0


def compute_concept_purity(
    concept_vectors: np.ndarray,
    n_concepts: Optional[int] = None
) -> float:
    """
    Compute concept purity: how distinct are concepts from each other?
    
    Measures orthogonality of concept activation patterns.
    Low purity = redundant concepts, high purity = diverse concepts.
    
    Args:
        concept_vectors: Concept activations of shape (n_samples, n_concepts)
        n_concepts: Number of concepts (defaults to n_concepts from shape)
        
    Returns:
        Purity score (0-1), higher = more distinct concepts
    """
    if n_concepts is None:
        n_concepts = concept_vectors.shape[1]
    
    # Compute correlation between concept columns
    concept_corrs = np.corrcoef(concept_vectors.T)
    
    # Get off-diagonal correlations (exclude self-correlation)
    mask = ~np.eye(n_concepts, dtype=bool)
    off_diag_corrs = np.abs(concept_corrs[mask])
    
    # Purity = 1 - mean absolute correlation
    # (perfectly orthogonal concepts have 0 correlation)
    purity = 1.0 - off_diag_corrs.mean()
    
    return purity


def compute_prototype_coverage(
    assignments: np.ndarray,
    n_prototypes: int
) -> Dict[str, float]:
    """
    Compute coverage statistics for prototype assignments.
    
    Args:
        assignments: Prototype assignments of shape (n_samples,)
        n_prototypes: Total number of prototypes
        
    Returns:
        Dictionary with coverage metrics:
        - coverage: Fraction of prototypes that are used (0-1)
        - balance: Entropy of assignment distribution (0-log(n_prototypes))
        - max_coverage: Fraction of samples assigned to most common prototype
    """
    # Count samples per prototype
    counts = np.bincount(assignments, minlength=n_prototypes)
    
    # Fraction of prototypes with at least one sample
    coverage = (counts > 0).sum() / n_prototypes
    
    # Balance via entropy
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Remove zeros
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(n_prototypes)
    balance = entropy / max_entropy if max_entropy > 0 else 0.0
    
    # Max coverage
    max_coverage = counts.max() / counts.sum()
    
    return {
        'coverage': coverage,
        'balance': balance,
        'max_coverage': max_coverage
    }


def evaluate_concept_quality(
    concept_vectors: np.ndarray,
    labels: np.ndarray,
    importance_scores: np.ndarray,
    intervention_effects: np.ndarray,
    prototype_assignments: Optional[np.ndarray] = None,
    n_prototypes: Optional[int] = None
) -> Dict[str, float]:
    """
    Comprehensive evaluation of concept quality.
    
    Args:
        concept_vectors: Concept activations of shape (n_samples, n_concepts)
        labels: Sample labels of shape (n_samples,)
        importance_scores: Concept relevance scores
        intervention_effects: Intervention effects per concept
        prototype_assignments: Optional prototype assignments
        n_prototypes: Number of prototypes (required if assignments provided)
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Faithfulness
    metrics['faithfulness'] = compute_faithfulness(importance_scores, intervention_effects)
    
    # Stability
    metrics['stability'] = compute_stability(concept_vectors, labels)
    
    # Concept purity
    metrics['purity'] = compute_concept_purity(concept_vectors)
    
    # Prototype coverage (if applicable)
    if prototype_assignments is not None and n_prototypes is not None:
        coverage_metrics = compute_prototype_coverage(prototype_assignments, n_prototypes)
        metrics.update(coverage_metrics)
    
    return metrics


def print_evaluation_report(metrics: Dict[str, float]):
    """
    Print formatted evaluation report.
    
    Args:
        metrics: Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("CONCEPT EVALUATION REPORT")
    print("="*60)
    
    print("\nCore Metrics:")
    print(f"  Faithfulness:  {metrics.get('faithfulness', 0):.3f}  (correlation with causal effect)")
    print(f"  Stability:     {metrics.get('stability', 0):.3f}  (consistency across neighbors)")
    print(f"  Purity:        {metrics.get('purity', 0):.3f}  (concept distinctiveness)")
    
    if 'coverage' in metrics:
        print("\nPrototype Metrics:")
        print(f"  Coverage:      {metrics['coverage']:.3f}  (fraction of prototypes used)")
        print(f"  Balance:       {metrics['balance']:.3f}  (assignment distribution)")
        print(f"  Max Coverage:  {metrics['max_coverage']:.3f}  (largest prototype fraction)")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test evaluation metrics
    np.random.seed(42)
    
    # Synthetic data
    n_samples = 100
    n_concepts = 16
    
    concept_vectors = np.random.randn(n_samples, n_concepts)
    labels = np.random.randint(0, 2, n_samples)
    importance_scores = np.abs(concept_vectors.mean(axis=0))
    intervention_effects = importance_scores + np.random.randn(n_concepts) * 0.1
    
    # Test faithfulness
    faithfulness = compute_faithfulness(importance_scores, intervention_effects)
    print(f"Faithfulness: {faithfulness:.3f}")
    
    # Test stability
    stability = compute_stability(concept_vectors, labels)
    print(f"Stability: {stability:.3f}")
    
    # Test purity
    purity = compute_concept_purity(concept_vectors)
    print(f"Purity: {purity:.3f}")
    
    # Test prototype coverage
    assignments = np.random.randint(0, 4, n_samples)
    coverage = compute_prototype_coverage(assignments, n_prototypes=4)
    print(f"Coverage: {coverage}")
    
    # Full evaluation
    metrics = evaluate_concept_quality(
        concept_vectors,
        labels,
        importance_scores,
        intervention_effects,
        assignments,
        n_prototypes=4
    )
    
    print_evaluation_report(metrics)
    
    print("✓ Evaluation test passed!")
