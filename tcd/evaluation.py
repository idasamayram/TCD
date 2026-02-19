"""
Evaluation metrics for concept quality and faithfulness.

Measures:
- Faithfulness: Do concepts causally affect predictions?
- Stability: Are concepts consistent across similar samples?
- Concept purity: Are concepts semantically distinct?
- Coverage: How many samples are explained by prototypes?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def compute_faithfulness_prototype_level(
    proto_results: Dict,
    prototype_discovery,
    importance_scores_per_class: Optional[Dict[int, np.ndarray]] = None,
    top_k: int = 5
) -> Dict:
    """
    Compute prototype-level faithfulness metric.

    Instead of correlating relevance vs intervention effect across ALL filters,
    this function focuses on only the top-K most important filters for each
    prototype (identified from the GMM mean). This gives a more meaningful
    faithfulness score for prototypes that rely on a small subset of filters.

    For each prototype:
    1. Take the top-K filters by absolute magnitude from the prototype's GMM mean
    2. For those K filters, compute Spearman correlation between:
       - Relevance: the absolute GMM mean values at those K positions
       - Intervention effect: the per-filter importance scores at those K positions
         (from ``importance_scores_per_class``), or if not provided, the GMM mean
         magnitudes are used as a self-consistency check.
    3. Average correlations across all prototypes per class, then across classes.

    Args:
        proto_results: Dictionary from ``prototype_intervention_analysis()``
            mapping class_id -> list of prototype result dicts.  Each result dict
            must contain ``'prototype_idx'``.
        prototype_discovery: A fitted ``TemporalPrototypeDiscovery`` instance
            whose ``gmms`` attribute is used to look up the prototype means.
        importance_scores_per_class: Optional dict mapping class_id ->
            per-filter importance scores array of shape ``(n_concepts,)``.
            When provided, the Spearman correlation is computed between the GMM
            mean magnitudes (relevance proxy) and these intervention effects for
            the top-K filters.  When ``None``, the function falls back to using
            the GMM mean magnitudes for both (measuring internal prototype
            structure consistency).
        top_k: Number of top filters to consider per prototype.

    Returns:
        Dictionary with keys:
        - ``'per_prototype'``: {class_id: {proto_idx: spearman_r}}
        - ``'per_class'``: {class_id: mean_spearman_r}
        - ``'mean'``: overall mean spearman_r across all prototypes
    """
    per_prototype: Dict[int, Dict[int, float]] = {}
    per_class: Dict[int, float] = {}
    all_corrs: List[float] = []

    # Normalize importance_scores_per_class keys to int once at entry
    if importance_scores_per_class is not None:
        importance_scores_per_class = {int(k): v for k, v in importance_scores_per_class.items()}

    for class_id, class_results in proto_results.items():
        class_id = int(class_id)
        if class_id not in prototype_discovery.gmms:
            continue

        gmm = prototype_discovery.gmms[class_id]
        class_corrs: List[float] = []
        per_prototype[class_id] = {}

        # Per-filter importance scores for this class (if available)
        class_importance = None
        if importance_scores_per_class is not None:
            class_importance = importance_scores_per_class.get(class_id)

        for result in class_results:
            proto_idx = result['prototype_idx']
            prototype_mean = gmm.means_[proto_idx]  # shape: (n_concepts,)

            # Top-K filter indices by absolute magnitude of the prototype mean
            top_filter_indices = np.argsort(np.abs(prototype_mean))[-top_k:]

            # Relevance scores for those filters (absolute prototype mean values)
            relevance_topk = np.abs(prototype_mean[top_filter_indices])

            # Intervention effects: per-filter importance scores if provided,
            # else fall back to prototype mean magnitudes
            if class_importance is not None:
                intervention_topk = np.abs(class_importance[top_filter_indices])
            else:
                intervention_topk = relevance_topk.copy()

            # Only compute correlation if we have enough distinct points
            if len(top_filter_indices) >= 3:
                corr, _ = spearmanr(relevance_topk, intervention_topk)
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0

            per_prototype[class_id][proto_idx] = float(corr)
            class_corrs.append(corr)
            all_corrs.append(corr)

        per_class[class_id] = float(np.mean(class_corrs)) if class_corrs else 0.0

    mean_corr = float(np.mean(all_corrs)) if all_corrs else 0.0

    return {
        'per_prototype': per_prototype,
        'per_class': per_class,
        'mean': mean_corr
    }


def compute_incremental_faithfulness(
    model,
    dataset,
    layer_name: str,
    concept_relevance_vectors: np.ndarray,
    n_samples: int = 100,
    n_steps: Optional[int] = None,
    target_class: int = 0,
    batch_size: int = 32,
    device: str = 'cpu'
) -> Dict:
    """
    PCX-style incremental faithfulness with AUC.

    Adapted from PCX ``experiments/faithfulness/faithfulness.py`` (lines 161-189)
    for 1D time-series.  Concepts (filters) are sorted by their mean relevance
    across the provided samples and suppressed one-by-one from most to least
    relevant.  At each step the mean logit change vs. the baseline is recorded.
    The Area Under the Curve (AUC) via ``np.trapz`` is the summary statistic —
    higher AUC means suppressing the most-relevant concepts causes larger logit
    changes, indicating the concepts are more faithful.

    Args:
        model: PyTorch model (already on the correct device).
        dataset: Dataset used to draw samples from ``target_class``.
        layer_name: Name of the layer to intervene on.
        concept_relevance_vectors: Array of shape ``(n_samples, n_concepts)``
            containing concept relevance vectors for ``target_class`` samples.
        n_samples: Maximum number of samples to use for the measurement.
        n_steps: Number of suppression steps (defaults to n_concepts).
        target_class: Class whose logit is monitored.
        batch_size: Batch size for forward passes.
        device: Torch device string.

    Returns:
        Dictionary with keys:
        - ``'auc'``: scalar AUC value (higher = more faithful)
        - ``'steps'``: list of k values (x-axis)
        - ``'mean_logit_changes'``: list of mean logit changes at each step
        - ``'n_samples_used'``: actual number of samples used
    """
    from torch.utils.data import DataLoader, Subset
    from tcd.intervention import ConceptInterventionHook

    model.eval()
    torch_device = torch.device(device)
    model.to(torch_device)

    # Normalize relevance vectors to always be 2D: (n_samples, n_concepts)
    concept_relevance_vectors = np.atleast_2d(concept_relevance_vectors)
    n_concepts = concept_relevance_vectors.shape[1]
    mean_relevance = np.abs(concept_relevance_vectors).mean(axis=0)

    # Sort concepts by mean absolute relevance (most relevant first)
    sorted_concept_indices = np.argsort(mean_relevance)[::-1]

    # Collect target-class sample indices
    class_indices = [i for i, (_, lbl) in enumerate(dataset) if lbl == target_class]
    class_indices = class_indices[:n_samples]
    if len(class_indices) == 0:
        return {'auc': 0.0, 'steps': [], 'mean_logit_changes': [], 'n_samples_used': 0}

    subset = Subset(dataset, class_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    # Collect all data and compute baseline logits
    all_data: List[torch.Tensor] = []
    for batch_x, _ in loader:
        all_data.append(batch_x)
    all_data_tensor = torch.cat(all_data).to(torch_device)

    with torch.no_grad():
        baseline_output = model(all_data_tensor)
    baseline_logits = baseline_output[:, target_class].cpu().numpy()  # (N,)

    # Find target layer module
    target_module = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_module = module
            break
    if target_module is None:
        raise ValueError(f"Layer '{layer_name}' not found in model")

    if n_steps is None:
        n_steps = n_concepts

    steps = list(range(1, min(n_steps, n_concepts) + 1))
    mean_logit_changes: List[float] = []

    for k in steps:
        # Suppress the top-k most relevant concepts
        suppressed_indices = sorted_concept_indices[:k].tolist()
        hook = ConceptInterventionHook(method='suppress', concept_indices=suppressed_indices)
        handle = target_module.register_forward_hook(hook)

        with torch.no_grad():
            intervened_output = model(all_data_tensor)
        handle.remove()

        intervened_logits = intervened_output[:, target_class].cpu().numpy()
        logit_change = np.abs(baseline_logits - intervened_logits).mean()
        mean_logit_changes.append(float(logit_change))

    # Compute AUC via trapezoidal rule
    auc = float(np.trapz(mean_logit_changes, x=steps)) if len(steps) > 1 else 0.0

    return {
        'auc': auc,
        'steps': steps,
        'mean_logit_changes': mean_logit_changes,
        'n_samples_used': len(class_indices)
    }



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


def compute_class_weighted_average(
    per_class_values: Dict[int, float],
    class_weights: Optional[np.ndarray] = None
) -> float:
    """
    Compute class-weighted average of per-class metric values.
    
    Args:
        per_class_values: Dictionary mapping class_id -> metric value
        class_weights: Optional class weights array. If None, uses uniform weights.
        
    Returns:
        Weighted average of metric values
    """
    if class_weights is None:
        # Uniform weights
        return np.mean(list(per_class_values.values()))
    
    # Apply class weights
    weighted_sum = 0.0
    total_weight = 0.0
    for class_id, value in per_class_values.items():
        weight = class_weights[class_id]
        weighted_sum += value * weight
        total_weight += weight
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def evaluate_concept_quality(
    concept_vectors: np.ndarray,
    labels: np.ndarray,
    importance_scores: np.ndarray,
    intervention_effects: np.ndarray,
    prototype_assignments: Optional[np.ndarray] = None,
    n_prototypes: Optional[int] = None,
    class_weights: Optional[np.ndarray] = None
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
        class_weights: Optional class weights for computing weighted averages
        
    Returns:
        Dictionary of evaluation metrics including per-class and weighted averages
    """
    metrics = {}
    
    # Overall metrics (computed on all samples)
    metrics['faithfulness'] = compute_faithfulness(importance_scores, intervention_effects)
    metrics['stability'] = compute_stability(concept_vectors, labels)
    metrics['purity'] = compute_concept_purity(concept_vectors)
    
    # Prototype coverage (if applicable)
    if prototype_assignments is not None and n_prototypes is not None:
        coverage_metrics = compute_prototype_coverage(prototype_assignments, n_prototypes)
        metrics.update(coverage_metrics)
    
    # Compute per-class metrics
    unique_classes = np.unique(labels)
    per_class_metrics = {}
    
    for class_id in unique_classes:
        class_mask = labels == class_id
        class_concept_vectors = concept_vectors[class_mask]
        class_labels = labels[class_mask]
        
        if len(class_concept_vectors) > 0:
            per_class_metrics[class_id] = {
                'stability': compute_stability(class_concept_vectors, class_labels),
                'purity': compute_concept_purity(class_concept_vectors)
            }
            
            # Per-class prototype coverage if applicable
            if prototype_assignments is not None and n_prototypes is not None:
                class_assignments = prototype_assignments[class_mask]
                class_coverage = compute_prototype_coverage(class_assignments, n_prototypes)
                per_class_metrics[class_id].update(class_coverage)
    
    # Store per-class metrics in the main metrics dict
    for class_id, class_metrics in per_class_metrics.items():
        for metric_name, value in class_metrics.items():
            metrics[f'{metric_name}_class_{class_id}'] = value
    
    # Compute class-weighted averages if weights provided
    if class_weights is not None and len(per_class_metrics) > 0:
        # Weighted stability
        stability_per_class = {cid: m['stability'] for cid, m in per_class_metrics.items()}
        metrics['stability_weighted'] = compute_class_weighted_average(stability_per_class, class_weights)
        
        # Weighted purity
        purity_per_class = {cid: m['purity'] for cid, m in per_class_metrics.items()}
        metrics['purity_weighted'] = compute_class_weighted_average(purity_per_class, class_weights)
        
        # Weighted coverage metrics if available
        if prototype_assignments is not None and n_prototypes is not None:
            for metric_name in ['coverage', 'balance', 'max_coverage']:
                if all(metric_name in m for m in per_class_metrics.values()):
                    values_per_class = {cid: m[metric_name] for cid, m in per_class_metrics.items()}
                    metrics[f'{metric_name}_weighted'] = compute_class_weighted_average(values_per_class, class_weights)
    
    return metrics


def print_evaluation_report(metrics: Dict[str, float], per_class_metrics: Dict[str, Dict[str, float]] = None):
    """
    Print formatted evaluation report.
    
    Args:
        metrics: Dictionary of overall evaluation metrics (includes weighted averages)
        per_class_metrics: Optional dictionary with 'class_0' and 'class_1' keys,
                          each containing a dictionary of per-class metrics
    """
    print("\n" + "="*60)
    print("CONCEPT EVALUATION REPORT")
    print("="*60)
    
    print("\nCore Metrics:")
    print(f"  Faithfulness:  {metrics.get('faithfulness', 0):.3f}  (correlation with causal effect)")
    print(f"  Stability:     {metrics.get('stability', 0):.3f}  (consistency across neighbors)")
    print(f"  Purity:        {metrics.get('purity', 0):.3f}  (concept distinctiveness)")
    
    # Show class-weighted metrics if available
    if 'stability_weighted' in metrics:
        print("\nClass-Weighted Metrics:")
        print(f"  Stability (weighted):     {metrics['stability_weighted']:.3f}")
        print(f"  Purity (weighted):        {metrics['purity_weighted']:.3f}")
    
    if 'coverage' in metrics:
        print("\nPrototype Metrics:")
        print(f"  Coverage:      {metrics['coverage']:.3f}  (fraction of prototypes used)")
        print(f"  Balance:       {metrics['balance']:.3f}  (assignment distribution)")
        print(f"  Max Coverage:  {metrics['max_coverage']:.3f}  (largest prototype fraction)")
        
        # Show weighted prototype metrics if available
        if 'coverage_weighted' in metrics:
            print("\n  Weighted Prototype Metrics:")
            print(f"    Coverage (weighted):      {metrics['coverage_weighted']:.3f}")
            print(f"    Balance (weighted):       {metrics['balance_weighted']:.3f}")
            print(f"    Max Coverage (weighted):  {metrics['max_coverage_weighted']:.3f}")
    
    # Print per-class comparison if provided
    if per_class_metrics:
        print("\n" + "="*60)
        print("PER-CLASS METRICS COMPARISON")
        print("="*60)
        
        class_0_metrics = per_class_metrics.get('class_0', {})
        class_1_metrics = per_class_metrics.get('class_1', {})
        
        print(f"\n{'Metric':<20} {'OK (Class 0)':<20} {'NOK (Class 1)':<20}")
        print("-"*60)
        
        # Core metrics
        for metric_name in ['faithfulness', 'stability', 'purity']:
            val_0 = class_0_metrics.get(metric_name, 0)
            val_1 = class_1_metrics.get(metric_name, 0)
            print(f"{metric_name.capitalize()+':':<20} {val_0:>19.3f} {val_1:>19.3f}")
        
        # Prototype metrics if available
        if 'coverage' in class_0_metrics or 'coverage' in class_1_metrics:
            print()
            # Use a mapping for consistent display names
            metric_display_names = {
                'coverage': 'Coverage:',
                'balance': 'Balance:',
                'max_coverage': 'Max Coverage:'
            }
            for metric_name in ['coverage', 'balance', 'max_coverage']:
                val_0 = class_0_metrics.get(metric_name, 0)
                val_1 = class_1_metrics.get(metric_name, 0)
                display_name = metric_display_names[metric_name]
                print(f"{display_name:<20} {val_0:>19.3f} {val_1:>19.3f}")
    
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
