"""
Cross-machine robustness analysis for temporal concepts.

Tests concept stability and transferability across different machines
or data subsets to identify machine-specific vs universal patterns.

This connects to thesis findings of accuracy drop from 99.8% → 95.9% on M03.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from tcd.prototypes import TemporalPrototypeDiscovery


def cross_machine_analysis(
    features_train: torch.Tensor,
    labels_train: torch.Tensor,
    outputs_train: torch.Tensor,
    features_test: torch.Tensor,
    labels_test: torch.Tensor,
    outputs_test: torch.Tensor,
    machine_ids_train: Optional[np.ndarray] = None,
    machine_ids_test: Optional[np.ndarray] = None,
    n_prototypes: int = 4,
    covariance_type: str = 'diag',
    n_init: int = 5,
    max_iter: int = 200
) -> Dict:
    """
    Cross-machine concept analysis.
    
    Discovers concepts on training machines (subset A) and tests
    concept stability on test machines (subset B).
    
    This tests:
    - Which concepts transfer across machines
    - Which concepts are machine-specific
    - Prediction accuracy per prototype on new machines
    - Concept coverage differences across machines
    
    Args:
        features_train: Concept relevance vectors for training (N_train, n_concepts)
        labels_train: Labels for training samples (N_train,)
        outputs_train: Model outputs for training samples (N_train, n_classes)
        features_test: Concept relevance vectors for testing (N_test, n_concepts)
        labels_test: Labels for test samples (N_test,)
        outputs_test: Model outputs for test samples (N_test, n_classes)
        machine_ids_train: Optional machine IDs for training samples
        machine_ids_test: Optional machine IDs for test samples
        n_prototypes: Number of prototypes per class
        covariance_type: GMM covariance type
        n_init: Number of GMM initializations
        max_iter: Maximum GMM iterations
        
    Returns:
        Dictionary with cross-machine analysis results:
        - prototype_discovery: Fitted TemporalPrototypeDiscovery object
        - train_coverage: Coverage per prototype on training set
        - test_coverage: Coverage per prototype on test set
        - train_accuracy_per_proto: Accuracy per prototype on training set
        - test_accuracy_per_proto: Accuracy per prototype on test set
        - transfer_score: Overall concept transfer score
        - per_machine_results: Per-machine breakdown (if machine IDs provided)
    """
    print("\n" + "="*80)
    print("CROSS-MACHINE ROBUSTNESS ANALYSIS")
    print("="*80)
    
    # Fit prototypes on training data
    print("\nStep 1: Discovering prototypes on training machines...")
    prototype_discovery = TemporalPrototypeDiscovery(
        n_prototypes=n_prototypes,
        covariance_type=covariance_type,
        n_init=n_init,
        max_iter=max_iter
    )
    
    prototype_discovery.fit(features_train, labels_train, outputs_train)
    
    # Analyze training set
    print("\nStep 2: Analyzing training set...")
    train_results = _analyze_prototype_performance(
        prototype_discovery,
        features_train,
        labels_train,
        outputs_train,
        "Training"
    )
    
    # Analyze test set
    print("\nStep 3: Testing on new machines...")
    test_results = _analyze_prototype_performance(
        prototype_discovery,
        features_test,
        labels_test,
        outputs_test,
        "Test"
    )
    
    # Compute transfer scores
    print("\nStep 4: Computing transfer scores...")
    transfer_score = _compute_transfer_score(train_results, test_results)
    
    # Per-machine analysis if machine IDs provided
    per_machine_results = {}
    if machine_ids_train is not None and machine_ids_test is not None:
        print("\nStep 5: Per-machine breakdown...")
        unique_machines_train = np.unique(machine_ids_train)
        unique_machines_test = np.unique(machine_ids_test)
        
        print(f"Training machines: {unique_machines_train}")
        print(f"Test machines: {unique_machines_test}")
        
        # Analyze each test machine separately
        for machine_id in unique_machines_test:
            mask = machine_ids_test == machine_id
            machine_features = features_test[mask]
            machine_labels = labels_test[mask]
            machine_outputs = outputs_test[mask]
            
            if len(machine_features) > 0:
                machine_results = _analyze_prototype_performance(
                    prototype_discovery,
                    machine_features,
                    machine_labels,
                    machine_outputs,
                    f"Machine {machine_id}"
                )
                per_machine_results[machine_id] = machine_results
    
    # Summary
    print("\n" + "="*80)
    print("CROSS-MACHINE ANALYSIS SUMMARY")
    print("="*80)
    print(f"Transfer Score: {transfer_score:.3f}")
    print(f"  (1.0 = perfect transfer, <0.8 indicates machine-specific patterns)")
    
    return {
        'prototype_discovery': prototype_discovery,
        'train_results': train_results,
        'test_results': test_results,
        'transfer_score': transfer_score,
        'per_machine_results': per_machine_results
    }


def _analyze_prototype_performance(
    prototype_discovery: TemporalPrototypeDiscovery,
    features: torch.Tensor,
    labels: torch.Tensor,
    outputs: torch.Tensor,
    dataset_name: str = "Dataset"
) -> Dict:
    """
    Analyze prototype coverage and accuracy on a dataset.
    
    Args:
        prototype_discovery: Fitted TemporalPrototypeDiscovery
        features: Concept relevance vectors (N, n_concepts)
        labels: True labels (N,)
        outputs: Model outputs (N, n_classes)
        dataset_name: Name for logging
        
    Returns:
        Dictionary with coverage and accuracy per prototype
    """
    predictions = outputs.argmax(dim=1)
    unique_classes = torch.unique(labels).cpu().numpy()
    
    results = {}
    
    for class_id in unique_classes:
        class_mask = labels == class_id
        class_features = features[class_mask]
        class_labels = labels[class_mask]
        class_predictions = predictions[class_mask]
        
        if class_id not in prototype_discovery.gmms:
            continue
        
        # Assign to prototypes
        assignments = prototype_discovery.assign_prototype(class_features, class_id)
        
        # Coverage: percentage of samples per prototype
        coverage = np.bincount(assignments, minlength=prototype_discovery.n_prototypes)
        coverage_pct = (coverage / coverage.sum()) * 100
        
        # Accuracy per prototype
        accuracy_per_proto = np.zeros(prototype_discovery.n_prototypes)
        
        for proto_idx in range(prototype_discovery.n_prototypes):
            proto_mask = assignments == proto_idx
            if proto_mask.sum() > 0:
                proto_labels = class_labels[proto_mask]
                proto_predictions = class_predictions[proto_mask]
                accuracy = (proto_labels == proto_predictions).float().mean().item()
                accuracy_per_proto[proto_idx] = accuracy
        
        results[class_id] = {
            'coverage': coverage,
            'coverage_pct': coverage_pct,
            'accuracy_per_proto': accuracy_per_proto,
            'overall_accuracy': (class_labels == class_predictions).float().mean().item()
        }
        
        print(f"\n{dataset_name} - Class {class_id}:")
        print(f"  Overall accuracy: {results[class_id]['overall_accuracy']:.3f}")
        print(f"  Prototype coverage:")
        for proto_idx in range(prototype_discovery.n_prototypes):
            print(f"    Prototype {proto_idx}: {coverage_pct[proto_idx]:>5.1f}% "
                  f"(accuracy: {accuracy_per_proto[proto_idx]:.3f})")
    
    return results


def _compute_transfer_score(
    train_results: Dict,
    test_results: Dict
) -> float:
    """
    Compute transfer score measuring concept stability across datasets.
    
    Transfer score = mean( min(test_cov, train_cov) / train_cov )
    
    High score (>0.8) = concepts transfer well
    Low score (<0.6) = machine-specific concepts that don't generalize
    
    Args:
        train_results: Results on training set
        test_results: Results on test set
        
    Returns:
        Transfer score between 0 and 1
    """
    transfer_scores = []
    
    for class_id in train_results:
        if class_id not in test_results:
            continue
        
        train_cov = train_results[class_id]['coverage_pct']
        test_cov = test_results[class_id]['coverage_pct']
        
        # For each prototype, compute how well coverage transfers
        for proto_idx in range(len(train_cov)):
            if train_cov[proto_idx] > 1.0:  # Only consider prototypes with >1% coverage in training
                # Transfer = how much of training coverage is preserved in test
                transfer = min(test_cov[proto_idx], train_cov[proto_idx]) / train_cov[proto_idx]
                transfer_scores.append(transfer)
    
    if len(transfer_scores) == 0:
        return 0.0
    
    return np.mean(transfer_scores)


def compare_prototype_distributions(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    prototype_discovery: TemporalPrototypeDiscovery,
    class_id: int
) -> Dict:
    """
    Compare prototype distributions between two datasets.
    
    Uses KL divergence to measure how different the prototype
    distributions are between dataset A and B.
    
    Args:
        features_a: Features from dataset A (N_a, n_concepts)
        features_b: Features from dataset B (N_b, n_concepts)
        prototype_discovery: Fitted TemporalPrototypeDiscovery
        class_id: Class to analyze
        
    Returns:
        Dictionary with KL divergence and distribution statistics
    """
    if class_id not in prototype_discovery.gmms:
        raise ValueError(f"No GMM fitted for class {class_id}")
    
    # Get prototype assignments
    assignments_a = prototype_discovery.assign_prototype(features_a, class_id)
    assignments_b = prototype_discovery.assign_prototype(features_b, class_id)
    
    # Compute distributions
    n_prototypes = prototype_discovery.n_prototypes
    dist_a = np.bincount(assignments_a, minlength=n_prototypes) / len(assignments_a)
    dist_b = np.bincount(assignments_b, minlength=n_prototypes) / len(assignments_b)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    dist_a = dist_a + epsilon
    dist_b = dist_b + epsilon
    
    # Normalize
    dist_a = dist_a / dist_a.sum()
    dist_b = dist_b / dist_b.sum()
    
    # KL divergence: D_KL(A || B) = sum(A * log(A/B))
    kl_div_ab = np.sum(dist_a * np.log(dist_a / dist_b))
    kl_div_ba = np.sum(dist_b * np.log(dist_b / dist_a))
    
    # Symmetric KL divergence
    kl_div_sym = (kl_div_ab + kl_div_ba) / 2
    
    return {
        'distribution_a': dist_a,
        'distribution_b': dist_b,
        'kl_divergence_ab': kl_div_ab,
        'kl_divergence_ba': kl_div_ba,
        'kl_divergence_symmetric': kl_div_sym,
        'interpretation': 'Low KL (<0.1) = similar distributions, High KL (>1.0) = very different'
    }


if __name__ == "__main__":
    # Test cross-machine analysis with synthetic data
    print("Testing cross-machine robustness analysis...")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Simulate training data (machines M01, M02)
    n_train = 200
    n_test = 100
    n_concepts = 64
    
    # Training: two machines with similar patterns
    features_train = torch.randn(n_train, n_concepts)
    labels_train = torch.randint(0, 2, (n_train,))
    outputs_train = torch.randn(n_train, 2)
    machine_ids_train = np.random.choice(['M01', 'M02'], n_train)
    
    # Test: one machine (M03) with slightly different distribution
    features_test = torch.randn(n_test, n_concepts) + 0.3  # Shift distribution
    labels_test = torch.randint(0, 2, (n_test,))
    outputs_test = torch.randn(n_test, 2)
    machine_ids_test = np.array(['M03'] * n_test)
    
    # Run analysis
    results = cross_machine_analysis(
        features_train, labels_train, outputs_train,
        features_test, labels_test, outputs_test,
        machine_ids_train, machine_ids_test,
        n_prototypes=3
    )
    
    print(f"\n✓ Cross-machine analysis complete!")
    print(f"Transfer score: {results['transfer_score']:.3f}")
