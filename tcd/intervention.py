"""
Concept intervention for causal testing.

Implements suppression and amplification of concept activations
to test their causal role in model predictions.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Callable, Dict
import numpy as np


class ConceptInterventionHook:
    """
    Hook for intervening on concept activations during forward pass.
    
    Allows suppression, amplification, or ablation of specific
    filter activations to test their causal importance.
    
    Usage:
        hook = ConceptInterventionHook(method='suppress', concept_indices=[3, 5])
        handle = model.conv1.register_forward_hook(hook)
        output = model(x)  # Concepts 3 and 5 are suppressed
        handle.remove()
    """
    
    def __init__(
        self,
        method: str = 'suppress',
        concept_indices: Optional[List[int]] = None,
        factor: float = 0.0
    ):
        """
        Initialize intervention hook.
        
        Args:
            method: Intervention method - 'suppress', 'amplify', or 'ablate'
            concept_indices: List of concept (channel) indices to intervene on
            factor: Scaling factor (0.0=full suppression, 2.0=double amplification)
        """
        self.method = method
        self.concept_indices = concept_indices or []
        self.factor = factor
        
        if method == 'suppress':
            self.factor = 0.0
        elif method == 'ablate':
            self.factor = 0.0
        # For 'amplify', use provided factor or default to 2.0
        elif method == 'amplify' and factor == 0.0:
            self.factor = 2.0
    
    def __call__(
        self,
        module: nn.Module,
        input: tuple,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Hook function called during forward pass.
        
        Args:
            module: Layer being hooked
            input: Input to the layer (tuple)
            output: Output from the layer (tensor)
            
        Returns:
            Modified output tensor
        """
        if len(self.concept_indices) == 0:
            return output
        
        # Clone output to avoid in-place modification
        modified_output = output.clone()
        
        # Intervene on specified channels
        for idx in self.concept_indices:
            if idx < output.shape[1]:  # Check channel exists
                if self.method in ['suppress', 'ablate']:
                    modified_output[:, idx, :] = self.factor
                elif self.method == 'amplify':
                    modified_output[:, idx, :] *= self.factor
        
        return modified_output


class PrototypeInterventionHook:
    """
    Hook for intervening on multiple concept channels simultaneously (prototype-level).
    
    Tests whether a complete prototype pattern (top-k filters) is causally necessary.
    Unlike single-filter intervention, this suppresses the entire characteristic
    pattern of a prototype.
    
    Usage:
        # Get top-k filters from prototype center μ
        top_filters = np.argsort(np.abs(prototype_mean))[-k:]
        
        hook = PrototypeInterventionHook(filter_indices=top_filters, method='suppress')
        handle = model.conv3.register_forward_hook(hook)
        output = model(x)  # Entire prototype pattern is suppressed
        handle.remove()
    """
    
    def __init__(
        self,
        filter_indices: List[int],
        method: str = 'suppress',
        factor: float = 0.0
    ):
        """
        Initialize prototype intervention hook.
        
        Args:
            filter_indices: List of filter indices to intervene on simultaneously
            method: Intervention method - 'suppress', 'amplify', or 'ablate'
            factor: Scaling factor (0.0=full suppression, 2.0=double amplification)
        """
        self.filter_indices = filter_indices
        self.method = method
        self.factor = factor
        
        if method == 'suppress':
            self.factor = 0.0
        elif method == 'ablate':
            self.factor = 0.0
        elif method == 'amplify' and factor == 0.0:
            self.factor = 2.0
    
    def __call__(
        self,
        module: nn.Module,
        input: tuple,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Hook function called during forward pass.
        
        Args:
            module: Layer being hooked
            input: Input to the layer (tuple)
            output: Output from the layer (tensor)
            
        Returns:
            Modified output tensor
        """
        if len(self.filter_indices) == 0:
            return output
        
        # Clone output to avoid in-place modification
        modified_output = output.clone()
        
        # Intervene on all specified channels simultaneously
        for idx in self.filter_indices:
            if idx < output.shape[1]:  # Check channel exists
                if self.method in ['suppress', 'ablate']:
                    modified_output[:, idx, :] = self.factor
                elif self.method == 'amplify':
                    modified_output[:, idx, :] *= self.factor
        
        return modified_output


def prototype_intervention_analysis(
    model: nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
    prototype_discovery,  # TemporalPrototypeDiscovery instance
    features: torch.Tensor,
    layer_name: str,
    top_k: int = 5,
    method: str = 'suppress'
) -> Dict[int, List[Dict]]:
    """
    Perform prototype-level intervention analysis.
    
    For each GMM prototype, suppress the top-k filters characterizing it
    simultaneously and measure prediction change on samples assigned to
    that prototype.
    
    Args:
        model: PyTorch model
        data: Input data of shape (N, C, T)
        labels: True labels of shape (N,)
        prototype_discovery: TemporalPrototypeDiscovery instance with fitted GMMs
        features: Concept relevance vectors of shape (N, n_concepts)
        layer_name: Layer to intervene on
        top_k: Number of top filters to suppress per prototype
        method: Intervention method ('suppress', 'amplify')
        
    Returns:
        Dictionary mapping class_id -> list of prototype intervention results
        Each result contains: {
            'prototype_idx': int,
            'top_filters': list of filter indices,
            'n_samples': number of samples assigned to prototype,
            'mean_prob_change': average prediction probability change,
            'flip_rate': fraction of samples that changed prediction
        }
    """
    model.eval()
    device = next(model.parameters()).device
    
    results = {}
    
    # Get unique classes
    unique_classes = torch.unique(labels).cpu().numpy()
    
    for class_id in unique_classes:
        if class_id not in prototype_discovery.gmms:
            continue
        
        print(f"\n{'='*60}")
        print(f"Class {class_id} Prototype Intervention Analysis")
        print(f"{'='*60}")
        
        gmm = prototype_discovery.gmms[class_id]
        class_mask = labels == class_id
        class_data = data[class_mask]
        class_features = features[class_mask]
        
        # Assign samples to prototypes
        prototype_assignments = prototype_discovery.assign_prototype(class_features, class_id)
        
        class_results = []
        
        for proto_idx in range(gmm.n_components):
            # Get prototype center μ
            prototype_mean = gmm.means_[proto_idx]  # shape: (n_concepts,)
            
            # Find top-k filters by absolute magnitude
            top_filter_indices = np.argsort(np.abs(prototype_mean))[-top_k:][::-1]
            
            # Get samples assigned to this prototype
            proto_mask = prototype_assignments == proto_idx
            proto_data = class_data[proto_mask]
            
            if len(proto_data) == 0:
                print(f"  Prototype {proto_idx}: No samples assigned, skipping")
                continue
            
            proto_data = proto_data.to(device)
            
            # Original prediction
            with torch.no_grad():
                original_output = model(proto_data)
                original_probs = torch.softmax(original_output, dim=1)
                original_class_probs = original_probs[:, class_id]
            
            # Apply prototype intervention
            hook = PrototypeInterventionHook(
                filter_indices=top_filter_indices.tolist(),
                method=method
            )
            
            # Find target layer
            target_module = None
            for name, module in model.named_modules():
                if name == layer_name:
                    target_module = module
                    break
            
            if target_module is None:
                raise ValueError(f"Layer {layer_name} not found in model")
            
            # Register hook and get intervened prediction
            handle = target_module.register_forward_hook(hook)
            
            with torch.no_grad():
                intervened_output = model(proto_data)
                intervened_probs = torch.softmax(intervened_output, dim=1)
                intervened_class_probs = intervened_probs[:, class_id]
            
            handle.remove()
            
            # Compute metrics
            prob_change = (intervened_class_probs - original_class_probs).cpu().numpy()
            mean_prob_change = prob_change.mean()
            
            prediction_flip = (original_output.argmax(1) != intervened_output.argmax(1)).float()
            flip_rate = prediction_flip.mean().item()
            
            result = {
                'prototype_idx': proto_idx,
                'top_filters': top_filter_indices.tolist(),
                'top_filter_values': prototype_mean[top_filter_indices].tolist(),
                'n_samples': len(proto_data),
                'mean_prob_change': mean_prob_change,
                'flip_rate': flip_rate,
                'std_prob_change': prob_change.std()
            }
            
            class_results.append(result)
            
            # Print results
            print(f"  Prototype {proto_idx}:")
            print(f"    Top-{top_k} filters: {top_filter_indices.tolist()}")
            print(f"    Filter magnitudes: {np.abs(prototype_mean[top_filter_indices])}")
            print(f"    Samples assigned: {len(proto_data)}")
            print(f"    Mean probability change: {mean_prob_change:.4f}")
            print(f"    Prediction flip rate: {flip_rate:.2%}")
        
        results[int(class_id)] = class_results
    
    return results


def compute_deviation(
    features: torch.Tensor,
    prototype_mean: np.ndarray
) -> torch.Tensor:
    """
    Compute deviation Δ(ν) = ν - μ for each sample from prototype center.
    
    Shows how each sample differs from the prototype in concept space.
    Positive values = filter is stronger than prototype average.
    Negative values = filter is weaker than prototype average.
    
    Args:
        features: Concept relevance vectors of shape (N, n_concepts)
        prototype_mean: Prototype center μ of shape (n_concepts,)
        
    Returns:
        Deviations of shape (N, n_concepts)
    """
    mean_tensor = torch.from_numpy(prototype_mean).float()
    deviations = features - mean_tensor[None, :]
    return deviations


def compute_intervention_effect(
    model: nn.Module,
    data: torch.Tensor,
    target: int,
    layer_name: str,
    concept_indices: List[int],
    method: str = 'suppress',
    factor: float = 0.0
) -> dict:
    """
    Compute effect of concept intervention on prediction.
    
    Measures change in model output when intervening on specific concepts.
    
    Args:
        model: PyTorch model
        data: Input data of shape (B, C, T)
        target: Target class
        layer_name: Layer to intervene on (e.g., 'conv1', 'conv2')
        concept_indices: Concepts to intervene on
        method: Intervention method
        factor: Scaling factor
        
    Returns:
        Dictionary with original and intervened predictions
    """
    model.eval()
    
    # Get original prediction
    with torch.no_grad():
        original_output = model(data)
        original_probs = torch.softmax(original_output, dim=1)
    
    # Apply intervention hook
    hook = ConceptInterventionHook(method=method, concept_indices=concept_indices, factor=factor)
    
    # Find the target layer
    target_module = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_module = module
            break
    
    if target_module is None:
        raise ValueError(f"Layer {layer_name} not found in model")
    
    # Register hook and get intervened prediction
    handle = target_module.register_forward_hook(hook)
    
    with torch.no_grad():
        intervened_output = model(data)
        intervened_probs = torch.softmax(intervened_output, dim=1)
    
    handle.remove()
    
    # Compute metrics
    prob_change = intervened_probs[:, target] - original_probs[:, target]
    prediction_flip = (original_output.argmax(1) != intervened_output.argmax(1)).float()
    
    return {
        'original_probs': original_probs.cpu().numpy(),
        'intervened_probs': intervened_probs.cpu().numpy(),
        'prob_change': prob_change.cpu().numpy(),
        'prediction_flip': prediction_flip.cpu().numpy(),
        'mean_prob_change': prob_change.mean().item(),
        'flip_rate': prediction_flip.mean().item()
    }


def measure_concept_importance(
    model: nn.Module,
    dataset,
    layer_name: str,
    n_concepts: int,
    target_class: int = 0,
    method: str = 'suppress',
    batch_size: int = 32
) -> np.ndarray:
    """
    Measure importance of each concept via intervention.
    
    Tests each concept individually by suppressing it and measuring
    the effect on predictions for the target class.
    
    Args:
        model: PyTorch model
        dataset: Dataset to test on
        layer_name: Layer to intervene on
        n_concepts: Number of concepts (channels) at that layer
        target_class: Class to measure importance for
        method: Intervention method
        batch_size: Batch size for testing
        
    Returns:
        Array of shape (n_concepts,) with importance scores
    """
    from torch.utils.data import DataLoader
    
    model.eval()
    
    # Store importance for each concept
    importance_scores = np.zeros(n_concepts)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    # Test each concept
    for concept_idx in range(n_concepts):
        total_prob_change = 0
        n_samples = 0
        
        for data, labels in dataloader:
            # Only test samples from target class
            mask = labels == target_class
            if mask.sum() == 0:
                continue

            data_target = data[mask].to(next(model.parameters()).device)
            
            # Measure intervention effect
            result = compute_intervention_effect(
                model=model,
                data=data_target,
                target=target_class,
                layer_name=layer_name,
                concept_indices=[concept_idx],
                method=method
            )
            
            # Accumulate probability change (negative = important concept)
            total_prob_change += result['prob_change'].sum()
            n_samples += len(data_target)
        
        if n_samples > 0:
            # Average probability drop (negate so higher = more important)
            importance_scores[concept_idx] = -total_prob_change / n_samples
    
    return importance_scores


if __name__ == "__main__":
    from models.cnn1d_model import CNN1D_Wide
    
    # Test intervention on synthetic data
    model = CNN1D_Wide()
    model.eval()
    
    x = torch.randn(4, 3, 2000)
    target = 0
    
    # Test suppressing first 2 concepts
    result = compute_intervention_effect(
        model=model,
        data=x,
        target=target,
        layer_name='conv1',
        concept_indices=[0, 1],
        method='suppress'
    )
    
    print(f"Original probs shape: {result['original_probs'].shape}")
    print(f"Mean prob change: {result['mean_prob_change']:.4f}")
    print(f"Flip rate: {result['flip_rate']:.2%}")
    
    print("✓ Intervention test passed!")
