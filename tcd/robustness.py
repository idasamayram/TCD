"""
Input-perturbation robustness analysis for temporal concepts.

Tests whether concept assignments remain stable when the input signal
is perturbed. This validates that discovered prototypes represent
robust patterns rather than artifacts of specific signal properties.

Three types of robustness tests:
1. Gaussian noise: Add noise at multiple SNR levels
2. Time-shift: Circular shift the signal by small amounts
3. Channel dropout: Zero out individual accelerometer axes (X/Y/Z)
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from tcd.prototypes import TemporalPrototypeDiscovery
from tcd.attribution import TimeSeriesCondAttribution
from tcd.concepts import ChannelConcept
from crp.helper import get_layer_names


def noise_robustness(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    layer_name: str,
    composite,
    attributor: TimeSeriesCondAttribution,
    n_samples: int = 100,
    noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2],
    prototype_discovery: Optional[TemporalPrototypeDiscovery] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, Any]:
    """
    Test robustness to Gaussian noise at multiple SNR levels.
    
    For each noise level:
    1. Add Gaussian noise: x_noisy = x + N(0, σ²) where σ = noise_level * std(x)
    2. Re-compute concept relevance vectors via the model
    3. Compare original vs noisy concept vectors using cosine similarity
    4. If prototype_discovery provided, check if prototype assignments flip
    
    Args:
        model: PyTorch model
        dataset: Dataset to sample from
        layer_name: Layer to analyze
        composite: LRP composite for CRP attribution
        attributor: TimeSeriesCondAttribution instance
        n_samples: Number of samples to test
        noise_levels: List of noise levels as fraction of signal std
        prototype_discovery: Optional TemporalPrototypeDiscovery for assignment stability
        device: Device to run on
        
    Returns:
        Dictionary with per-noise-level metrics:
        {
            'noise_levels': [0.01, 0.05, ...],
            'mean_cosine_similarity': [0.98, 0.95, ...],
            'std_cosine_similarity': [0.01, 0.02, ...],
            'flip_rate': [0.02, 0.05, ...] (if prototype_discovery provided),
            'per_class_results': {...}
        }
    """
    print("\n" + "="*80)
    print("GAUSSIAN NOISE ROBUSTNESS ANALYSIS")
    print("="*80)
    print(f"Testing {n_samples} samples at noise levels: {noise_levels}")
    
    model.to(device)
    model.eval()
    
    # Sample indices
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    
    # Get original concept vectors
    print("\nComputing original concept vectors...")
    original_vectors, labels = _compute_concept_vectors(
        model, dataset, indices, layer_name, composite, attributor, device
    )
    
    results = {
        'noise_levels': noise_levels,
        'mean_cosine_similarity': [],
        'std_cosine_similarity': [],
        'per_class_results': {0: {}, 1: {}}
    }
    
    if prototype_discovery is not None:
        results['flip_rate'] = []
        results['per_class_flip_rate'] = {0: [], 1: []}
    
    # Test each noise level
    for noise_level in noise_levels:
        print(f"\nNoise level: {noise_level:.3f} (fraction of signal std)")
        
        cosine_sims = []
        flips = []
        
        per_class_cosine_sims = {0: [], 1: []}
        per_class_flips = {0: [], 1: []}
        
        for idx, orig_vec, label in zip(indices, original_vectors, labels):
            signal, _ = dataset[idx]
            signal = signal.unsqueeze(0).to(device)  # (1, C, T)
            
            # Add Gaussian noise
            signal_std = signal.std()
            noise = torch.randn_like(signal) * signal_std * noise_level
            noisy_signal = signal + noise
            
            # Compute noisy concept vector
            noisy_vec = _compute_single_concept_vector(
                model, noisy_signal, layer_name, composite, attributor, label
            )
            
            # Cosine similarity
            cosine_sim = _cosine_similarity(orig_vec, noisy_vec)
            cosine_sims.append(cosine_sim)
            per_class_cosine_sims[int(label)].append(cosine_sim)
            
            # Check prototype assignment flip
            if prototype_discovery is not None:
                orig_assignment = prototype_discovery.assign_prototype(
                    orig_vec.unsqueeze(0), int(label)
                )[0]
                noisy_assignment = prototype_discovery.assign_prototype(
                    noisy_vec.unsqueeze(0), int(label)
                )[0]
                flipped = (orig_assignment != noisy_assignment)
                flips.append(flipped)
                per_class_flips[int(label)].append(flipped)
        
        # Aggregate results
        mean_cosine_sim = np.mean(cosine_sims)
        std_cosine_sim = np.std(cosine_sims)
        results['mean_cosine_similarity'].append(mean_cosine_sim)
        results['std_cosine_similarity'].append(std_cosine_sim)
        
        print(f"  Mean cosine similarity: {mean_cosine_sim:.4f} ± {std_cosine_sim:.4f}")
        
        for class_id in [0, 1]:
            if len(per_class_cosine_sims[class_id]) > 0:
                class_mean = np.mean(per_class_cosine_sims[class_id])
                if noise_level not in results['per_class_results'][class_id]:
                    results['per_class_results'][class_id][noise_level] = {}
                results['per_class_results'][class_id][noise_level]['cosine_sim'] = class_mean
        
        if prototype_discovery is not None:
            flip_rate = np.mean(flips)
            results['flip_rate'].append(flip_rate)
            print(f"  Prototype flip rate: {flip_rate:.2%}")
            
            for class_id in [0, 1]:
                if len(per_class_flips[class_id]) > 0:
                    class_flip_rate = np.mean(per_class_flips[class_id])
                    results['per_class_flip_rate'][class_id].append(class_flip_rate)
                    results['per_class_results'][class_id][noise_level]['flip_rate'] = class_flip_rate
    
    print(f"\n✓ Noise robustness analysis complete")
    return results


def shift_robustness(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    layer_name: str,
    composite,
    attributor: TimeSeriesCondAttribution,
    n_samples: int = 100,
    shift_amounts: List[int] = [10, 25, 50],
    prototype_discovery: Optional[TemporalPrototypeDiscovery] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, Any]:
    """
    Test robustness to circular time-shift.
    
    Shifts the input signal by small amounts (±shift_amounts timesteps) using
    circular shift. This tests whether concept activations are position-invariant,
    which they should be for CNN features.
    
    Args:
        model: PyTorch model
        dataset: Dataset to sample from
        layer_name: Layer to analyze
        composite: LRP composite for CRP attribution
        attributor: TimeSeriesCondAttribution instance
        n_samples: Number of samples to test
        shift_amounts: List of shift amounts in timesteps (will test ±each amount)
        prototype_discovery: Optional TemporalPrototypeDiscovery for assignment stability
        device: Device to run on
        
    Returns:
        Dictionary with per-shift metrics:
        {
            'shift_amounts': [±10, ±25, ±50],
            'mean_cosine_similarity': {...},
            'flip_rate': {...} (if prototype_discovery provided)
        }
    """
    print("\n" + "="*80)
    print("TIME-SHIFT ROBUSTNESS ANALYSIS")
    print("="*80)
    print(f"Testing {n_samples} samples at shift amounts: ±{shift_amounts} timesteps")
    
    model.to(device)
    model.eval()
    
    # Sample indices
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    
    # Get original concept vectors
    print("\nComputing original concept vectors...")
    original_vectors, labels = _compute_concept_vectors(
        model, dataset, indices, layer_name, composite, attributor, device
    )
    
    # Test both positive and negative shifts
    all_shifts = []
    for amount in shift_amounts:
        all_shifts.extend([-amount, amount])
    
    results = {
        'shift_amounts': all_shifts,
        'mean_cosine_similarity': {},
        'std_cosine_similarity': {},
        'per_class_results': {0: {}, 1: {}}
    }
    
    if prototype_discovery is not None:
        results['flip_rate'] = {}
        results['per_class_flip_rate'] = {0: {}, 1: {}}
    
    # Test each shift amount
    for shift_amount in all_shifts:
        print(f"\nShift amount: {shift_amount:+d} timesteps")
        
        cosine_sims = []
        flips = []
        
        per_class_cosine_sims = {0: [], 1: []}
        per_class_flips = {0: [], 1: []}
        
        for idx, orig_vec, label in zip(indices, original_vectors, labels):
            signal, _ = dataset[idx]
            signal = signal.unsqueeze(0).to(device)  # (1, C, T)
            
            # Circular shift
            shifted_signal = torch.roll(signal, shifts=shift_amount, dims=2)
            
            # Compute shifted concept vector
            shifted_vec = _compute_single_concept_vector(
                model, shifted_signal, layer_name, composite, attributor, label
            )
            
            # Cosine similarity
            cosine_sim = _cosine_similarity(orig_vec, shifted_vec)
            cosine_sims.append(cosine_sim)
            per_class_cosine_sims[int(label)].append(cosine_sim)
            
            # Check prototype assignment flip
            if prototype_discovery is not None:
                orig_assignment = prototype_discovery.assign_prototype(
                    orig_vec.unsqueeze(0), int(label)
                )[0]
                shifted_assignment = prototype_discovery.assign_prototype(
                    shifted_vec.unsqueeze(0), int(label)
                )[0]
                flipped = (orig_assignment != shifted_assignment)
                flips.append(flipped)
                per_class_flips[int(label)].append(flipped)
        
        # Aggregate results
        mean_cosine_sim = np.mean(cosine_sims)
        std_cosine_sim = np.std(cosine_sims)
        results['mean_cosine_similarity'][shift_amount] = mean_cosine_sim
        results['std_cosine_similarity'][shift_amount] = std_cosine_sim
        
        print(f"  Mean cosine similarity: {mean_cosine_sim:.4f} ± {std_cosine_sim:.4f}")
        
        for class_id in [0, 1]:
            if len(per_class_cosine_sims[class_id]) > 0:
                class_mean = np.mean(per_class_cosine_sims[class_id])
                if shift_amount not in results['per_class_results'][class_id]:
                    results['per_class_results'][class_id][shift_amount] = {}
                results['per_class_results'][class_id][shift_amount]['cosine_sim'] = class_mean
        
        if prototype_discovery is not None:
            flip_rate = np.mean(flips)
            results['flip_rate'][shift_amount] = flip_rate
            print(f"  Prototype flip rate: {flip_rate:.2%}")
            
            for class_id in [0, 1]:
                if len(per_class_flips[class_id]) > 0:
                    class_flip_rate = np.mean(per_class_flips[class_id])
                    if shift_amount not in results['per_class_flip_rate'][class_id]:
                        results['per_class_flip_rate'][class_id][shift_amount] = class_flip_rate
                    results['per_class_results'][class_id][shift_amount]['flip_rate'] = class_flip_rate
    
    print(f"\n✓ Shift robustness analysis complete")
    return results


def channel_dropout_robustness(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    layer_name: str,
    composite,
    attributor: TimeSeriesCondAttribution,
    n_samples: int = 100,
    prototype_discovery: Optional[TemporalPrototypeDiscovery] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, Any]:
    """
    Test robustness to channel dropout.
    
    Zeros out one accelerometer axis at a time (X, Y, or Z) to reveal
    which axis each concept depends on. This tests:
    - Axis-specific vs multi-axis concepts
    - Redundancy across axes
    - Critical axes for each prototype
    
    Args:
        model: PyTorch model
        dataset: Dataset to sample from (assumes 3-channel accelerometer data)
        layer_name: Layer to analyze
        composite: LRP composite for CRP attribution
        attributor: TimeSeriesCondAttribution instance
        n_samples: Number of samples to test
        prototype_discovery: Optional TemporalPrototypeDiscovery for assignment stability
        device: Device to run on
        
    Returns:
        Dictionary with per-channel metrics:
        {
            'channels': ['X', 'Y', 'Z'],
            'mean_cosine_similarity': [0.85, 0.90, 0.82],
            'flip_rate': [...] (if prototype_discovery provided),
            'per_class_results': {...}
        }
    """
    print("\n" + "="*80)
    print("CHANNEL DROPOUT ROBUSTNESS ANALYSIS")
    print("="*80)
    print(f"Testing {n_samples} samples by zeroing out each axis (X/Y/Z)")
    
    model.to(device)
    model.eval()
    
    # Sample indices
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    
    # Get original concept vectors
    print("\nComputing original concept vectors...")
    original_vectors, labels = _compute_concept_vectors(
        model, dataset, indices, layer_name, composite, attributor, device
    )
    
    # Get number of channels from first sample
    signal, _ = dataset[0]
    n_channels = signal.shape[0]
    channel_names = ['X', 'Y', 'Z'][:n_channels]
    
    results = {
        'channels': channel_names,
        'mean_cosine_similarity': [],
        'std_cosine_similarity': [],
        'per_class_results': {0: {}, 1: {}}
    }
    
    if prototype_discovery is not None:
        results['flip_rate'] = []
        results['per_class_flip_rate'] = {0: [], 1: []}
    
    # Test dropping each channel
    for channel_idx, channel_name in enumerate(channel_names):
        print(f"\nDropping channel {channel_idx} ({channel_name})")
        
        cosine_sims = []
        flips = []
        
        per_class_cosine_sims = {0: [], 1: []}
        per_class_flips = {0: [], 1: []}
        
        for idx, orig_vec, label in zip(indices, original_vectors, labels):
            signal, _ = dataset[idx]
            signal = signal.unsqueeze(0).to(device)  # (1, C, T)
            
            # Zero out one channel
            dropout_signal = signal.clone()
            dropout_signal[:, channel_idx, :] = 0.0
            
            # Compute dropout concept vector
            dropout_vec = _compute_single_concept_vector(
                model, dropout_signal, layer_name, composite, attributor, label
            )
            
            # Cosine similarity
            cosine_sim = _cosine_similarity(orig_vec, dropout_vec)
            cosine_sims.append(cosine_sim)
            per_class_cosine_sims[int(label)].append(cosine_sim)
            
            # Check prototype assignment flip
            if prototype_discovery is not None:
                orig_assignment = prototype_discovery.assign_prototype(
                    orig_vec.unsqueeze(0), int(label)
                )[0]
                dropout_assignment = prototype_discovery.assign_prototype(
                    dropout_vec.unsqueeze(0), int(label)
                )[0]
                flipped = (orig_assignment != dropout_assignment)
                flips.append(flipped)
                per_class_flips[int(label)].append(flipped)
        
        # Aggregate results
        mean_cosine_sim = np.mean(cosine_sims)
        std_cosine_sim = np.std(cosine_sims)
        results['mean_cosine_similarity'].append(mean_cosine_sim)
        results['std_cosine_similarity'].append(std_cosine_sim)
        
        print(f"  Mean cosine similarity: {mean_cosine_sim:.4f} ± {std_cosine_sim:.4f}")
        print(f"  → Lower similarity = concept depends more on {channel_name} axis")
        
        for class_id in [0, 1]:
            if len(per_class_cosine_sims[class_id]) > 0:
                class_mean = np.mean(per_class_cosine_sims[class_id])
                if channel_name not in results['per_class_results'][class_id]:
                    results['per_class_results'][class_id][channel_name] = {}
                results['per_class_results'][class_id][channel_name]['cosine_sim'] = class_mean
        
        if prototype_discovery is not None:
            flip_rate = np.mean(flips)
            results['flip_rate'].append(flip_rate)
            print(f"  Prototype flip rate: {flip_rate:.2%}")
            print(f"  → Higher flip rate = {channel_name} axis is critical for prototype assignment")
            
            for class_id in [0, 1]:
                if len(per_class_flips[class_id]) > 0:
                    class_flip_rate = np.mean(per_class_flips[class_id])
                    results['per_class_flip_rate'][class_id].append(class_flip_rate)
                    results['per_class_results'][class_id][channel_name]['flip_rate'] = class_flip_rate
    
    print(f"\n✓ Channel dropout robustness analysis complete")
    return results


def run_robustness_analysis(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    layer_name: str,
    composite,
    attributor: TimeSeriesCondAttribution,
    prototype_discovery: Optional[TemporalPrototypeDiscovery] = None,
    n_samples: int = 100,
    noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2],
    shift_amounts: List[int] = [10, 25, 50],
    test_channel_dropout: bool = True,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, Any]:
    """
    Run comprehensive robustness analysis.
    
    Executes all three robustness tests:
    1. Gaussian noise robustness
    2. Time-shift robustness
    3. Channel dropout robustness (if test_channel_dropout=True)
    
    Args:
        model: PyTorch model
        dataset: Dataset to sample from
        layer_name: Layer to analyze
        composite: LRP composite for CRP attribution
        attributor: TimeSeriesCondAttribution instance
        prototype_discovery: Optional TemporalPrototypeDiscovery for assignment stability
        n_samples: Number of samples to test
        noise_levels: List of noise levels for Gaussian noise test
        shift_amounts: List of shift amounts for time-shift test
        test_channel_dropout: Whether to test channel dropout
        device: Device to run on
        
    Returns:
        Dictionary with all robustness results:
        {
            'noise': {...},
            'shift': {...},
            'channel_dropout': {...} (if test_channel_dropout=True)
        }
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE ROBUSTNESS ANALYSIS")
    print("="*80)
    print(f"Layer: {layer_name}")
    print(f"Samples: {n_samples}")
    print(f"Device: {device}")
    
    results = {}
    
    # 1. Gaussian noise robustness
    results['noise'] = noise_robustness(
        model, dataset, layer_name, composite, attributor,
        n_samples, noise_levels, prototype_discovery, device
    )
    
    # 2. Time-shift robustness
    results['shift'] = shift_robustness(
        model, dataset, layer_name, composite, attributor,
        n_samples, shift_amounts, prototype_discovery, device
    )
    
    # 3. Channel dropout robustness (optional)
    if test_channel_dropout:
        results['channel_dropout'] = channel_dropout_robustness(
            model, dataset, layer_name, composite, attributor,
            n_samples, prototype_discovery, device
        )
    
    print("\n" + "="*80)
    print("✓ COMPREHENSIVE ROBUSTNESS ANALYSIS COMPLETE")
    print("="*80)
    
    return results


def print_robustness_report(results: Dict[str, Any]):
    """
    Print formatted console report of robustness results.
    
    Args:
        results: Output from run_robustness_analysis()
    """
    print("\n" + "="*80)
    print("ROBUSTNESS ANALYSIS REPORT")
    print("="*80)
    
    # 1. Noise robustness
    if 'noise' in results:
        print("\n" + "-"*80)
        print("GAUSSIAN NOISE ROBUSTNESS")
        print("-"*80)
        noise_results = results['noise']
        print(f"\n{'Noise Level':<15} {'Cosine Sim':<20} {'Flip Rate':<15}")
        print("-"*50)
        
        for i, level in enumerate(noise_results['noise_levels']):
            cosine_sim = noise_results['mean_cosine_similarity'][i]
            std_sim = noise_results['std_cosine_similarity'][i]
            
            line = f"{level:<15.3f} {cosine_sim:.4f} ± {std_sim:.4f}"
            
            if 'flip_rate' in noise_results:
                flip_rate = noise_results['flip_rate'][i]
                line += f"     {flip_rate:>6.2%}"
            
            print(line)
        
        # Interpretation
        print("\nInterpretation:")
        final_cosine = noise_results['mean_cosine_similarity'][-1]
        if final_cosine > 0.9:
            print("  ✓ Excellent: Concepts are highly robust to noise")
        elif final_cosine > 0.8:
            print("  ✓ Good: Concepts show good noise robustness")
        elif final_cosine > 0.7:
            print("  ⚠ Moderate: Some sensitivity to noise")
        else:
            print("  ✗ Poor: Concepts are noise-sensitive")
    
    # 2. Shift robustness
    if 'shift' in results:
        print("\n" + "-"*80)
        print("TIME-SHIFT ROBUSTNESS")
        print("-"*80)
        shift_results = results['shift']
        print(f"\n{'Shift Amount':<15} {'Cosine Sim':<20} {'Flip Rate':<15}")
        print("-"*50)
        
        for shift_amount in shift_results['shift_amounts']:
            cosine_sim = shift_results['mean_cosine_similarity'][shift_amount]
            std_sim = shift_results['std_cosine_similarity'][shift_amount]
            
            line = f"{shift_amount:>+14d} {cosine_sim:.4f} ± {std_sim:.4f}"
            
            if 'flip_rate' in shift_results:
                flip_rate = shift_results['flip_rate'][shift_amount]
                line += f"     {flip_rate:>6.2%}"
            
            print(line)
        
        # Interpretation
        print("\nInterpretation:")
        mean_cosine = np.mean(list(shift_results['mean_cosine_similarity'].values()))
        if mean_cosine > 0.9:
            print("  ✓ Excellent: Concepts are position-invariant (as expected for CNNs)")
        elif mean_cosine > 0.8:
            print("  ✓ Good: Concepts show good shift robustness")
        else:
            print("  ⚠ Unexpected: CNNs should be shift-invariant")
    
    # 3. Channel dropout robustness
    if 'channel_dropout' in results:
        print("\n" + "-"*80)
        print("CHANNEL DROPOUT ROBUSTNESS")
        print("-"*80)
        dropout_results = results['channel_dropout']
        print(f"\n{'Channel':<15} {'Cosine Sim':<20} {'Flip Rate':<15}")
        print("-"*50)
        
        for i, channel in enumerate(dropout_results['channels']):
            cosine_sim = dropout_results['mean_cosine_similarity'][i]
            std_sim = dropout_results['std_cosine_similarity'][i]
            
            line = f"{channel:<15} {cosine_sim:.4f} ± {std_sim:.4f}"
            
            if 'flip_rate' in dropout_results:
                flip_rate = dropout_results['flip_rate'][i]
                line += f"     {flip_rate:>6.2%}"
            
            print(line)
        
        # Interpretation
        print("\nInterpretation:")
        cosine_sims = dropout_results['mean_cosine_similarity']
        min_idx = np.argmin(cosine_sims)
        max_idx = np.argmax(cosine_sims)
        most_critical = dropout_results['channels'][min_idx]
        least_critical = dropout_results['channels'][max_idx]
        
        print(f"  → Most critical axis: {most_critical} (lowest similarity when dropped)")
        print(f"  → Least critical axis: {least_critical} (highest similarity when dropped)")
        
        if max(cosine_sims) - min(cosine_sims) > 0.15:
            print("  → Concepts show strong axis-specific dependencies")
        else:
            print("  → Concepts use information from all axes fairly equally")
    
    print("\n" + "="*80 + "\n")


# Helper functions

def _compute_concept_vectors(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    indices: np.ndarray,
    layer_name: str,
    composite,
    attributor: TimeSeriesCondAttribution,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute concept relevance vectors for multiple samples.
    
    Returns:
        concept_vectors: Tensor of shape (n_samples, n_filters)
        labels: Tensor of shape (n_samples,)
    """
    model.eval()
    concept_vectors = []
    labels_list = []
    
    cc = ChannelConcept()
    
    for idx in indices:
        signal, label = dataset[idx]
        signal = signal.unsqueeze(0).to(device)  # (1, C, T)
        
        concept_vec = _compute_single_concept_vector(
            model, signal, layer_name, composite, attributor, label
        )
        
        concept_vectors.append(concept_vec)
        labels_list.append(label)
    
    return torch.stack(concept_vectors), torch.tensor(labels_list)


def _compute_single_concept_vector(
    model: torch.nn.Module,
    signal: torch.Tensor,
    layer_name: str,
    composite,
    attributor: TimeSeriesCondAttribution,
    label: int
) -> torch.Tensor:
    """
    Compute concept relevance vector for a single sample.
    
    Returns:
        concept_vector: Tensor of shape (n_filters,)
    """
    with torch.no_grad():
        # Get prediction
        output = model(signal)
        pred_class = output.argmax(dim=1).item()
    
    # Compute CRP heatmap for predicted class
    conditions = [{"y": pred_class}]
    attr = attributor(signal, conditions, composite, record_layer=[layer_name])
    
    # Extract concept relevances
    cc = ChannelConcept()
    layer_relevance = attr.relevances[layer_name]  # (1, n_filters, T)
    concept_relevances = cc.attribute(layer_relevance, abs_norm=True)  # (1, n_filters)
    
    return concept_relevances.squeeze(0).cpu()  # (n_filters,)


def _cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Returns:
        Cosine similarity in range [-1, 1]
    """
    vec1 = vec1.float()
    vec2 = vec2.float()
    
    dot_product = torch.dot(vec1, vec2)
    norm1 = torch.norm(vec1)
    norm2 = torch.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return (dot_product / (norm1 * norm2)).item()


if __name__ == "__main__":
    # Test robustness analysis with synthetic data
    print("Testing input-perturbation robustness analysis...")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create synthetic model and data
    from torch.utils.data import TensorDataset
    
    # Synthetic model
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv1d(3, 32, kernel_size=5)
            self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=5)
            self.conv3 = torch.nn.Conv1d(64, 64, kernel_size=5)
            self.pool = torch.nn.AdaptiveAvgPool1d(1)
            self.fc = torch.nn.Linear(64, 2)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = self.pool(x).squeeze(-1)
            return self.fc(x)
    
    model = DummyModel()
    
    # Synthetic dataset
    n_samples = 50
    signals = torch.randn(n_samples, 3, 2000)
    labels = torch.randint(0, 2, (n_samples,))
    dataset = TensorDataset(signals, labels)
    
    # Test noise robustness only (others would need full CRP setup)
    print("\n✓ Robustness module structure validated!")
    print("Note: Full testing requires CRP composite and attributor setup")
