"""
Visualization functions for 1D time-series heatmaps and concept analysis.

ALL visualization is 1D - no image functions.
Replaces crp.image.imgify and vis_opaque_img with signal plotting.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import List, Optional, Tuple, Union, Dict


def plot_ts_heatmap(
    signal: np.ndarray,
    heatmap: np.ndarray,
    axes_names: List[str] = ['X', 'Y', 'Z'],
    title: str = '',
    figsize: Tuple[int, int] = (15, 6),
    alpha: float = 0.3,
    cmap: str = 'bwr'
) -> plt.Figure:
    """
    Plot time-series signal with heatmap overlay.
    
    Visualizes multi-channel 1D signal with relevance heatmap
    using color overlay (blue=negative, red=positive relevance).
    
    Args:
        signal: Signal array of shape (C, T) or (T,)
        heatmap: Heatmap array of shape (C, T) or (T,)
        axes_names: Names for each channel
        title: Plot title
        figsize: Figure size
        alpha: Transparency of heatmap overlay (0-1)
        cmap: Colormap for heatmap (default: 'bwr' - blue-white-red)
        
    Returns:
        Matplotlib figure
    """
    # Handle 1D or 2D inputs
    if signal.ndim == 1:
        signal = signal[None, :]
        heatmap = heatmap[None, :]
    
    n_channels, n_timesteps = signal.shape
    
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]
    
    # Normalize heatmap for colormap
    heatmap_flat = heatmap.flatten()
    vmax = np.abs(heatmap_flat).max()
    norm = Normalize(vmin=-vmax, vmax=vmax)
    
    for i, ax in enumerate(axes):
        # Plot signal
        time = np.arange(n_timesteps)
        ax.plot(time, signal[i], 'k-', linewidth=1, alpha=0.7, label='Signal')
        
        # Overlay heatmap as colored background
        heat_colors = plt.cm.get_cmap(cmap)(norm(heatmap[i]))
        for t in range(n_timesteps - 1):
            ax.axvspan(t, t + 1, 
                      facecolor=heat_colors[t][:3], 
                      alpha=alpha)
        
        ax.set_ylabel(f'{axes_names[i]}' if i < len(axes_names) else f'Ch{i}')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (samples)')
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    return fig


def plot_concept_relevance(
    signal: np.ndarray,
    relevance_per_concept: np.ndarray,
    concept_labels: List[str],
    title: str = '',
    figsize: Tuple[int, int] = (15, 8)
) -> plt.Figure:
    """
    Plot signal with per-concept relevance traces.
    
    Shows original signal and relevance attributed to each concept.
    Useful for filterbank or learned cluster visualization.
    
    Args:
        signal: Original signal of shape (C, T) or (T,)
        relevance_per_concept: Relevance per concept of shape (n_concepts, T)
        concept_labels: Labels for each concept
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if signal.ndim == 2:
        # Use mean across channels for simplicity
        signal = signal.mean(axis=0)
    
    n_concepts = relevance_per_concept.shape[0]
    
    fig, axes = plt.subplots(n_concepts + 1, 1, figsize=figsize, sharex=True)
    
    # Plot original signal
    time = np.arange(len(signal))
    axes[0].plot(time, signal, 'k-', linewidth=1)
    axes[0].set_ylabel('Signal')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Original Signal')
    
    # Plot per-concept relevance
    colors = plt.cm.Set2(np.linspace(0, 1, n_concepts))
    for i in range(n_concepts):
        axes[i + 1].fill_between(
            time, 
            0, 
            relevance_per_concept[i], 
            color=colors[i],
            alpha=0.6,
            label=concept_labels[i]
        )
        axes[i + 1].set_ylabel(concept_labels[i])
        axes[i + 1].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (samples)')
    if title:
        fig.suptitle(title, y=0.995)
    
    plt.tight_layout()
    return fig


def plot_prototype_grid(
    prototype_signals: List[np.ndarray],
    coverage_pct: List[float],
    cosine_sims: List[float],
    titles: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot grid of prototype signals with metadata.
    
    From pcx_codes plot_prototypes.py - adapted for 1D signals.
    Shows representative samples for each discovered prototype.
    
    Args:
        prototype_signals: List of signal arrays, each of shape (C, T)
        coverage_pct: Percentage of samples assigned to each prototype
        cosine_sims: Mean cosine similarity to prototype center
        titles: Optional titles for each prototype
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_prototypes = len(prototype_signals)
    n_cols = min(3, n_prototypes)
    n_rows = (n_prototypes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes[None, :]
    axes = axes.flatten()
    
    for i, (signal, cov, sim) in enumerate(zip(prototype_signals, coverage_pct, cosine_sims)):
        ax = axes[i]
        
        # Plot each channel
        if signal.ndim == 2:
            time = np.arange(signal.shape[1])
            for c in range(signal.shape[0]):
                ax.plot(time, signal[c], alpha=0.7, label=f'Ch{c}')
        else:
            time = np.arange(len(signal))
            ax.plot(time, signal, 'k-')
        
        title = titles[i] if titles and i < len(titles) else f'Prototype {i}'
        ax.set_title(f'{title}\nCoverage: {cov:.1f}% | Sim: {sim:.3f}')
        ax.grid(True, alpha=0.3)
        if signal.ndim == 2:
            ax.legend(loc='upper right', fontsize=8)
    
    # Hide unused axes
    for i in range(n_prototypes, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_deviation_matrix(
    deviations: np.ndarray,
    concept_labels: List[str],
    sample_labels: Optional[List[str]] = None,
    title: str = 'Concept Deviations from Prototype',
    figsize: Tuple[int, int] = (10, 6),
    cmap: str = 'RdBu_r'
) -> plt.Figure:
    """
    Plot heatmap of concept deviations Δ = ν - μ_i.
    
    From PCX paper: visualize how samples deviate from prototype centers
    in concept space. Positive (red) = higher than prototype,
    negative (blue) = lower than prototype.
    
    Args:
        deviations: Deviation matrix of shape (n_samples, n_concepts)
        concept_labels: Labels for each concept dimension
        sample_labels: Optional labels for samples
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    vmax = np.abs(deviations).max()
    im = ax.imshow(
        deviations.T, 
        aspect='auto', 
        cmap=cmap,
        vmin=-vmax, 
        vmax=vmax
    )
    
    ax.set_yticks(range(len(concept_labels)))
    ax.set_yticklabels(concept_labels)
    ax.set_ylabel('Concept')
    
    if sample_labels:
        ax.set_xticks(range(len(sample_labels)))
        ax.set_xticklabels(sample_labels, rotation=45, ha='right')
    ax.set_xlabel('Sample')
    
    ax.set_title(title)
    
    plt.colorbar(im, ax=ax, label='Deviation (ν - μ)')
    plt.tight_layout()
    return fig


def plot_intervention_comparison(
    original_signal: np.ndarray,
    original_heatmap: np.ndarray,
    intervened_heatmap: np.ndarray,
    concept_idx: int,
    axes_names: List[str] = ['X', 'Y', 'Z'],
    figsize: Tuple[int, int] = (15, 8)
) -> plt.Figure:
    """
    Compare original vs intervened heatmaps side-by-side.
    
    Visualizes the effect of concept suppression/amplification.
    
    Args:
        original_signal: Input signal of shape (C, T)
        original_heatmap: Original heatmap of shape (C, T)
        intervened_heatmap: Heatmap after intervention of shape (C, T)
        concept_idx: Index of intervened concept
        axes_names: Names for channels
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Original heatmap
    ax1 = plt.subplot(2, 1, 1)
    plt.title(f'Original Heatmap')
    _plot_heatmap_subplot(original_signal, original_heatmap, axes_names, ax1)
    
    # Intervened heatmap
    ax2 = plt.subplot(2, 1, 2)
    plt.title(f'After Suppressing Concept {concept_idx}')
    _plot_heatmap_subplot(original_signal, intervened_heatmap, axes_names, ax2)
    
    plt.tight_layout()
    return fig


def _plot_heatmap_subplot(
    signal: np.ndarray, 
    heatmap: np.ndarray, 
    axes_names: List[str],
    ax: plt.Axes
):
    """Helper to plot heatmap in subplot."""
    n_channels = signal.shape[0]
    time = np.arange(signal.shape[1])
    
    # Just show first channel for simplicity in comparison
    ax.plot(time, signal[0], 'k-', linewidth=1, alpha=0.7)
    
    # Color by heatmap
    vmax = np.abs(heatmap).max()
    norm = Normalize(vmin=-vmax, vmax=vmax)
    heat_colors = plt.cm.bwr(norm(heatmap[0]))
    
    for t in range(len(time) - 1):
        ax.axvspan(t, t + 1, facecolor=heat_colors[t][:3], alpha=0.3)
    
    ax.set_ylabel(axes_names[0])
    ax.grid(True, alpha=0.3)


def plot_prototype_samples(
    signals: List[np.ndarray],
    heatmaps: List[np.ndarray],
    prototype_idx: int,
    sample_distances: Optional[List[float]] = None,
    axes_names: List[str] = ['X', 'Y', 'Z'],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
    alpha: float = 0.3,
    cmap: str = 'bwr'
) -> plt.Figure:
    """
    Plot representative samples for a single prototype.
    
    For each GMM prototype, displays the N closest real samples (by Euclidean
    distance to centroid μ in CRP filter space) along with their LRP heatmaps.
    
    This answers: "What does prototype X actually look like as a vibration signal?"
    
    Args:
        signals: List of signal arrays, each of shape (C, T)
        heatmaps: List of heatmap arrays, each of shape (C, T)
        prototype_idx: Prototype index for labeling
        sample_distances: Optional distances to prototype center
        axes_names: Names for each channel
        title: Optional plot title
        figsize: Figure size
        alpha: Transparency of heatmap overlay
        cmap: Colormap for heatmap
        
    Returns:
        Matplotlib figure
    """
    n_samples = len(signals)
    n_channels = signals[0].shape[0]
    
    fig, axes = plt.subplots(n_samples, n_channels, figsize=figsize, sharex=True)
    if n_samples == 1:
        axes = axes[None, :]
    if n_channels == 1:
        axes = axes[:, None]
    
    # Overall title
    if title is None:
        title = f'Prototype {prototype_idx}: Representative Samples'
    fig.suptitle(title, fontsize=14, y=0.995)
    
    for i, (signal, heatmap) in enumerate(zip(signals, heatmaps)):
        for c in range(n_channels):
            ax = axes[i, c]
            
            # Plot signal
            time = np.arange(signal.shape[1])
            ax.plot(time, signal[c], 'k-', linewidth=1, alpha=0.7)
            
            # Overlay heatmap
            heatmap_flat = heatmap.flatten()
            vmax = np.abs(heatmap_flat).max()
            norm = Normalize(vmin=-vmax, vmax=vmax)
            heat_colors = plt.cm.get_cmap(cmap)(norm(heatmap[c]))
            
            for t in range(len(time) - 1):
                ax.axvspan(t, t + 1, facecolor=heat_colors[t][:3], alpha=alpha)
            
            # Labels
            if i == 0:
                ax.set_title(axes_names[c] if c < len(axes_names) else f'Ch{c}')
            if c == 0:
                dist_str = f' (d={sample_distances[i]:.3f})' if sample_distances else ''
                ax.set_ylabel(f'Sample {i+1}{dist_str}', fontsize=10)
            if i == n_samples - 1:
                ax.set_xlabel('Time (samples)')
            
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_prototype_gallery(
    all_signals: Dict[int, List[np.ndarray]],
    all_heatmaps: Dict[int, List[np.ndarray]],
    all_distances: Dict[int, List[float]],
    class_id: int,
    axes_names: List[str] = ['X', 'Y', 'Z'],
    max_samples_per_proto: int = 3,
    figsize: Tuple[int, int] = (20, 15),
    alpha: float = 0.3,
    cmap: str = 'bwr'
) -> plt.Figure:
    """
    Create a gallery showing all prototypes for a class.
    
    Rows = prototypes, columns = representative samples.
    
    Args:
        all_signals: Dict mapping prototype_idx -> list of signal arrays
        all_heatmaps: Dict mapping prototype_idx -> list of heatmap arrays
        all_distances: Dict mapping prototype_idx -> list of distances
        class_id: Class ID for labeling
        axes_names: Names for each channel
        max_samples_per_proto: Maximum samples to show per prototype
        figsize: Figure size
        alpha: Transparency of heatmap overlay
        cmap: Colormap for heatmap
        
    Returns:
        Matplotlib figure
    """
    n_prototypes = len(all_signals)
    n_channels = all_signals[0][0].shape[0] if n_prototypes > 0 else 3
    
    # Each prototype gets a row with multiple sample columns
    fig = plt.figure(figsize=figsize)
    fig.suptitle(f'Class {class_id} Prototype Gallery', fontsize=16, y=0.995)
    
    gs = fig.add_gridspec(n_prototypes, max_samples_per_proto * n_channels, 
                          hspace=0.3, wspace=0.1)
    
    for proto_idx in range(n_prototypes):
        signals = all_signals[proto_idx][:max_samples_per_proto]
        heatmaps = all_heatmaps[proto_idx][:max_samples_per_proto]
        distances = all_distances[proto_idx][:max_samples_per_proto]
        
        for samp_idx, (signal, heatmap, dist) in enumerate(zip(signals, heatmaps, distances)):
            for c in range(n_channels):
                col_idx = samp_idx * n_channels + c
                ax = fig.add_subplot(gs[proto_idx, col_idx])
                
                # Plot signal with heatmap overlay
                time = np.arange(signal.shape[1])
                ax.plot(time, signal[c], 'k-', linewidth=0.8, alpha=0.7)
                
                # Heatmap overlay
                vmax = np.abs(heatmap).max()
                norm = Normalize(vmin=-vmax, vmax=vmax)
                heat_colors = plt.cm.get_cmap(cmap)(norm(heatmap[c]))
                
                for t in range(len(time) - 1):
                    ax.axvspan(t, t + 1, facecolor=heat_colors[t][:3], alpha=alpha)
                
                # Labels
                if proto_idx == 0 and samp_idx == 0:
                    ax.set_title(axes_names[c] if c < len(axes_names) else f'Ch{c}', 
                                fontsize=10)
                if c == 0:
                    ax.set_ylabel(f'P{proto_idx}S{samp_idx+1}\nd={dist:.2f}', 
                                 fontsize=9, rotation=0, labelpad=30)
                
                ax.tick_params(labelbottom=False, labelleft=False)
                ax.grid(True, alpha=0.2)
    
    return fig


def plot_prototype_comparison(
    ok_prototypes: List[np.ndarray],
    nok_prototypes: List[np.ndarray],
    filter_names: Optional[List[str]] = None,
    top_k: int = 10,
    figsize: Tuple[int, int] = (15, 8)
) -> plt.Figure:
    """
    Side-by-side comparison of OK vs NOK prototype filter patterns.
    
    Shows which filters differ most between classes to identify
    discriminative patterns.
    
    Args:
        ok_prototypes: List of OK prototype centers (each of shape (n_filters,))
        nok_prototypes: List of NOK prototype centers (each of shape (n_filters,))
        filter_names: Optional names for filters
        top_k: Number of top differing filters to highlight
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_ok = len(ok_prototypes)
    n_nok = len(nok_prototypes)
    n_filters = ok_prototypes[0].shape[0]
    
    # Average prototype per class
    ok_mean = np.mean(ok_prototypes, axis=0)
    nok_mean = np.mean(nok_prototypes, axis=0)
    
    # Compute difference
    diff = np.abs(nok_mean - ok_mean)
    top_diff_indices = np.argsort(diff)[-top_k:][::-1]
    
    fig = plt.figure(figsize=figsize)
    
    # Plot 1: Heatmap of all prototypes
    ax1 = plt.subplot(2, 2, 1)
    all_prototypes = np.vstack([ok_prototypes, nok_prototypes])
    im1 = ax1.imshow(all_prototypes, aspect='auto', cmap='RdBu_r', 
                     vmin=-np.abs(all_prototypes).max(), 
                     vmax=np.abs(all_prototypes).max())
    ax1.axhline(y=n_ok - 0.5, color='k', linestyle='--', linewidth=2)
    ax1.set_ylabel('Prototype')
    ax1.set_xlabel('Filter Index')
    ax1.set_title('All Prototype Patterns')
    ax1.set_yticks(range(n_ok + n_nok))
    ax1.set_yticklabels([f'OK-{i}' for i in range(n_ok)] + 
                        [f'NOK-{i}' for i in range(n_nok)])
    plt.colorbar(im1, ax=ax1, label='Filter Value')
    
    # Plot 2: Mean patterns per class
    ax2 = plt.subplot(2, 2, 2)
    filter_indices = np.arange(n_filters)
    ax2.bar(filter_indices, ok_mean, alpha=0.6, label='OK Mean', color='blue')
    ax2.bar(filter_indices, nok_mean, alpha=0.6, label='NOK Mean', color='red')
    # Highlight top-k different filters
    for idx in top_diff_indices:
        ax2.axvline(x=idx, color='yellow', alpha=0.3, linewidth=3)
    ax2.set_xlabel('Filter Index')
    ax2.set_ylabel('Mean Filter Value')
    ax2.set_title('Mean Prototype per Class')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Difference magnitude
    ax3 = plt.subplot(2, 2, 3)
    colors = ['yellow' if i in top_diff_indices else 'gray' for i in range(n_filters)]
    ax3.bar(filter_indices, diff, color=colors, alpha=0.7)
    ax3.set_xlabel('Filter Index')
    ax3.set_ylabel('|NOK - OK|')
    ax3.set_title(f'Filter Differences (Top-{top_k} highlighted)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Top-k filters detail
    ax4 = plt.subplot(2, 2, 4)
    x_pos = np.arange(top_k)
    width = 0.35
    ax4.bar(x_pos - width/2, ok_mean[top_diff_indices], width, 
            label='OK', color='blue', alpha=0.6)
    ax4.bar(x_pos + width/2, nok_mean[top_diff_indices], width, 
            label='NOK', color='red', alpha=0.6)
    ax4.set_xlabel('Filter')
    ax4.set_ylabel('Filter Value')
    ax4.set_title(f'Top-{top_k} Most Different Filters')
    ax4.set_xticks(x_pos)
    if filter_names:
        ax4.set_xticklabels([filter_names[i] if i < len(filter_names) 
                            else f'F{i}' for i in top_diff_indices], rotation=45)
    else:
        ax4.set_xticklabels([f'F{i}' for i in top_diff_indices], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Test visualization functions
    np.random.seed(42)
    
    # Synthetic signal
    t = np.linspace(0, 5, 2000)
    signal = np.array([
        np.sin(2 * np.pi * 10 * t),
        np.sin(2 * np.pi * 20 * t),
        np.sin(2 * np.pi * 30 * t)
    ])
    
    # Synthetic heatmap
    heatmap = np.random.randn(3, 2000) * 0.5
    
    # Test plot_ts_heatmap
    fig = plot_ts_heatmap(signal, heatmap, title='Test Heatmap')
    plt.savefig('/tmp/test_heatmap.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Created test_heatmap.png")
    
    # Test plot_concept_relevance
    relevance_per_concept = np.abs(np.random.randn(4, 2000))
    concept_labels = ['Concept A', 'Concept B', 'Concept C', 'Concept D']
    fig = plot_concept_relevance(signal, relevance_per_concept, concept_labels)
    plt.savefig('/tmp/test_concepts.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Created test_concepts.png")
    
    print("✓ All visualization tests passed!")
