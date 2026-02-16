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
from typing import List, Optional, Tuple, Union


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
