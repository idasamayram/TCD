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
from typing import List, Optional, Tuple, Union, Dict, Any


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


def plot_attribution_graph(
    prototype_mean: np.ndarray,
    class_id: int,
    top_k: int = 10,
    filter_labels: Optional[List[str]] = None,
    class_names: Dict[int, str] = {0: "OK", 1: "NOK"},
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Visualize concept-to-output attribution flow as a graph.
    
    Shows the top-k filters (concepts) from a prototype center μ as nodes on the left,
    connected to the output class node on the right. Edge width is proportional to
    filter importance |μ_k|, and edge color indicates sign (positive=green, negative=red).
    
    This visualizes which concepts contribute to a specific prototype/class prediction.
    
    Args:
        prototype_mean: Prototype center μ of shape (n_filters,)
        class_id: Output class ID
        top_k: Number of top filters to show
        filter_labels: Optional custom labels for filters
        class_names: Class ID to name mapping
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_filters = len(prototype_mean)
    
    # Get top-k filters by absolute magnitude
    top_indices = np.argsort(np.abs(prototype_mean))[-top_k:][::-1]
    top_values = prototype_mean[top_indices]
    
    # Normalize edge widths to [1, 10]
    abs_values = np.abs(top_values)
    max_abs = abs_values.max()
    edge_widths = 1 + 9 * (abs_values / max_abs)
    
    # Layout: filters on left, class on right
    filter_x = 0.2
    class_x = 0.8
    
    filter_y_positions = np.linspace(0.1, 0.9, top_k)
    class_y = 0.5
    
    # Draw edges first (so they appear behind nodes)
    for i, (filter_idx, value, width) in enumerate(zip(top_indices, top_values, edge_widths)):
        filter_y = filter_y_positions[i]
        
        # Edge color: green for positive, red for negative
        edge_color = 'green' if value > 0 else 'red'
        edge_alpha = 0.6
        
        # Draw edge
        ax.plot([filter_x, class_x], [filter_y, class_y],
               color=edge_color, linewidth=width, alpha=edge_alpha, zorder=1)
        
        # Add edge label showing weight
        mid_x = (filter_x + class_x) / 2
        mid_y = (filter_y + class_y) / 2
        ax.text(mid_x, mid_y, f'{value:+.3f}',
               fontsize=8, ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Draw filter nodes
    for i, filter_idx in enumerate(top_indices):
        filter_y = filter_y_positions[i]
        value = top_values[i]
        
        # Node circle
        circle = plt.Circle((filter_x, filter_y), 0.03, 
                           color='steelblue', zorder=2)
        ax.add_patch(circle)
        
        # Node label
        if filter_labels and filter_idx < len(filter_labels):
            label = filter_labels[filter_idx]
        else:
            label = f'Filter {filter_idx}'
        
        ax.text(filter_x - 0.05, filter_y, 
               f'{label}\n|μ|={np.abs(value):.3f}',
               fontsize=9, ha='right', va='center')
    
    # Draw class node
    circle = plt.Circle((class_x, class_y), 0.05, 
                       color='orange', zorder=2)
    ax.add_patch(circle)
    
    class_name = class_names.get(class_id, f'Class {class_id}')
    ax.text(class_x + 0.05, class_y, class_name,
           fontsize=12, ha='left', va='center', fontweight='bold')
    
    # Set axis limits and remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Title
    ax.set_title(f'Prototype → {class_name} Attribution Graph\n'
                f'(Top-{top_k} Concepts)', fontsize=14, fontweight='bold')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=3, label='Positive contribution'),
        Line2D([0], [0], color='red', linewidth=3, label='Negative contribution')
    ]
    ax.legend(handles=legend_elements, loc='lower center', 
             bbox_to_anchor=(0.5, -0.05), ncol=2)
    
    plt.tight_layout()
    return fig


def plot_robustness_summary(
    results: Dict[str, Any],
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Create bar chart summary of robustness scores across perturbation types.
    
    Visualizes cosine similarity (robustness) for:
    - Gaussian noise at different levels
    - Time shifts at different amounts
    - Channel dropout per axis
    
    Args:
        results: Output from run_robustness_analysis()
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Determine number of subplots needed
    n_plots = sum([
        'noise' in results,
        'shift' in results,
        'channel_dropout' in results
    ])
    
    if n_plots == 0:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No robustness results to display',
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    plot_idx = 1
    
    # 1. Noise robustness
    if 'noise' in results:
        ax = fig.add_subplot(1, n_plots, plot_idx)
        plot_idx += 1
        
        noise_results = results['noise']
        noise_levels = noise_results['noise_levels']
        cosine_sims = noise_results['mean_cosine_similarity']
        std_sims = noise_results['std_cosine_similarity']
        
        x_pos = np.arange(len(noise_levels))
        bars = ax.bar(x_pos, cosine_sims, yerr=std_sims, 
                     alpha=0.7, capsize=5, color='steelblue')
        
        # Color bars by quality: green (>0.9), yellow (0.8-0.9), red (<0.8)
        for bar, sim in zip(bars, cosine_sims):
            if sim > 0.9:
                bar.set_color('green')
            elif sim > 0.8:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax.set_xlabel('Noise Level (fraction of signal std)')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Noise Robustness')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{level:.2f}' for level in noise_levels])
        ax.set_ylim([0, 1.05])
        ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent (>0.9)')
        ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Good (>0.8)')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=8)
    
    # 2. Shift robustness
    if 'shift' in results:
        ax = fig.add_subplot(1, n_plots, plot_idx)
        plot_idx += 1
        
        shift_results = results['shift']
        shift_amounts = shift_results['shift_amounts']
        cosine_sims = [shift_results['mean_cosine_similarity'][s] for s in shift_amounts]
        std_sims = [shift_results['std_cosine_similarity'][s] for s in shift_amounts]
        
        x_pos = np.arange(len(shift_amounts))
        bars = ax.bar(x_pos, cosine_sims, yerr=std_sims, 
                     alpha=0.7, capsize=5, color='steelblue')
        
        # Color bars by quality
        for bar, sim in zip(bars, cosine_sims):
            if sim > 0.9:
                bar.set_color('green')
            elif sim > 0.8:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax.set_xlabel('Shift Amount (timesteps)')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Time-Shift Robustness')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{s:+d}' for s in shift_amounts], rotation=45)
        ax.set_ylim([0, 1.05])
        ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Channel dropout robustness
    if 'channel_dropout' in results:
        ax = fig.add_subplot(1, n_plots, plot_idx)
        
        dropout_results = results['channel_dropout']
        channels = dropout_results['channels']
        cosine_sims = dropout_results['mean_cosine_similarity']
        std_sims = dropout_results['std_cosine_similarity']
        
        x_pos = np.arange(len(channels))
        bars = ax.bar(x_pos, cosine_sims, yerr=std_sims, 
                     alpha=0.7, capsize=5, color='steelblue')
        
        # Color bars: lower similarity = more critical channel (red)
        # Higher similarity = less critical channel (green)
        min_sim = min(cosine_sims)
        max_sim = max(cosine_sims)
        sim_range = max_sim - min_sim
        
        for bar, sim in zip(bars, cosine_sims):
            # Normalize to [0, 1] where 0=most critical, 1=least critical
            normalized = (sim - min_sim) / (sim_range + 1e-6)
            # Red for critical, green for non-critical
            color = plt.cm.RdYlGn(normalized)
            bar.set_color(color)
        
        ax.set_xlabel('Dropped Channel')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Channel Dropout Robustness\n(Lower = More Critical)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(channels)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Robustness Analysis Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def generate_concept_heatmaps(
    model,
    dataset,
    prototype_discovery,
    layer_name: str,
    composite,
    output_dir: str,
    top_k: int = 5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    features: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    axes_names: List[str] = ['X', 'Y', 'Z']
) -> Dict[Tuple[int, int], str]:
    """
    Generate concept conditional heatmaps (attribution graph) for each prototype.

    PCX's core visualization adapted for 1D signals. For each prototype:
    1. Find the closest real sample to the prototype center μ (via Euclidean
       distance in the pre-computed CRP feature space)
    2. Identify the top-k most important filters (by |μ_k|)
    3. For each top-k filter, compute conditional CRP heatmap using
       TimeSeriesCondAttribution
    4. Plot raw signal with concept's conditional heatmap overlaid (bwr cmap,
       per-axis subplots)

    Args:
        model: PyTorch model
        dataset: Dataset to load samples from
        prototype_discovery: TemporalPrototypeDiscovery instance with fitted GMMs
        layer_name: Layer to analyze
        composite: LRP composite for CRP attribution
        output_dir: Directory to save figures
        top_k: Number of top concepts to visualize per prototype
        device: Device to run on
        features: Pre-computed CRP feature tensor (N, n_filters); used to find
            the closest sample to each prototype centroid. When None the
            function falls back to the first class sample.
        labels: Class labels tensor (N,) aligned with ``features``.
        axes_names: Names of the signal channels (default: X, Y, Z)

    Returns:
        Dictionary mapping (class_id, proto_idx) -> figure path
    """
    import os
    from tcd.attribution import TimeSeriesCondAttribution

    print("\n" + "="*60)
    print("GENERATING CONCEPT CONDITIONAL HEATMAPS")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    model.to(device)
    model.eval()

    # Build attributor
    attributor = TimeSeriesCondAttribution(model)

    figures = {}

    for class_id in [0, 1]:
        if class_id not in prototype_discovery.gmms:
            continue

        class_name = "OK" if class_id == 0 else "NOK"
        print(f"\nClass {class_id} ({class_name}):")

        gmm = prototype_discovery.gmms[class_id]

        # Build index mapping: dataset_idx -> feature_row for this class
        if features is not None and labels is not None:
            features_np = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
            labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
            class_mask = labels_np == class_id
            class_feat = features_np[class_mask]
            class_dataset_indices = np.where(class_mask)[0]
        else:
            # Fallback: collect indices from dataset directly
            class_dataset_indices = np.array([
                i for i, (_, lbl) in enumerate(dataset) if int(lbl) == class_id
            ])
            class_feat = None

        for proto_idx in range(gmm.n_components):
            # Get prototype center μ
            prototype_mean = gmm.means_[proto_idx]

            # Find top-k filters by absolute magnitude
            top_filter_indices = np.argsort(np.abs(prototype_mean))[-top_k:][::-1]

            print(f"  Prototype {proto_idx}: Top-{top_k} filters = {top_filter_indices.tolist()}")

            # --- Find closest sample to prototype centroid ---
            if class_feat is not None and len(class_feat) > 0:
                dists = np.linalg.norm(class_feat - prototype_mean[None], axis=1)
                closest_row = int(np.argmin(dists))
                closest_dataset_idx = int(class_dataset_indices[closest_row])
            elif len(class_dataset_indices) > 0:
                closest_dataset_idx = int(class_dataset_indices[0])
            else:
                print(f"    No samples for class {class_id}, skipping")
                continue

            # Load the actual signal
            data_p, target = dataset[closest_dataset_idx]
            data_p = data_p.unsqueeze(0).to(device).requires_grad_(True)
            target_int = int(target)

            # --- Run conditional CRP per filter ---
            n_channels = data_p.shape[1]
            used_axes = axes_names[:n_channels]

            fig, axes = plt.subplots(
                top_k, n_channels,
                figsize=(5 * n_channels, 3 * top_k),
                squeeze=False
            )
            fig.suptitle(
                f'Class {class_id} ({class_name}) Prototype {proto_idx} — Conditional CRP Heatmaps',
                fontsize=13
            )

            signal_np = data_p.detach().cpu().numpy()[0]  # (C, T)
            n_timesteps = signal_np.shape[1]
            t = np.arange(n_timesteps)

            for row, filter_idx in enumerate(top_filter_indices):
                # Conditional CRP: condition on class AND specific filter
                conditions = [{"y": target_int, layer_name: int(filter_idx)}]
                try:
                    cond_heatmap, _, _, _ = attributor(
                        data_p, conditions, composite, record_layer=[]
                    )
                    # cond_heatmap shape: (1, C, T)
                    heat_np = cond_heatmap.detach().cpu().numpy()[0]  # (C, T)
                except Exception as exc:
                    print(f"    Warning: CRP failed for filter {filter_idx}: {exc}")
                    heat_np = np.zeros_like(signal_np)

                # Normalize heatmap for colormap

                # abs_max = np.abs(heat_np).max()   # 99 percentile is more robust to outliers and more visually informative than absolute max
                abs_max = np.percentile(np.abs(heat_np), 99)
                if abs_max < 1e-12:
                    abs_max = 1.0
                heat_norm = heat_np / abs_max  # in [-1, 1]

                cmap = plt.get_cmap('bwr')
                norm = Normalize(vmin=-1, vmax=1)

                for col, ch_name in enumerate(used_axes):
                    ax = axes[row][col]
                    ch_heat = heat_norm[col] if col < heat_norm.shape[0] else heat_norm[0]
                    ch_sig = signal_np[col] if col < signal_np.shape[0] else signal_np[0]

                    # Color background segments by heatmap value
                    for ti in range(n_timesteps - 1):
                        color = cmap(norm(float(ch_heat[ti])))
                        ax.axvspan(t[ti], t[ti + 1], color=color, alpha=0.5, linewidth=0)

                    # Signal as black line on top
                    ax.plot(t, ch_sig, color='black', linewidth=0.6)

                    ax.set_xlim(0, n_timesteps - 1)
                    if row == 0:
                        ax.set_title(f'Axis {ch_name}', fontsize=10)
                    if col == 0:
                        ax.set_ylabel(f'Filter {filter_idx}', fontsize=9)
                    ax.tick_params(labelsize=7)

            plt.tight_layout()

            fig_path = os.path.join(
                output_dir,
                f'concept_heatmaps_class_{class_id}_proto_{proto_idx}.png'
            )
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            figures[(class_id, proto_idx)] = fig_path
            print(f"    Saved to {fig_path}")

    print(f"\n✓ Generated {len(figures)} concept heatmap figures")
    return figures


def plot_pcx_prototype_concept_grid(
    model,
    dataset,
    prototype_idx: int,
    proto_mean: np.ndarray,
    closest_sample_signal: np.ndarray,
    closest_sample_heatmap: np.ndarray,
    layer_name: str,
    composite,
    top_k: int = 5,
    class_id: int = 0,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    axes_names: List[str] = ['X', 'Y', 'Z'],
    figsize: Optional[Tuple[int, int]] = None,
    alpha: float = 0.5,
    cmap: str = 'bwr'
) -> plt.Figure:
    """
    PCX Figure 2-style combined prototype + concept grid for 1D vibration signals.

    Layout (adapted from PCX paper for 1D signals):
    - Left large panel: representative signal for the closest sample to the
      prototype, with the full CRP heatmap overlay.
    - Right panels (one per top-k concept/filter): same signal but showing
      ONLY that concept's conditional CRP heatmap.
    - Bottom bar chart: prototype filter-magnitude vector (μ), highlighting
      the top-k concepts.

    Args:
        model: PyTorch model.
        dataset: Dataset to load samples from.
        prototype_idx: Index of this prototype within the class GMM.
        proto_mean: GMM mean vector for this prototype, shape ``(n_filters,)``.
        closest_sample_signal: Raw signal array ``(C, T)`` for the closest sample.
        closest_sample_heatmap: Full CRP heatmap ``(C, T)`` for the closest sample.
        layer_name: Layer name to condition CRP on (e.g. ``'conv3'``).
        composite: Zennit composite for CRP attribution.
        top_k: Number of top concepts to show in the right panels.
        class_id: Class ID (0=OK, 1=NOK) used for the ``y`` condition.
        device: Device string.
        axes_names: Channel axis names.
        figsize: Figure size; auto-computed when ``None``.
        alpha: Heatmap overlay transparency.
        cmap: Colormap for heatmap overlay.

    Returns:
        Matplotlib figure.
    """
    from tcd.attribution import TimeSeriesCondAttribution

    n_channels = closest_sample_signal.shape[0]
    n_timesteps = closest_sample_signal.shape[1]
    t = np.arange(n_timesteps)

    # Top-k filter indices by absolute proto_mean magnitude
    top_filter_indices = np.argsort(np.abs(proto_mean))[-top_k:][::-1]

    used_axes = (axes_names[:n_channels] if len(axes_names) >= n_channels
                 else [f'Ch{c}' for c in range(n_channels)])

    # --- Layout ---
    # Rows: n_channels (one per signal axis)
    # Columns: 1 (full heatmap) + top_k (per-concept heatmaps)
    # Plus an extra row at the bottom for the bar chart
    n_cols = 1 + top_k
    n_rows = n_channels + 1  # +1 for bar chart

    if figsize is None:
        figsize = (3 * n_cols + 2, 3 * n_channels + 2)

    fig = plt.figure(figsize=figsize)
    class_name = "OK" if class_id == 0 else "NOK"
    fig.suptitle(
        f'{class_name} Prototype {prototype_idx} — Signal + Top-{top_k} Concept Heatmaps',
        fontsize=13, y=1.01
    )

    gs = fig.add_gridspec(
        n_rows, n_cols,
        height_ratios=[2] * n_channels + [1],
        hspace=0.35, wspace=0.2
    )

    cmap_obj = plt.get_cmap(cmap)

    def _overlay_signal_heatmap(ax, signal_ch, heat_ch, title='', ylabel=''):
        abs_max = np.percentile(np.abs(heat_ch), 99)
        if abs_max < 1e-12:
            abs_max = 1.0
        norm = Normalize(vmin=-abs_max, vmax=abs_max)
        for ti in range(n_timesteps - 1):
            color = cmap_obj(norm(float(heat_ch[ti])))
            ax.axvspan(t[ti], t[ti + 1], color=color, alpha=alpha, linewidth=0)
        ax.plot(t, signal_ch, color='black', linewidth=0.7)
        ax.set_xlim(0, n_timesteps - 1)
        ax.tick_params(labelsize=7)
        if title:
            ax.set_title(title, fontsize=9)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=9)

    # Column 0: full heatmap overlay
    for c in range(n_channels):
        ax = fig.add_subplot(gs[c, 0])
        ch_heat = closest_sample_heatmap[c] if c < closest_sample_heatmap.shape[0] else closest_sample_heatmap[0]
        ch_sig = closest_sample_signal[c] if c < closest_sample_signal.shape[0] else closest_sample_signal[0]
        title = 'Full heatmap' if c == 0 else ''
        _overlay_signal_heatmap(ax, ch_sig, ch_heat, title=title, ylabel=used_axes[c])

    # Compute per-concept conditional CRP heatmaps
    attributor = TimeSeriesCondAttribution(model, no_param_grad=True)
    signal_t = torch.tensor(closest_sample_signal, dtype=torch.float32).unsqueeze(0).to(device)

    concept_heatmaps: List[Optional[np.ndarray]] = []
    for filter_idx in top_filter_indices:
        conditions = [{"y": class_id, layer_name: int(filter_idx)}]
        try:
            cond_heatmap, _, _, _ = attributor(signal_t, conditions, composite, record_layer=[])
            heat_np = cond_heatmap.detach().cpu().numpy()[0]  # (C, T)
        except Exception:
            heat_np = np.zeros_like(closest_sample_signal)
        concept_heatmaps.append(heat_np)

    # Columns 1..top_k: per-concept conditional heatmaps
    for col_offset, (filter_idx, heat_np) in enumerate(zip(top_filter_indices, concept_heatmaps)):
        col = 1 + col_offset
        for c in range(n_channels):
            ax = fig.add_subplot(gs[c, col])
            ch_heat = heat_np[c] if c < heat_np.shape[0] else heat_np[0]
            ch_sig = closest_sample_signal[c] if c < closest_sample_signal.shape[0] else closest_sample_signal[0]
            title = f'Filter {filter_idx}' if c == 0 else ''
            _overlay_signal_heatmap(ax, ch_sig, ch_heat, title=title)

    # Bottom row: bar chart of prototype filter magnitudes
    ax_bar = fig.add_subplot(gs[n_channels, :])
    n_filters = len(proto_mean)
    bar_colors = ['#d62728' if i in top_filter_indices else '#aec7e8' for i in range(n_filters)]
    ax_bar.bar(np.arange(n_filters), np.abs(proto_mean), color=bar_colors, width=0.8)
    ax_bar.set_xlabel('Filter index', fontsize=9)
    ax_bar.set_ylabel('|μ|', fontsize=9)
    ax_bar.set_title('Prototype filter magnitudes (red = top-k)', fontsize=9)
    ax_bar.tick_params(labelsize=7)

    plt.tight_layout()
    return fig


def plot_umap_prototypes(
    features: np.ndarray,
    labels: np.ndarray,
    prototype_assignments: np.ndarray,
    gmm_means: Optional[Dict[int, np.ndarray]] = None,
    class_names: Dict[int, str] = {0: "OK", 1: "NOK"},
    figsize: Tuple[int, int] = (12, 8),
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42
) -> plt.Figure:
    """
    UMAP visualization of CRV (Concept Relevance Vector) space.

    Projects high-dimensional CRP feature vectors into 2D using UMAP (or PCA as
    fallback) and colors points by class and prototype assignment.  Optionally
    overlays GMM component means projected into the same 2D space.

    Args:
        features: CRV feature matrix of shape (N, n_filters).
        labels: Class labels of shape (N,).
        prototype_assignments: Prototype index per sample of shape (N,).
        gmm_means: Optional dict mapping class_id -> array (n_proto, n_filters)
            of GMM component means.  When provided, means are projected and
            plotted as large stars.
        class_names: Mapping from class ID to display name.
        figsize: Figure size.
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        random_state: Random seed for reproducibility.

    Returns:
        Matplotlib figure with two side-by-side scatter plots: one colored by
        class, one colored by prototype assignment.
    """
    # --- Dimensionality reduction ---
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state
        )
        method_name = "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=random_state)
        method_name = "PCA"

    # Fit on all samples (optionally include GMM means for consistent projection)
    all_points = features
    if gmm_means is not None:
        means_list = [m for m in gmm_means.values()]
        if means_list:
            all_points = np.vstack([features] + means_list)

    embedding = reducer.fit_transform(all_points)
    feat_emb = embedding[:len(features)]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f"CRV Space — {method_name} Projection", fontsize=14, fontweight='bold')

    # --- Left: colored by class ---
    ax = axes[0]
    class_colors = {0: 'green', 1: 'red'}
    for cid, cname in class_names.items():
        mask = labels == cid
        if mask.sum() == 0:
            continue
        ax.scatter(
            feat_emb[mask, 0], feat_emb[mask, 1],
            c=class_colors.get(cid, 'blue'),
            label=cname, alpha=0.5, s=20, edgecolors='none'
        )
    ax.set_title("By Class")
    ax.set_xlabel(f"{method_name} 1")
    ax.set_ylabel(f"{method_name} 2")
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.3)

    # --- Right: colored by prototype assignment ---
    ax = axes[1]
    unique_protos = np.unique(prototype_assignments[prototype_assignments >= 0])
    proto_cmap = plt.get_cmap('tab10')
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '+']

    for pi in unique_protos:
        mask = prototype_assignments == pi
        color = proto_cmap(int(pi) % 10)
        marker = markers[int(pi) % len(markers)]
        ax.scatter(
            feat_emb[mask, 0], feat_emb[mask, 1],
            c=[color], marker=marker,
            label=f"Proto {pi}", alpha=0.6, s=25, edgecolors='none'
        )

    # Overlay GMM means if provided
    if gmm_means is not None:
        offset = len(features)
        for cid, means in gmm_means.items():
            for pi in range(len(means)):
                mean_emb = embedding[offset]
                offset += 1
                ax.scatter(
                    mean_emb[0], mean_emb[1],
                    c='black', marker='*', s=200, zorder=5,
                    label=f"μ class{cid}" if pi == 0 else None
                )

    ax.set_title("By Prototype Assignment")
    ax.set_xlabel(f"{method_name} 1")
    ax.set_ylabel(f"{method_name} 2")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(markerscale=2, fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_umap_metadata(
    features: np.ndarray,
    labels: np.ndarray,
    metadata_df: Any,
    class_names: Dict[int, str] = {0: "OK", 1: "NOK"},
    figsize: Tuple[int, int] = (18, 6),
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42
) -> plt.Figure:
    """
    UMAP visualization colored by metadata (class, machine, operation).

    Creates 3 side-by-side panels: colored by class, by machine, by operation.

    Args:
        features: CRV feature matrix of shape (N, n_filters).
        labels: Class labels of shape (N,).
        metadata_df: DataFrame with columns 'machine' and 'operation'.
        class_names: Mapping from class ID to display name.
        figsize: Figure size.
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        random_state: Random seed for reproducibility.

    Returns:
        Matplotlib figure with three side-by-side scatter plots colored by
        class, machine, and operation respectively.
    """
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state
        )
        method_name = "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=random_state)
        method_name = "PCA"

    embedding = reducer.fit_transform(features)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f"CRV Space — {method_name} Colored by Metadata", fontsize=14, fontweight='bold')

    # Panel 1: By class
    ax = axes[0]
    class_colors = {0: 'green', 1: 'red'}
    for cid, cname in class_names.items():
        mask = labels == cid
        if mask.sum() > 0:
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=class_colors.get(cid, 'blue'), label=cname,
                alpha=0.5, s=20, edgecolors='none'
            )
    ax.set_title("By Class")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(markerscale=2)
    ax.grid(True, alpha=0.3)

    # Panel 2: By machine
    ax = axes[1]
    machines = metadata_df['machine'].values
    unique_machines = sorted(set(m for m in machines if m != 'unknown'))
    machine_cmap = plt.get_cmap('Set1')
    for i, machine in enumerate(unique_machines):
        mask = machines == machine
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            color=machine_cmap(i), label=machine,
            alpha=0.5, s=20, edgecolors='none'
        )
    ax.set_title("By Machine")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(markerscale=2)
    ax.grid(True, alpha=0.3)

    # Panel 3: By operation
    ax = axes[2]
    operations = metadata_df['operation'].values
    unique_ops = sorted(set(o for o in operations if o != 'unknown'))
    op_cmap = plt.get_cmap('tab10')
    for i, op in enumerate(unique_ops):
        mask = operations == op
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            color=op_cmap(i % 10), label=op,
            alpha=0.5, s=20, edgecolors='none'
        )
    ax.set_title("By Operation")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(markerscale=2, fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel(f"{method_name} 1")
        ax.set_ylabel(f"{method_name} 2")

    plt.tight_layout()
    return fig



def plot_concept_prototype_matrix(
    gmm_means: np.ndarray,
    class_id: int,
    n_top_concepts: int = 5,
    coverage_pct: Optional[np.ndarray] = None,
    class_names: Dict[int, str] = {0: "OK", 1: "NOK"},
    filter_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Concept-Prototype Matrix (PCX paper Figure 5 equivalent).

    Shows at a glance which concepts characterize each prototype.  Follows the
    PCX ``plot_prototypes_with_concepts.py`` logic:

        prototypes = torch.from_numpy(gmm.means_)
        top_concepts = torch.topk(prototypes.abs(), n_top_concepts).indices.flatten().unique()
        top_concepts = top_concepts[
            prototypes[:, top_concepts].abs().amax(0).argsort(descending=True)
        ]
        concept_matrix = prototypes[:, top_concepts].T  # (n_top, n_proto)

    Cell (i, j) shows ``μ_j[concept_i] * 100`` with red=positive, blue=negative
    intensity proportional to magnitude.

    Args:
        gmm_means: GMM component means of shape (n_prototypes, n_filters).
        class_id: Class index (used for title).
        n_top_concepts: Number of top concepts to display.
        coverage_pct: Optional per-prototype coverage percentages of shape
            (n_prototypes,) to show in column headers.
        class_names: Mapping from class_id to display name.
        filter_names: Optional list of filter labels.
        figsize: Figure size.

    Returns:
        Matplotlib figure containing the concept-prototype matrix.
    """
    import torch as _torch

    prototypes = _torch.from_numpy(np.array(gmm_means, dtype=np.float32))
    n_proto, n_filters = prototypes.shape

    # PCX-style top concept selection
    top_k = min(n_top_concepts, n_filters)
    top_concepts = _torch.topk(prototypes.abs(), top_k).indices.flatten().unique()
    # Sort by max absolute value across prototypes (most discriminative first)
    top_concepts = top_concepts[
        prototypes[:, top_concepts].abs().amax(0).argsort(descending=True)
    ]
    top_concepts_np = top_concepts.numpy()

    concept_matrix = prototypes[:, top_concepts].T.numpy()  # (n_top, n_proto)

    # Build labels
    if filter_names is not None:
        row_labels = [
            filter_names[int(i)] if int(i) < len(filter_names) else f"Filter {int(i)}"
            for i in top_concepts_np
        ]
    else:
        row_labels = [f"Filter {int(i)}" for i in top_concepts_np]

    class_name = class_names.get(class_id, f"Class {class_id}")
    col_labels = []
    for pi in range(n_proto):
        label = f"Proto {pi}"
        if coverage_pct is not None and pi < len(coverage_pct):
            label += f"\n({coverage_pct[pi]:.0f}%)"
        col_labels.append(label)

    # --- Plot ---
    n_rows_mat, n_cols_mat = concept_matrix.shape
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(
        f"Concept-Prototype Matrix — {class_name}",
        fontsize=14, fontweight='bold'
    )

    # Symmetric color scale
    vmax = np.abs(concept_matrix * 100).max()
    if vmax < 1e-6:
        vmax = 1.0

    im = ax.imshow(
        concept_matrix * 100,
        cmap='RdBu_r', vmin=-vmax, vmax=vmax,
        aspect='auto'
    )

    # Annotate cells
    for i in range(n_rows_mat):
        for j in range(n_cols_mat):
            val = concept_matrix[i, j] * 100
            text_color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax.text(j, i, f"{val:+.1f}", ha='center', va='center',
                    fontsize=9, color=text_color)

    ax.set_xticks(np.arange(n_cols_mat))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(np.arange(n_rows_mat))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Prototype")
    ax.set_ylabel("Concept (filter)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("μ × 100 (% contribution)", fontsize=9)

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
