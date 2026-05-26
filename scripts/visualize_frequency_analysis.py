#!/usr/bin/env python3
"""
Comprehensive frequency analysis visualizations.

Generates comparison plots across multiple frequency relevance methods
(DFT-LRP, VIL IDFT, VIL STDFT) and diagnostic visualizations for
prototype quality.

Expected input: directories from analyze_frequency_relevance.py with CSVs:
  - prototype_frequency_relevance_{method}.csv
  - conservation_check_{method}.csv
  - prototype_sampling_stats.csv (shared across methods)

Usage:
    python scripts/visualize_frequency_analysis.py \
        --methods-dir results/frequency_relevance_* \
        --concepts results/variantC_conv3 \
        --output results/frequency_visualizations
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os
import pickle
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    from sklearn.decomposition import PCA

from tcd.frequency_relevance import DEFAULT_CNC_BANDS


def _load_frequency_results(method_dir: str, method_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load frequency relevance and conservation CSVs for a method."""
    freq_path = os.path.join(method_dir, f"prototype_frequency_relevance_{method_name}.csv")
    cons_path = os.path.join(method_dir, f"conservation_check_{method_name}.csv")
    
    if not os.path.exists(freq_path):
        raise FileNotFoundError(f"Missing: {freq_path}")
    if not os.path.exists(cons_path):
        raise FileNotFoundError(f"Missing: {cons_path}")
    
    freq_df = pd.read_csv(freq_path)
    cons_df = pd.read_csv(cons_path)
    return freq_df, cons_df


def _load_concepts(concepts_dir: str):
    """Load GMM from concepts directory."""
    model_path = os.path.join(concepts_dir, "tcd_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing: {model_path}")
    with open(model_path, "rb") as f:
        tcd = pickle.load(f)
    return tcd


def plot_method_comparison(
    results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    output_dir: str,
) -> None:
    """
    Compare peak frequencies, total relevance, and conservation error
    across methods (DFT-LRP, VIL IDFT, VIL STDFT).
    """
    methods = sorted(results.keys())
    n_methods = len(methods)
    
    # Aggregate by method
    agg_data = {}
    for method in methods:
        freq_df, _ = results[method]
        freq_df = freq_df.groupby(['class_id', 'prototype_id']).agg({
            'peak_freq_hz_abs_relevance': 'mean',
            'total_abs_relevance': 'mean',
        }).reset_index()
        agg_data[method] = freq_df
    
    # Figure 1: Peak frequencies per method
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4), sharey=True)
    if n_methods == 1:
        axes = [axes]
    
    for ax, method in zip(axes, methods):
        df = agg_data[method]
        for class_id in [0, 1]:
            class_df = df[df['class_id'] == class_id]
            label = "OK" if class_id == 0 else "NOK"
            color = "green" if class_id == 0 else "red"
            ax.scatter(class_df['prototype_id'], class_df['peak_freq_hz_abs_relevance'],
                      label=label, color=color, s=100, alpha=0.7)
        ax.set_title(f"Peak Frequencies: {method}")
        ax.set_xlabel("Prototype ID")
        ax.set_ylabel("Peak Frequency (Hz)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    fig.suptitle("Peak Frequencies Across Methods", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_peak_frequencies_comparison.png"),
               dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✓ Created: 01_peak_frequencies_comparison.png")
    
    # Figure 2: Total relevance per method
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4), sharey=True)
    if n_methods == 1:
        axes = [axes]
    
    for ax, method in zip(axes, methods):
        df = agg_data[method]
        for class_id in [0, 1]:
            class_df = df[df['class_id'] == class_id]
            label = "OK" if class_id == 0 else "NOK"
            color = "green" if class_id == 0 else "red"
            ax.bar(class_df['prototype_id'] + (0.2 if class_id == 1 else -0.2),
                  class_df['total_abs_relevance'], width=0.4,
                  label=label, color=color, alpha=0.7)
        ax.set_title(f"Total Relevance: {method}")
        ax.set_xlabel("Prototype ID")
        ax.set_ylabel("Total |Relevance|")
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
    
    fig.suptitle("Total Relevance Across Methods", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_total_relevance_comparison.png"),
               dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✓ Created: 02_total_relevance_comparison.png")


def plot_frequency_bands_matrix(
    results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    output_dir: str,
) -> None:
    """
    Heatmap: (n_prototypes, n_bands) colored by band relevance.
    """
    methods = sorted(results.keys())
    
    for method in methods:
        freq_df, _ = results[method]
        
        # Extract band columns
        band_cols = [c for c in freq_df.columns if c.startswith('band_')]
        if not band_cols:
            continue
        
        # Aggregate by prototype
        proto_agg = freq_df.groupby(['class_id', 'prototype_id'])[band_cols].mean().reset_index()
        
        # Separate by class
        for class_id in [0, 1]:
            class_df = proto_agg[proto_agg['class_id'] == class_id]
            if class_df.empty:
                continue
            
            # Extract relevance (drop _ratio columns)
            rel_cols = [c for c in band_cols if not c.endswith('_ratio')]
            matrix = class_df[rel_cols].values
            proto_ids = class_df['prototype_id'].values
            
            # Band names from DEFAULT_CNC_BANDS
            band_names = [f"{low}-{high}Hz" for _, low, high in DEFAULT_CNC_BANDS]
            
            # Plot
            fig, ax = plt.subplots(figsize=(8, 4))
            im = ax.imshow(matrix.T, aspect='auto', cmap='YlOrRd')
            ax.set_yticks(range(len(band_names)))
            ax.set_yticklabels(band_names)
            ax.set_xticks(range(len(proto_ids)))
            ax.set_xticklabels(proto_ids)
            ax.set_xlabel("Prototype ID")
            ax.set_ylabel("Frequency Band")
            
            class_name = "OK" if class_id == 0 else "NOK"
            ax.set_title(f"Frequency Band Relevance: {class_name} Class ({method})")
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Relevance")
            
            fig.tight_layout()
            class_suffix = "ok" if class_id == 0 else "nok"
            fig.savefig(
                os.path.join(output_dir, f"03_frequency_bands_{class_suffix}_{method}.png"),
                dpi=150, bbox_inches='tight'
            )
            plt.close(fig)
            print(f"✓ Created: 03_frequency_bands_{class_suffix}_{method}.png")


def plot_conservation_error(
    results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    output_dir: str,
) -> None:
    """
    Bar chart: conservation error across methods.
    """
    methods = sorted(results.keys())
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    errors_by_method = {}
    for method in methods:
        _, cons_df = results[method]
        errors_by_method[method] = cons_df['conservation_error'].mean()
    
    x = np.arange(len(methods))
    errors = [errors_by_method[m] for m in methods]
    colors = ['steelblue', 'orange', 'green'][:len(methods)]
    
    ax.bar(x, errors, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Mean Conservation Error")
    ax.set_title("LRP Conservation Error Across Methods")
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on top of bars
    for i, (method, err) in enumerate(zip(methods, errors)):
        ax.text(i, err + 0.01, f"{err:.4f}", ha='center', va='bottom', fontsize=10)
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_conservation_error.png"),
               dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✓ Created: 04_conservation_error.png")


def plot_prototype_distance_histograms(
    tcd,
    features_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Diagnostic: histograms of distances to prototype centroid.
    """
    gmms = getattr(tcd.prototype_discovery, "gmms", {})
    
    for class_id, gmm in gmms.items():
        n_proto = gmm.n_components
        fig, axes = plt.subplots(1, n_proto, figsize=(4 * n_proto, 4))
        if n_proto == 1:
            axes = [axes]
        
        class_name = "OK" if class_id == 0 else "NOK"
        
        for proto_idx, ax in enumerate(axes):
            # Compute distances for this prototype
            proto_mask = (features_df['class_id'] == class_id) & (
                features_df['prototype_id'] == proto_idx
            )
            
            if proto_mask.sum() == 0:
                continue
            
            proto_features = features_df[proto_mask]['features'].values
            proto_centroid = gmm.means_[proto_idx]
            
            distances = np.linalg.norm(proto_features - proto_centroid[None, :], axis=1)
            
            ax.hist(distances, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_xlabel("Distance to Centroid")
            ax.set_ylabel("Count")
            ax.set_title(f"Proto {proto_idx}: {len(distances)} samples")
            ax.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(f"Prototype Distance Histograms: {class_name}", fontsize=14, fontweight='bold')
        fig.tight_layout()
        class_suffix = "ok" if class_id == 0 else "nok"
        fig.savefig(os.path.join(output_dir, f"05_distance_histograms_{class_suffix}.png"),
                   dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Created: 05_distance_histograms_{class_suffix}.png")


def plot_umap_with_covariance_ellipses(
    features_np: np.ndarray,
    labels: np.ndarray,
    prototype_assignments: np.ndarray,
    tcd,
    output_dir: str,
) -> None:
    """
    Enhanced UMAP with GMM covariance ellipses and centroids.
    """
    # Dimensionality reduction
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        method_name = "UMAP"
    else:
        reducer = PCA(n_components=2, random_state=42)
        method_name = "PCA"
    
    embedding = reducer.fit_transform(features_np)
    
    gmms = getattr(tcd.prototype_discovery, "gmms", {})
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot samples by class
    class_colors = {0: 'green', 1: 'red'}
    for class_id in [0, 1]:
        mask = labels == class_id
        class_name = "OK" if class_id == 0 else "NOK"
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                  c=class_colors[class_id], label=class_name, alpha=0.3, s=20)
    
    # Draw GMM covariance ellipses
    for class_id, gmm in gmms.items():
        for proto_idx in range(gmm.n_components):
            mean = gmm.means_[proto_idx]
            cov = gmm.covariances_[proto_idx]
            
            # Project mean and cov to 2D (approximate)
            # For now, just mark centroid
            proto_mask = (labels == class_id) & (prototype_assignments == proto_idx)
            if proto_mask.sum() > 0:
                proto_embedding = embedding[proto_mask]
                centroid_2d = proto_embedding.mean(axis=0)
                
                marker = 'o' if class_id == 0 else 's'
                ax.scatter(centroid_2d[0], centroid_2d[1],
                          marker=marker, s=200, c='black', edgecolor='yellow',
                          linewidth=2, zorder=5,
                          label=f"μ class{class_id} proto{proto_idx}" if proto_idx == 0 else "")
    
    ax.set_xlabel(f"{method_name} 1")
    ax.set_ylabel(f"{method_name} 2")
    ax.set_title(f"CRV Space with GMM Centroids ({method_name})")
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "06_umap_with_centroids.png"),
               dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✓ Created: 06_umap_with_centroids.png")


def plot_stdft_spectrograms(
    results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    output_dir: str,
) -> None:
    """
    For vil_stdft: show time-frequency heatmaps for select prototypes.
    """
    if 'vil_stdft' not in results:
        print("⊘ STDFT spectrograms: vil_stdft not in results, skipping.")
        return
    
    cons_df = results['vil_stdft'][1]
    
    # Group by prototype
    for (class_id, proto_id), group in cons_df.groupby(['class_id', 'prototype_id']):
        if len(group) < 5:
            continue  # Skip if too few samples
        
        # Aggregate time-frequency data (approximate)
        # For simplicity, plot conservation error over samples
        fig, ax = plt.subplots(figsize=(10, 4))
        
        ax.plot(range(len(group)), group['conservation_error'].values, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel("Sample Index (sorted by distance to centroid)")
        ax.set_ylabel("Conservation Error")
        ax.set_title(f"STDFT Conservation: Class {class_id} Proto {proto_id}")
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        class_suffix = "ok" if class_id == 0 else "nok"
        fig.savefig(
            os.path.join(output_dir, f"07_stdft_conservation_class{class_id}_proto{proto_id}.png"),
            dpi=150, bbox_inches='tight'
        )
        plt.close(fig)
    
    print("✓ Created: STDFT conservation plots")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive frequency analysis visualizations"
    )
    parser.add_argument('--methods-dir', nargs='+', required=True,
                       help='Directories from analyze_frequency_relevance.py (can specify multiple)')
    parser.add_argument('--concepts', required=True,
                       help='Path to Variant C output directory (for GMM)')
    parser.add_argument('--output', default='results/frequency_visualizations',
                       help='Output directory for visualizations')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Load results from each method
    results = {}
    for method_dir in args.methods_dir:
        # Infer method name from directory name
        method_name = os.path.basename(method_dir).replace('frequency_relevance_', '')
        print(f"Loading {method_name} from {method_dir}...")
        try:
            freq_df, cons_df = _load_frequency_results(method_dir, method_name)
            results[method_name] = (freq_df, cons_df)
        except FileNotFoundError as e:
            print(f"⚠ Skipping {method_name}: {e}")
    
    if not results:
        raise ValueError("No valid results directories found")
    
    print(f"\nLoaded {len(results)} method(s): {', '.join(results.keys())}")
    
    # Load concepts (GMM)
    print(f"\nLoading concepts from {args.concepts}...")
    tcd = _load_concepts(args.concepts)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_method_comparison(results, args.output)
    plot_frequency_bands_matrix(results, args.output)
    plot_conservation_error(results, args.output)
    plot_stdft_spectrograms(results, args.output)
    
    print(f"\n✓ All visualizations saved to {args.output}")


if __name__ == "__main__":
    main()
