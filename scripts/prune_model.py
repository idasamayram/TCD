#!/usr/bin/env python3
"""
Prune CNN1D_Wide using CRP relevance scores.

Loads pre-computed CRP features, computes per-filter importance, runs iterative
pruning across all four conv layers, and saves the best pruned model together
with diagnostic plots.

Usage:
    python scripts/prune_model.py \\
        --model ./cnn1d_model_final.ckpt \\
        --features results/crp_features \\
        --data ./data \\
        --output results/pruning
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.cnn1d_model import CNN1D_Wide, VibrationDataset
from tcd.pruning import RelevancePruner


def main():
    parser = argparse.ArgumentParser(
        description="Prune CNN1D_Wide using CRP relevance-based filter importance"
    )
    parser.add_argument(
        '--model', type=str, default='./cnn1d_model_final.ckpt',
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--features', type=str, default='results/crp_features',
        help='Path to CRP features directory (from run_analysis.py)'
    )
    parser.add_argument(
        '--data', type=str, default='./data',
        help='Path to data directory'
    )
    parser.add_argument(
        '--output', type=str, default='results/pruning',
        help='Output directory for pruning results'
    )
    parser.add_argument(
        '--keep-ratios', type=float, nargs='+',
        default=[0.9, 0.8, 0.7, 0.5, 0.3],
        help='Keep ratios to evaluate (default: 0.9 0.8 0.7 0.5 0.3)'
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='Torch device (default: cpu)'
    )
    parser.add_argument(
        '--concepts', type=str, default=None,
        help='Optional Variant C folder (results/concepts_C) for merged prototype-aware pruning'
    )
    parser.add_argument(
        '--layer', type=str, default='conv3', choices=['conv1', 'conv2', 'conv3', 'conv4'],
        help='Layer for merged/projection pruning (default: conv3)'
    )
    parser.add_argument(
        '--merge-alpha', type=float, default=0.5,
        help='Merged importance weight: alpha*relevance + (1-alpha)*prototype (default: 0.5)'
    )
    parser.add_argument(
        '--run-projection', action='store_true',
        help='Also run projection pruning analysis on --layer using selected importance'
    )
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.model):
        print(f"Error: Model checkpoint not found at {args.model}")
        sys.exit(1)
    if not os.path.exists(args.features):
        print(f"Error: CRP features directory not found at {args.features}")
        sys.exit(1)
    if not os.path.exists(args.data):
        print(f"Error: Data directory not found at {args.data}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # ------------------------------------------------------------------
    print("\nLoading model...")
    model = CNN1D_Wide()
    state = torch.load(args.model, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded model from {args.model}")
    n_orig = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Original parameters: {n_orig:,}")

    # ------------------------------------------------------------------
    print("\nLoading dataset...")
    dataset = VibrationDataset(args.data)
    print(f"  Dataset size: {len(dataset)}")

    # ------------------------------------------------------------------
    print("\nInitialising RelevancePruner...")
    pruner = RelevancePruner(args.features)

    # ------------------------------------------------------------------
    print("\nComputing filter importance for all layers...")
    layer_importance = pruner.compute_all_layer_importance()

    merged_importance = None
    if args.concepts:
        print(
            f"\nComputing merged importance for {args.layer} "
            f"(alpha={args.merge_alpha:.2f}) using {args.concepts}..."
        )
        merged_importance = pruner.compute_merged_importance(
            layer_name=args.layer,
            concepts_dir=args.concepts,
            merge_alpha=args.merge_alpha
        )
        layer_importance[args.layer] = merged_importance
        np.save(
            os.path.join(args.output, f"merged_importance_{args.layer}.npy"),
            merged_importance
        )
        print(f"  Saved merged importance for {args.layer}")

    # Plot filter importance bar charts
    for layer_name, importance in layer_importance.items():
        fig, ax = plt.subplots(figsize=(max(6, len(importance) // 2), 4))
        colors = plt.cm.RdYlGn(importance / (importance.max() + 1e-10))
        ax.bar(range(len(importance)), importance, color=colors)
        ax.set_xlabel('Filter index')
        ax.set_ylabel('Mean absolute relevance')
        ax.set_title(f'Filter importance — {layer_name}')
        plt.tight_layout()
        out_path = os.path.join(args.output, f'filter_importance_{layer_name}.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out_path}")

    # ------------------------------------------------------------------
    print(f"\nRunning iterative pruning with keep_ratios={args.keep_ratios}...")
    results = pruner.iterative_prune(
        model, dataset, keep_ratios=args.keep_ratios, device=args.device
    )

    # Print table
    print("\n" + "=" * 65)
    print(f"{'keep_ratio':>12}  {'n_params':>12}  {'accuracy':>10}  {'drop':>10}")
    print("-" * 65)
    for row in results:
        print(
            f"  {row['keep_ratio']:>10.2f}  "
            f"{row['n_params']:>12,}  "
            f"{row['accuracy']:>10.4f}  "
            f"{row['accuracy_drop']:>+10.4f}"
        )
    print("=" * 65)

    # Find knee point
    knee = pruner.find_knee_point(results, threshold=0.01)
    print(f"\nKnee point (>1% accuracy drop): keep_ratio = {knee:.2f}")

    # ------------------------------------------------------------------
    # Save best pruned model (at knee point)
    best_rows = [r for r in results if r['keep_ratio'] == knee]
    if best_rows:
        best_model = pruner.prune_all_layers(model, knee)
        best_path = os.path.join(args.output, 'best_pruned_model.pt')
        pruner.export_pruned_model(best_model, best_path)

    # ------------------------------------------------------------------
    # Accuracy-vs-compression curve
    ratios = [r['keep_ratio'] for r in results]
    accs = [r['accuracy'] for r in results]
    n_params_list = [r['n_params'] for r in results]
    compression = [n_orig / n for n in n_params_list]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(compression, accs, marker='o', linewidth=2)
    ax.axhline(accs[0] if results else 0, color='grey', linestyle='--',
               linewidth=1, label='Baseline (keep_ratio=1.0)')
    for row, comp in zip(results, compression):
        ax.annotate(
            f"{row['keep_ratio']:.1f}",
            (comp, row['accuracy']),
            textcoords='offset points', xytext=(4, 4), fontsize=8
        )
    ax.set_xlabel('Compression ratio (original / pruned params)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Compression')
    ax.legend()
    plt.tight_layout()
    curve_path = os.path.join(args.output, 'accuracy_vs_compression.png')
    fig.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved accuracy-vs-compression curve to {curve_path}")
    print(f"\n✓ All pruning results saved to {args.output}")

    # ------------------------------------------------------------------
    # Optional projection pruning analysis
    if args.run_projection:
        print("\nRunning projection pruning analysis...")
        if merged_importance is not None:
            projection_importance = merged_importance
            print(f"  Using merged importance for layer {args.layer}")
        else:
            projection_importance = layer_importance[args.layer]
            print(f"  Using relevance importance for layer {args.layer}")

        proj_results = pruner.evaluate_projection_pruning(
            model=model,
            dataset=dataset,
            layer_name=args.layer,
            importance=projection_importance,
            keep_ratios=args.keep_ratios,
            device=args.device
        )

        print("\n" + "=" * 72)
        print(f"{'PROJECTION keep_ratio':>22}  {'kept':>10}  {'accuracy':>10}  {'drop':>10}")
        print("-" * 72)
        for row in proj_results:
            print(
                f"  {row['keep_ratio']:>18.2f}  "
                f"{row['n_kept_filters']:>10d}  "
                f"{row['accuracy']:>10.4f}  "
                f"{row['accuracy_drop']:>+10.4f}"
            )
        print("=" * 72)

        proj_path = os.path.join(args.output, f'projection_pruning_{args.layer}.npy')
        np.save(proj_path, np.array(proj_results, dtype=object), allow_pickle=True)
        print(f"Saved projection pruning results to {proj_path}")


if __name__ == '__main__':
    main()
