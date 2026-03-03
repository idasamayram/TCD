#!/usr/bin/env python3
"""
Output layer geometry analysis for CNN1D_Wide.

Analyses the ``fc2`` weight vectors to understand decision geometry,
computes gradient-based filter importance, and saves visualisation plots.

Usage:
    python scripts/analyze_output.py \\
        --model ./cnn1d_model_final.ckpt \\
        --data ./data \\
        --output results/output_analysis
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))



import argparse
import os

import torch

from models.cnn1d_model import CNN1D_Wide, VibrationDataset
from tcd.output_analysis import OutputAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Analyse CNN1D_Wide output-layer geometry"
    )
    parser.add_argument(
        '--model', type=str, default='./cnn1d_model_final.ckpt',
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--data', type=str, default='./data',
        help='Path to data directory'
    )
    parser.add_argument(
        '--output', type=str, default='results/output_analysis',
        help='Output directory'
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='Torch device (default: cpu)'
    )
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.model):
        print(f"Error: Model checkpoint not found at {args.model}")
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

    # ------------------------------------------------------------------
    analyzer = OutputAnalyzer()

    # Weight geometry
    print("\n" + "=" * 60)
    print("OUTPUT WEIGHT GEOMETRY")
    print("=" * 60)
    info = analyzer.analyze_weights(model)
    print(f"  Cosine similarity (w0, w1) : {info['cosine_similarity']:.4f}")
    print(f"  Angle between w0 and w1   : {info['angle_degrees']:.2f}°")
    print(f"  ||w0||                    : {info['norm_w0']:.4f}")
    print(f"  ||w1||                    : {info['norm_w1']:.4f}")
    print(
        f"  Single neuron sufficient  : "
        f"{'YES (cos_sim < -0.9)' if info['single_neuron_sufficient'] else 'no'}"
    )

    # Dataset-dependent analyses
    dataset = None
    if os.path.exists(args.data):
        print("\nLoading dataset...")
        dataset = VibrationDataset(args.data)
        print(f"  Dataset size: {len(dataset)}")

        # Gradient-based filter importance
        print("\n" + "=" * 60)
        print("GRADIENT-BASED FILTER IMPORTANCE VIA OUTPUT")
        print("=" * 60)
        grad_importance = analyzer.analyze_filter_importance_via_output(
            model, dataset, device=args.device
        )
        for layer_name, importance in grad_importance.items():
            if importance.size == 0:
                continue
            top3 = importance.argsort()[::-1][:3]
            print(
                f"  {layer_name}: top-3 filters = {top3.tolist()}  "
                f"(importance: {importance[top3].tolist()})"
            )
    else:
        print(f"\nWarning: Data directory not found at {args.data}; skipping gradient analysis")

    # Plots
    print("\nGenerating output geometry plots...")
    analyzer.plot_output_geometry(model, args.output, dataset=dataset, device=args.device)

    print(f"\n✓ All results saved to {args.output}")


if __name__ == '__main__':
    main()
