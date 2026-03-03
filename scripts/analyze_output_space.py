#!/usr/bin/env python3
"""
Output space visualization and decision axis analysis for CNN1D_Wide.

Generates:
  - plots/output_space.png          — scatter of model logits (NOK visible on top)
  - plots/decision_axis_analysis.png — 1D score distribution, top-k filter weights,
                                       fixed output space scatter

Usage:
    python scripts/analyze_output_space.py \\
        --model ./cnn1d_model_final.ckpt \\
        --data  ./data \\
        --output plots
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

import torch

from models.cnn1d_model import CNN1D_Wide, VibrationDataset
from tcd.output_analysis import OutputAnalyzer


def main():
    parser = argparse.ArgumentParser(description='Output space & decision axis analysis')
    parser.add_argument('--model', required=True, help='Path to model checkpoint (.ckpt)')
    parser.add_argument('--data', required=True, help='Path to data directory')
    parser.add_argument('--output', default='plots', help='Output directory for plots')
    parser.add_argument('--device', default='cpu', help='Torch device (cpu/cuda)')
    parser.add_argument('--top-k', type=int, default=20,
                        help='Number of top decision-axis filters to show')
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model} …")
    model = CNN1D_Wide(num_classes=2, num_channels=3)
    state = torch.load(args.model, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    elif isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    model.eval()

    # Load dataset
    dataset = None
    data_path = Path(args.data)
    if data_path.exists():
        try:
            dataset = VibrationDataset(str(data_path), split='test')
            print(f"Loaded dataset: {len(dataset)} samples")
        except Exception as exc:
            print(f"Warning: could not load dataset — {exc}")
    else:
        print(f"Warning: data directory not found at {args.data}; skipping sample plots")

    analyzer = OutputAnalyzer()

    # Standard output geometry (weights + scatter with fix)
    print("\nGenerating output geometry plots …")
    analyzer.plot_output_geometry(model, args.output, dataset=dataset, device=args.device)

    # Decision axis analysis
    print("\nGenerating decision axis analysis …")
    analyzer.plot_decision_axis_analysis(
        model, args.output, dataset=dataset, device=args.device, top_k=args.top_k
    )

    print(f"\n✓ All plots saved to {args.output}")


if __name__ == '__main__':
    main()
