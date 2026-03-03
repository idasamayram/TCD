#!/usr/bin/env python3
"""
Metadata-driven prototype validation.

Parses CNC sample filenames, cross-references with discovered prototypes, and
generates distribution plots and a human-readable report.

Usage:
    python scripts/analyze_metadata.py \\
        --data ./data \\
        --concepts results/concepts_C \\
        --output results/metadata
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os
import pickle

import numpy as np

from models.cnn1d_model import VibrationDataset
from tcd.metadata import MetadataAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Analyse CNC sample metadata vs. prototype assignments"
    )
    parser.add_argument(
        '--data', type=str, default='./data',
        help='Path to data directory'
    )
    parser.add_argument(
        '--concepts', type=str, default='results/concepts_C',
        help='Path to concept results directory (from discover_concepts.py --variant C)'
    )
    parser.add_argument(
        '--output', type=str, default='results/metadata',
        help='Output directory for metadata analysis'
    )
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.data):
        print(f"Error: Data directory not found at {args.data}")
        sys.exit(1)

    results_pkl = os.path.join(args.concepts, 'results.pkl')
    if not os.path.exists(results_pkl):
        print(f"Error: Concept results not found at {results_pkl}")
        print("Run 'python scripts/discover_concepts.py --variant C ...' first.")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # ------------------------------------------------------------------
    print("\nLoading dataset...")
    dataset = VibrationDataset(args.data)
    print(f"  Dataset size: {len(dataset)}")

    # ------------------------------------------------------------------
    print("\nLoading prototype assignments...")
    with open(results_pkl, 'rb') as f:
        results = pickle.load(f)

    # Build per-sample prototype assignments (same logic as discover_concepts.py)
    features = results.get('features')
    labels_arr = results.get('labels')
    prototype_assignments = None

    # Try to reconstruct assignments from saved tcd_model
    tcd_model_path = os.path.join(args.concepts, 'tcd_model.pkl')
    if os.path.exists(tcd_model_path):
        try:
            import torch
            with open(tcd_model_path, 'rb') as f:
                tcd = pickle.load(f)
            features_t = torch.from_numpy(features).float()
            labels_t = torch.from_numpy(labels_arr).long()
            prototype_assignments = np.full(len(features), -1, dtype=int)
            for class_id in [0, 1]:
                class_mask = labels_arr == class_id
                if class_id in tcd.prototype_discovery.gmms:
                    gmm = tcd.prototype_discovery.gmms[class_id]
                    class_feat = features_t[class_mask]
                    assigns = gmm.predict(class_feat.numpy())
                    # Offset prototype IDs by class: assumes equal n_components per class
                    prototype_assignments[class_mask] = (
                        assigns + class_id * gmm.n_components
                    )
            print(f"  Loaded prototype assignments from tcd_model.pkl")
        except Exception as e:
            print(f"  Warning: Could not reconstruct assignments: {e}")
            prototype_assignments = None

    if prototype_assignments is None:
        print("  Warning: No prototype assignments found; using dummy -1 values")
        prototype_assignments = np.full(len(dataset), -1, dtype=int)

    # ------------------------------------------------------------------
    print("\nParsing filenames...")
    analyzer = MetadataAnalyzer()
    metadata_df = analyzer.parse_filenames(dataset, prototype_assignments)

    unknown_count = (metadata_df['machine'] == 'unknown').sum()
    print(
        f"  Parsed {len(metadata_df)} samples "
        f"({unknown_count} with unrecognised filename pattern)"
    )

    # ------------------------------------------------------------------
    print("\nGenerating metadata plots...")
    analyzer.plot_prototype_metadata(metadata_df, args.output)

    # ------------------------------------------------------------------
    analyzer.generate_report(metadata_df)

    # Save metadata DataFrame
    csv_path = os.path.join(args.output, 'metadata.csv')
    metadata_df.to_csv(csv_path, index=False)
    print(f"  Saved metadata table to {csv_path}")
    print(f"\n✓ All metadata results saved to {args.output}")


if __name__ == '__main__':
    main()
