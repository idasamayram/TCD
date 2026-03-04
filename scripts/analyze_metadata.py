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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
            if len(features) != len(dataset):
                print(
                    f"  Warning: features length ({len(features)}) != dataset length "
                    f"({len(dataset)}). Some samples may not have prototype assignments."
                )
            features_t = torch.from_numpy(features).float()
            # Build per-class cumulative offset so unequal n_components are handled correctly
            feature_assignments = np.full(len(features), -1, dtype=int)
            n_prototypes_per_class = {}
            offset = 0
            for class_id in [0, 1]:
                class_mask = labels_arr == class_id
                if class_id in tcd.prototype_discovery.gmms:
                    gmm = tcd.prototype_discovery.gmms[class_id]
                    n_prototypes_per_class[class_id] = gmm.n_components
                    class_feat = features_t[class_mask]
                    if len(class_feat) > 0:
                        assigns = gmm.predict(class_feat.numpy())
                        feature_assignments[class_mask] = assigns + offset
                    offset += gmm.n_components
            # Copy into a dataset-sized array (entries beyond len(features) stay -1)
            prototype_assignments = np.full(len(dataset), -1, dtype=int)
            copy_len = min(len(features), len(dataset))
            prototype_assignments[:copy_len] = feature_assignments[:copy_len]
            print(f"  Loaded prototype assignments from tcd_model.pkl")
        except Exception as e:
            print(f"  Warning: Could not reconstruct assignments: {e}")
            prototype_assignments = None
            n_prototypes_per_class = {}

    if prototype_assignments is None:
        print("  Warning: No prototype assignments found; using dummy -1 values")
        prototype_assignments = np.full(len(dataset), -1, dtype=int)
        n_prototypes_per_class = {}

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
    # UMAP colored by metadata
    print("\nGenerating UMAP colored by metadata...")
    try:
        from tcd.visualization import plot_umap_metadata

        features_np = features.astype(np.float32) if not isinstance(features, np.ndarray) else features
        labels_np = labels_arr.astype(int)

        fig = plot_umap_metadata(
            features=features_np,
            labels=labels_np,
            metadata_df=metadata_df,
        )
        umap_path = os.path.join(args.output, 'umap_metadata.png')
        fig.savefig(umap_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved metadata UMAP to {umap_path}")
    except Exception as e:
        print(f"  Warning: Could not generate metadata UMAP: {e}")
        import traceback
        traceback.print_exc()

    # ------------------------------------------------------------------
    analyzer.generate_report(metadata_df, n_prototypes_per_class=n_prototypes_per_class)

    # Save metadata DataFrame
    csv_path = os.path.join(args.output, 'metadata.csv')
    metadata_df.to_csv(csv_path, index=False)
    print(f"  Saved metadata table to {csv_path}")
    print(f"\n✓ All metadata results saved to {args.output}")


if __name__ == '__main__':
    main()
