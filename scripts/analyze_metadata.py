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
            features_t = torch.from_numpy(features).float()
            prototype_assignments = np.full(len(features), -1, dtype=int)

            # Check for joint-GMM first (single GMM fitted on all samples)
            joint_gmm = results.get('joint_gmm') or getattr(
                tcd.prototype_discovery, 'joint_gmm', None
            )
            if joint_gmm is not None:
                prototype_assignments = joint_gmm.predict(features).astype(int)
                print(f"  Loaded prototype assignments from joint GMM")
            else:
                # Per-class GMMs
                found_per_class_gmm = False
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
                        found_per_class_gmm = True
                if found_per_class_gmm:
                    print(f"  Loaded prototype assignments from tcd_model.pkl")
                else:
                    print(f"  Warning: tcd_model has neither joint_gmm nor per-class gmms")
                    prototype_assignments = None
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
    # UMAP colored by metadata + prototypes
    print("\nGenerating combined metadata + prototype UMAP...")
    try:
        from tcd.visualization import plot_umap_metadata_with_prototypes

        # Build gmm_means_dict if available
        gmm_means_dict = None
        try:
            with open(tcd_model_path, 'rb') as f:
                tcd_reload = pickle.load(f)
            gmm_means_dict = {}
            for class_id in [0, 1]:
                if class_id in tcd_reload.prototype_discovery.gmms:
                    gmm_means_dict[class_id] = tcd_reload.prototype_discovery.gmms[class_id].means_
        except Exception:
            gmm_means_dict = None

        fig = plot_umap_metadata_with_prototypes(
            features=features_np,
            labels=labels_np,
            prototype_assignments=prototype_assignments,
            metadata_df=metadata_df,
            gmm_means=gmm_means_dict,
        )
        combined_path = os.path.join(args.output, 'umap_metadata_prototypes.png')
        fig.savefig(combined_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved combined UMAP to {combined_path}")
    except Exception as e:
        print(f"  Warning: Could not generate combined UMAP: {e}")
        import traceback
        traceback.print_exc()

    # ------------------------------------------------------------------
    analyzer.generate_report(metadata_df)

    # Save metadata DataFrame
    csv_path = os.path.join(args.output, 'metadata.csv')
    metadata_df.to_csv(csv_path, index=False)
    print(f"  Saved metadata table to {csv_path}")
    print(f"\n✓ All metadata results saved to {args.output}")


if __name__ == '__main__':
    main()
