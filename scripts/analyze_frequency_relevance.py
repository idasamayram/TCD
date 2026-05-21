#!/usr/bin/env python3
"""
Prototype-conditioned frequency relevance analysis.

This script complements ``scripts/analyze_frequency.py``. The Welch/PSD script
summarizes which frequencies are present in prototype-assigned samples; this
script summarizes which frequencies are relevant to the model by applying
one of the supported virtual inspection methods to saved input-level
relevance heatmaps:

- dft_lrp: existing DFT-LRP rule (tcd.frequency_relevance)
- vil_idft: Virtual Inspection Layer (IDFT) real-valued formulation
- vil_stdft: Virtual Inspection Layer with windowed STDFT aggregation

Expected pipeline:
    1. python scripts/run_analysis.py --output results/crp_features ...
    2. python scripts/discover_concepts.py --variant C --features results/crp_features \
           --output results/variantC_conv3 --layer conv3 ...
    3. python scripts/analyze_frequency_relevance.py \
           --data ./data \
           --features results/crp_features \
           --concepts results/variantC_conv3 \
           --output results/frequency_relevance
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
import os
import pickle
from typing import Dict, List, Tuple

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.cnn1d_model import VibrationDataset
from tcd.frequency_relevance import (
    DEFAULT_CNC_BANDS,
    band_relevance,
    dft_lrp_frequency_relevance,
)
from tcd.virtual_inspection_layer import (
    vil_idft_frequency_relevance,
    vil_stdft_frequency_relevance,
)


AXIS_NAMES = ["X", "Y", "Z"]


def _load_concepts(concepts_dir: str):
    results_path = os.path.join(concepts_dir, "results.pkl")
    model_path = os.path.join(concepts_dir, "tcd_model.pkl")
    if not os.path.exists(results_path) or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing results.pkl or tcd_model.pkl in {concepts_dir}. "
            "Run discover_concepts.py --variant C first."
        )
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    with open(model_path, "rb") as f:
        tcd = pickle.load(f)
    return results, tcd


def _load_heatmaps_and_sample_ids(features_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load heatmaps, labels, and original dataset indices in CRV feature order."""
    heatmaps: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    sample_ids: List[np.ndarray] = []

    for class_id in [0, 1]:
        heatmap_path = os.path.join(features_dir, f"heatmaps_class_{class_id}.hdf5")
        sample_id_path = os.path.join(features_dir, f"sample_ids_class_{class_id}.pt")
        if not os.path.exists(heatmap_path):
            raise FileNotFoundError(f"Missing heatmap file: {heatmap_path}")
        if not os.path.exists(sample_id_path):
            raise FileNotFoundError(f"Missing sample id file: {sample_id_path}")

        with h5py.File(heatmap_path, "r") as f:
            if "heatmaps" not in f:
                raise KeyError(f"Dataset 'heatmaps' not found in {heatmap_path}")
            class_heatmaps = np.asarray(f["heatmaps"], dtype=np.float64)

        class_sample_ids = np.asarray(torch.load(sample_id_path), dtype=int)
        if len(class_heatmaps) != len(class_sample_ids):
            raise ValueError(
                f"Class {class_id} heatmap/sample-id mismatch: "
                f"{len(class_heatmaps)} vs {len(class_sample_ids)}"
            )

        heatmaps.append(class_heatmaps)
        labels.append(np.full(len(class_heatmaps), class_id, dtype=int))
        sample_ids.append(class_sample_ids)

    return (
        np.concatenate(heatmaps, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(sample_ids, axis=0),
    )


def _assign_per_class_prototypes(
    features: np.ndarray,
    labels: np.ndarray,
    tcd,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return global prototype id, local prototype id, and class id for each sample.

    Global ids are offset by preceding class prototype counts for plotting.  Local
    ids remain the original GMM component index within the class.
    """
    global_assignments = np.full(len(features), -1, dtype=int)
    local_assignments = np.full(len(features), -1, dtype=int)
    assignment_classes = np.asarray(labels, dtype=int).copy()

    gmms = getattr(tcd.prototype_discovery, "gmms", {})
    offset = 0
    for class_id in [0, 1]:
        class_mask = labels == class_id
        if class_id not in gmms or class_mask.sum() == 0:
            continue
        gmm = gmms[class_id]
        local = gmm.predict(features[class_mask]).astype(int)
        local_assignments[class_mask] = local
        global_assignments[class_mask] = local + offset
        offset += gmm.n_components

    return global_assignments, local_assignments, assignment_classes


def _load_raw_signals(dataset: VibrationDataset, sample_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw signals from the dataset using saved original sample indices."""
    signals: List[np.ndarray] = []
    labels: List[int] = []
    for idx in sample_ids:
        x, y = dataset[int(idx)]
        signals.append(x.numpy())
        labels.append(int(y))
    return np.stack(signals, axis=0), np.asarray(labels, dtype=int)


def _get_closest_samples(
    features: np.ndarray,
    proto_mask: np.ndarray,
    proto_centroid: np.ndarray,
    max_samples: int,
) -> np.ndarray:
    """
    Select samples closest to prototype centroid in feature space.
    
    Args:
        features: Feature matrix (N, n_features)
        proto_mask: Boolean mask for samples assigned to this prototype
        proto_centroid: Prototype centroid (n_features,)
        max_samples: Max samples to return; <=0 returns all
        
    Returns:
        Indices of closest samples (sorted)
    """
    proto_indices = np.where(proto_mask)[0]
    
    if max_samples <= 0 or len(proto_indices) <= max_samples:
        return proto_indices
    
    # Compute distances to centroid for samples in this prototype
    proto_features = features[proto_indices]
    distances = np.linalg.norm(proto_features - proto_centroid[None, :], axis=1)
    
    # Select closest max_samples by distance
    closest_relative_idx = np.argsort(distances)[:max_samples]
    closest_absolute_idx = proto_indices[closest_relative_idx]
    
    return np.sort(closest_absolute_idx)


def _compute_frequency_relevance(
    method: str,
    signal: np.ndarray,
    relevance: np.ndarray,
    sample_rate: float,
    eps: float,
    renormalize: bool,
    window_width: int,
    window_shift: int | None,
    window_shape: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    if method == "dft_lrp":
        return dft_lrp_frequency_relevance(
            signal=signal,
            relevance=relevance,
            sample_rate=sample_rate,
            eps=eps,
            one_sided=True,
            renormalize=renormalize,
        )
    if method == "vil_idft":
        return vil_idft_frequency_relevance(
            signal=signal,
            relevance=relevance,
            sample_rate=sample_rate,
            eps=eps,
            one_sided=True,
            renormalize=renormalize,
        )
    if method == "vil_stdft":
        freqs, relevance_tf, diagnostics = vil_stdft_frequency_relevance(
            signal=signal,
            relevance=relevance,
            sample_rate=sample_rate,
            eps=eps,
            one_sided=True,
            renormalize=renormalize,
            window_width=window_width,
            window_shift=window_shift,
            window_shape=window_shape,
        )
        if relevance_tf.size == 0:
            freq_rel = np.zeros_like(freqs)
        else:
            freq_rel = np.mean(relevance_tf, axis=0)
        return freqs, freq_rel, diagnostics
    raise ValueError(f"Unknown method: {method}")


def _plot_prototype_relevance(
    output_dir: str,
    class_id: int,
    prototype_id: int,
    freqs: np.ndarray,
    mean_signed: np.ndarray,
    mean_abs: np.ndarray,
    method: str,
    n_samples: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    colors = ["tab:red", "tab:green", "tab:blue"]

    for axis_idx, axis_name in enumerate(AXIS_NAMES[:mean_signed.shape[0]]):
        axes[0].plot(freqs, mean_signed[axis_idx], color=colors[axis_idx], label=axis_name)
        axes[1].plot(freqs, mean_abs[axis_idx], color=colors[axis_idx], label=axis_name)

    for ax in axes:
        for _, low, high in DEFAULT_CNC_BANDS:
            ax.axvspan(low, high, alpha=0.06, color="grey")
        ax.set_xlim(0, min(200, freqs.max()))
        ax.set_xlabel("Frequency (Hz)")
        ax.legend()

    axes[0].set_title(f"Mean signed relevance ({method})")
    axes[0].set_ylabel("Relevance")
    axes[1].set_title(f"Mean absolute relevance ({method})")
    axes[1].set_ylabel("|Relevance|")
    fig.suptitle(f"Class {class_id} proto {prototype_id}: frequency relevance (n={n_samples})")
    fig.tight_layout()

    path = os.path.join(
        output_dir, f"class{class_id}_prototype{prototype_id}_freq_relevance_{method}.png"
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Prototype-conditioned frequency relevance analysis"
    )
    parser.add_argument('--data', required=True, help='Path to CNC data directory')
    parser.add_argument('--features', required=True, help='Path to CRP feature directory from run_analysis.py')
    parser.add_argument('--concepts', required=True, help='Path to Variant C output directory')
    parser.add_argument('--output', default='results/frequency_relevance', help='Output directory')
    parser.add_argument('--sample-rate', type=float, default=400.0, help='Sampling rate in Hz')
    parser.add_argument('--eps', type=float, default=1e-6, help='Division stabilizer')
    parser.add_argument('--max-samples-per-prototype', type=int, default=0,
                        help='Max samples per prototype (by distance-to-centroid); 0=use all')
    parser.add_argument('--renormalize', action='store_true',
                        help='Scale frequency relevance sums to match time relevance sums')
    parser.add_argument('--method', default='dft_lrp',
                        choices=['dft_lrp', 'vil_idft', 'vil_stdft'],
                        help='Frequency relevance method to use')
    parser.add_argument('--vil-window-width', type=int, default=128,
                        help='STDFT window width (samples)')
    parser.add_argument('--vil-window-shift', type=int, default=None,
                        help='STDFT window shift (samples); default uses 50% overlap')
    parser.add_argument('--vil-window-shape', type=str, default='rectangle',
                        choices=['rectangle', 'halfsine'],
                        help='STDFT window shape')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    results, tcd = _load_concepts(args.concepts)
    features = np.asarray(results['features'], dtype=np.float64)
    concept_labels = np.asarray(results['labels'], dtype=int)

    heatmaps, heatmap_labels, sample_ids = _load_heatmaps_and_sample_ids(args.features)
    if len(features) != len(heatmaps):
        raise ValueError(
            f"Feature/heatmap mismatch: {len(features)} CRVs vs {len(heatmaps)} heatmaps. "
            "Use matching run_analysis and discover_concepts outputs."
        )
    if not np.array_equal(concept_labels, heatmap_labels):
        print("Warning: concept labels differ from heatmap labels; using concept labels for assignments.")

    dataset = VibrationDataset(args.data)
    raw_signals, dataset_labels = _load_raw_signals(dataset, sample_ids)
    if not np.array_equal(dataset_labels, heatmap_labels):
        print("Warning: dataset labels from sample IDs differ from heatmap labels.")

    global_assign, local_assign, assignment_classes = _assign_per_class_prototypes(
        features, concept_labels, tcd
    )
    valid = global_assign >= 0
    if valid.sum() == 0:
        raise ValueError("No valid prototype assignments found in tcd_model.pkl")

    summary_rows: List[Dict[str, float]] = []
    conservation_rows: List[Dict[str, float]] = []
    prototype_stats: List[Dict[str, int]] = []

    prototype_keys = sorted({
        (int(assignment_classes[i]), int(local_assign[i]), int(global_assign[i]))
        for i in np.where(valid)[0]
    })

    print(f"\nAnalyzing {len(prototype_keys)} prototypes")
    print(f"Sampling strategy: closest {args.max_samples_per_prototype if args.max_samples_per_prototype > 0 else 'all'} samples to centroid\n")
    
    for class_id, proto_id, global_proto_id in prototype_keys:
        proto_mask = (assignment_classes == class_id) & (local_assign == proto_id)
        n_total = proto_mask.sum()
        
        # Get prototype centroid from GMM
        gmms = getattr(tcd.prototype_discovery, "gmms", {})
        proto_centroid = gmms[class_id].means_[proto_id]
        
        # Select closest samples
        proto_indices = _get_closest_samples(
            features, proto_mask, proto_centroid, args.max_samples_per_prototype
        )
        
        if len(proto_indices) == 0:
            continue

        n_used = len(proto_indices)
        print(
            f"Class {class_id}, prototype {proto_id} (global {global_proto_id}): "
            f"{n_used}/{n_total} samples (closest by centroid distance)"
        )
        
        prototype_stats.append({
            "class_id": class_id,
            "prototype_id": proto_id,
            "global_prototype_id": global_proto_id,
            "n_total_assigned": n_total,
            "n_analyzed": n_used,
            "pct_used": 100.0 * n_used / max(1, n_total),
        })

        per_axis_signed: List[List[np.ndarray]] = [[] for _ in range(raw_signals.shape[1])]
        per_axis_abs: List[List[np.ndarray]] = [[] for _ in range(raw_signals.shape[1])]
        freqs_ref = None

        for sample_index in proto_indices:
            for axis_idx in range(raw_signals.shape[1]):
                freqs, freq_rel, diagnostics = _compute_frequency_relevance(
                    method=args.method,
                    signal=raw_signals[sample_index, axis_idx],
                    relevance=heatmaps[sample_index, axis_idx],
                    sample_rate=args.sample_rate,
                    eps=args.eps,
                    renormalize=args.renormalize,
                    window_width=args.vil_window_width,
                    window_shift=args.vil_window_shift,
                    window_shape=args.vil_window_shape,
                )
                freqs_ref = freqs
                per_axis_signed[axis_idx].append(freq_rel)
                per_axis_abs[axis_idx].append(np.abs(freq_rel))
                conservation_rows.append({
                    "sample_feature_index": int(sample_index),
                    "dataset_sample_id": int(sample_ids[sample_index]),
                    "class_id": int(class_id),
                    "prototype_id": int(proto_id),
                    "global_prototype_id": int(global_proto_id),
                    "axis": AXIS_NAMES[axis_idx] if axis_idx < len(AXIS_NAMES) else str(axis_idx),
                    "method": args.method,
                    **diagnostics,
                })

        mean_signed = np.stack([
            np.mean(np.stack(axis_values, axis=0), axis=0)
            for axis_values in per_axis_signed
        ])
        mean_abs = np.stack([
            np.mean(np.stack(axis_values, axis=0), axis=0)
            for axis_values in per_axis_abs
        ])

        _plot_prototype_relevance(
            args.output, class_id, proto_id, freqs_ref, mean_signed, mean_abs, args.method, n_used
        )

        for axis_idx in range(mean_abs.shape[0]):
            axis_name = AXIS_NAMES[axis_idx] if axis_idx < len(AXIS_NAMES) else str(axis_idx)
            peak_idx = int(np.argmax(mean_abs[axis_idx]))
            row: Dict[str, float] = {
                "class_id": int(class_id),
                "prototype_id": int(proto_id),
                "global_prototype_id": int(global_proto_id),
                "axis": axis_name,
                "n_samples_analyzed": int(n_used),
                "n_samples_assigned": int(n_total),
                "method": args.method,
                "peak_freq_hz_abs_relevance": float(freqs_ref[peak_idx]),
                "total_abs_relevance": float(np.sum(mean_abs[axis_idx])),
                "total_signed_relevance": float(np.sum(mean_signed[axis_idx])),
            }
            row.update(band_relevance(freqs_ref, mean_abs[axis_idx], DEFAULT_CNC_BANDS, use_absolute=True))
            summary_rows.append(row)

    # Save prototype statistics
    stats_path = os.path.join(args.output, "prototype_sampling_stats.csv")
    if prototype_stats:
        with open(stats_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(prototype_stats[0].keys()))
            writer.writeheader()
            writer.writerows(prototype_stats)
        print(f"\nSaved prototype statistics to {stats_path}")

    summary_path = os.path.join(
        args.output, f"prototype_frequency_relevance_{args.method}.csv"
    )
    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"Saved prototype frequency relevance summary to {summary_path}")

    conservation_path = os.path.join(
        args.output, f"conservation_check_{args.method}.csv"
    )
    if conservation_rows:
        with open(conservation_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(conservation_rows[0].keys()))
            writer.writeheader()
            writer.writerows(conservation_rows)
    print(f"Saved conservation diagnostics to {conservation_path}")

    print(f"\n✓ Frequency relevance analysis complete: {args.output}")


if __name__ == "__main__":
    main()
