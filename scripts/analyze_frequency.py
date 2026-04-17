#!/usr/bin/env python3
"""
Frequency-domain analysis of Variant C prototypes.

This script is intentionally post-hoc:
1) Load Variant C outputs (results.pkl + tcd_model.pkl)
2) Reconstruct per-sample prototype assignments
3) For each prototype, compute PSD (Welch) over raw CNC signals
4) Save prototype-level spectral summaries and plots

Usage:
    python scripts/analyze_frequency.py \
        --data ./data \
        --concepts results/concepts_C \
        --output results/frequency
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import torch

from models.cnn1d_model import VibrationDataset


def _load_signals_class_order(dataset: VibrationDataset) -> tuple[torch.Tensor, np.ndarray]:
    """
    Return signals and labels in class-concatenated order (class 0 then class 1),
    matching how run_analysis/discover_concepts concatenate per-class CRP features.
    """
    labels_np = dataset.labels
    idx_order = np.concatenate([
        np.where(labels_np == 0)[0],
        np.where(labels_np == 1)[0]
    ])

    signals = []
    labels = []
    for idx in idx_order:
        x, y = dataset[idx]
        signals.append(x)
        labels.append(int(y))
    signals_t = torch.stack(signals)  # (N, C, T)
    labels_np = np.asarray(labels, dtype=int)
    return signals_t, labels_np


def _assign_prototypes(features: np.ndarray, labels: np.ndarray, tcd, results: dict) -> np.ndarray:
    """Reconstruct per-sample prototype assignments for per-class or joint GMM."""
    assignments = np.full(len(features), -1, dtype=int)

    # Joint-GMM mode
    if results.get("joint_gmm", False):
        joint = getattr(tcd.prototype_discovery, "_joint_gmm", None)
        if joint is not None:
            return joint.predict(features).astype(int)
        raise ValueError("Results indicate joint_gmm=True but no joint model was found.")

    # Per-class mode
    gmms = getattr(tcd.prototype_discovery, "gmms", {})
    offset = 0
    for class_id in [0, 1]:
        class_mask = labels == class_id
        if class_id not in gmms or class_mask.sum() == 0:
            continue
        gmm = gmms[class_id]
        class_pred = gmm.predict(features[class_mask]).astype(int)
        assignments[class_mask] = class_pred + offset
        offset += gmm.n_components
    return assignments


def main():
    parser = argparse.ArgumentParser(description="Prototype frequency analysis (Variant C)")
    parser.add_argument('--data', type=str, required=True, help='Path to CNC data directory')
    parser.add_argument('--concepts', type=str, required=True, help='Path to Variant C output directory')
    parser.add_argument('--output', type=str, default='results/frequency', help='Output directory')
    parser.add_argument('--sample-rate', type=float, default=400.0, help='Sampling rate in Hz (default: 400)')
    parser.add_argument('--nperseg', type=int, default=256, help='Welch segment length (default: 256)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    results_pkl = os.path.join(args.concepts, 'results.pkl')
    tcd_pkl = os.path.join(args.concepts, 'tcd_model.pkl')
    if not os.path.exists(results_pkl) or not os.path.exists(tcd_pkl):
        raise FileNotFoundError("Missing results.pkl or tcd_model.pkl in --concepts path.")

    with open(results_pkl, 'rb') as f:
        results = pickle.load(f)
    with open(tcd_pkl, 'rb') as f:
        tcd = pickle.load(f)

    features = np.asarray(results['features'], dtype=np.float32)
    labels = np.asarray(results['labels'], dtype=int)

    print("Loading dataset...")
    dataset = VibrationDataset(args.data)
    signals, labels_dataset = _load_signals_class_order(dataset)

    if len(signals) != len(features):
        raise ValueError(
            f"Feature/sample mismatch: features={len(features)}, signals={len(signals)}. "
            "Ensure concepts were generated from the same dataset split."
        )
    if not np.array_equal(labels, labels_dataset):
        print("Warning: labels between concepts and dataset order differ; continuing with concepts labels.")

    proto_assign = _assign_prototypes(features, labels, tcd, results)
    valid = proto_assign >= 0
    if valid.sum() == 0:
        raise ValueError("No valid prototype assignments found.")

    unique_prototypes = sorted(np.unique(proto_assign[valid]).tolist())
    print(f"Found {len(unique_prototypes)} prototypes: {unique_prototypes}")

    # Predefine frequency bands used in CNC interpretation.
    bands = {
        "0-10": (0, 10),
        "10-50": (10, 50),
        "50-100": (50, 100),
        "100-200": (100, 200),
    }

    summary_rows = []

    for proto_id in unique_prototypes:
        mask = proto_assign == proto_id
        proto_signals = signals[mask].numpy()  # (N_p, C, T)
        n_samples = proto_signals.shape[0]
        if n_samples == 0:
            continue

        print(f"Analyzing prototype {proto_id}: {n_samples} samples")
        # Compute per-sample/channel PSD, then average
        psd_per_axis = []
        freqs_ref = None
        for axis in range(proto_signals.shape[1]):
            axis_psd = []
            for i in range(n_samples):
                freqs, pxx = welch(
                    proto_signals[i, axis, :],
                    fs=args.sample_rate,
                    nperseg=min(args.nperseg, proto_signals.shape[-1])
                )
                freqs_ref = freqs
                axis_psd.append(pxx)
            psd_per_axis.append(np.mean(np.stack(axis_psd), axis=0))
        psd_per_axis = np.stack(psd_per_axis, axis=0)  # (C, F)
        psd_mean = psd_per_axis.mean(axis=0)

        # Peak frequency from overall mean spectrum
        peak_idx = int(np.argmax(psd_mean))
        peak_freq = float(freqs_ref[peak_idx])

        # Band energy ratios
        total_energy = float(np.trapz(psd_mean, freqs_ref) + 1e-12)
        band_ratios = {}
        for band_name, (f0, f1) in bands.items():
            bmask = (freqs_ref >= f0) & (freqs_ref < f1)
            e = float(np.trapz(psd_mean[bmask], freqs_ref[bmask])) if bmask.any() else 0.0
            band_ratios[band_name] = e / total_energy

        # Save row
        row = {
            "prototype_id": int(proto_id),
            "n_samples": int(n_samples),
            "peak_freq_hz": peak_freq,
            **{f"band_ratio_{k}": v for k, v in band_ratios.items()}
        }
        summary_rows.append(row)

        # Plot overall and per-axis PSD
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(freqs_ref, psd_mean, color='black', linewidth=2, label='Mean PSD')
        for bname, (f0, f1) in bands.items():
            ax[0].axvspan(f0, f1, alpha=0.08, label=bname)
        ax[0].set_title(f'Prototype {proto_id} - mean PSD')
        ax[0].set_xlabel('Frequency (Hz)')
        ax[0].set_ylabel('Power')
        ax[0].set_xlim(0, 200)
        ax[0].legend(fontsize=7, ncol=2)

        colors = ['tab:red', 'tab:green', 'tab:blue']
        axis_names = ['X', 'Y', 'Z']
        for a in range(psd_per_axis.shape[0]):
            ax[1].plot(
                freqs_ref, psd_per_axis[a], color=colors[a],
                label=f'Axis {axis_names[a]}'
            )
        ax[1].set_title(f'Prototype {proto_id} - per-axis PSD')
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('Power')
        ax[1].set_xlim(0, 200)
        ax[1].legend()

        fig.tight_layout()
        fig.savefig(os.path.join(args.output, f'prototype_{proto_id}_psd.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Save summary CSV
    import csv
    csv_path = os.path.join(args.output, 'frequency_summary.csv')
    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"Saved summary CSV to {csv_path}")

    # Global bar chart: band ratios per prototype
    if summary_rows:
        proto_ids = [r['prototype_id'] for r in summary_rows]
        band_keys = [k for k in summary_rows[0].keys() if k.startswith("band_ratio_")]
        x = np.arange(len(proto_ids))
        width = 0.18
        fig, ax = plt.subplots(figsize=(9, 4))
        for i, bk in enumerate(band_keys):
            vals = [r[bk] for r in summary_rows]
            ax.bar(x + i * width, vals, width=width, label=bk.replace("band_ratio_", ""))
        ax.set_xticks(x + width * (len(band_keys) - 1) / 2)
        ax.set_xticklabels([str(p) for p in proto_ids])
        ax.set_xlabel("Prototype ID")
        ax.set_ylabel("Band energy ratio")
        ax.set_title("Prototype spectral fingerprints")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.output, 'prototype_band_ratios.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"\n✓ Frequency analysis complete. Outputs in {args.output}")


if __name__ == "__main__":
    main()
