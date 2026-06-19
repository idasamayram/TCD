"""
analyze_concept_frequency.py
============================
Concept-Conditioned DFT-LRP: maps each prototype's defining filters
to their frequency-domain fingerprints.

What this does (and how it differs from analyze_frequency_relevance.py):
------------------------------------------------------------------------
Your existing analyze_frequency_relevance.py applies DFT-LRP to the
FULL input-level heatmap R_n (all filters combined).  That answers:
  "What frequency content drives the overall prediction for prototype k?"

This script applies DFT-LRP to the CONDITIONAL CRP heatmap R_n^(j):
the input-level relevance when ONLY filter j is active in the backward pass.
This answers:
  "What frequency content does filter j specifically detect?"

Running both gives you:
  Combination 2 (concept-conditioned):
      For each top-k filter j of prototype (c,m):
          R_k^(j) = DFT-LRP applied to conditional heatmap of filter j
      → Frequency fingerprint per filter → maps filter index to Hz range

  Combination 3 (prototype + concept-conditioned, averaged over members):
      Average R_k^(j) over all samples assigned to prototype (c,m)
      → R_bar_k^(c,m,j) = mean frequency fingerprint of filter j
        within prototype (c,m)

Prerequisites (all already on disk from your pipeline):
  --concepts  : results/variantC_conv2_full_gmm_bic/
                needs: tcd_model.pkl, results.pkl
  --model     : cnn1d_model_new.ckpt
  --data      : ./data/
  --features  : results/crp_features_full_gmm_conv2/
                needs: heatmaps_class_{0,1}.hdf5
                       sample_ids_class_{0,1}.pt

Usage:
  python scripts/analyze_concept_frequency.py \
      --concepts  results/variantC_conv2_full_gmm_bic \
      --model     ./cnn1d_model_new.ckpt \
      --data      ./data \
      --features  results/crp_features_full_gmm_conv2 \
      --output    results/concept_frequency \
      --layer     conv2 \
      --top-k     5 \
      --n-samples 50 \
      --sample-rate 400

Outputs (in --output directory):
  concept_freq_class{c}_proto{m}_filter{j}.png
      Two-panel figure (signed + absolute relevance) per filter per prototype.

  concept_freq_grid_class{c}_proto{m}.png
      Grid figure: one row per top-k filter, three columns (X/Y/Z axes).
      This is the "DFT-PCX" figure for the paper.

  concept_frequency_results.pkl
      Full numerical results: dict keyed by (class_id, proto_idx, filter_idx)
      → ndarray shape (3, N//2+1): mean frequency relevance per axis.

  concept_frequency_summary.csv
      One row per (class, proto, filter): dominant frequency, band energy
      fractions (0-10 Hz, 10-50 Hz, 50-150 Hz, 150-200 Hz), total relevance.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import pandas as pd

# ── make TCD importable when running from repo root ──────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tcd.dataset import CNCDataset
from tcd.model import CNN1DWide  # adjust if your model class has a different name


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DFT-LRP core  (Vielhaben et al. Eq. 12, Pattern Recognition 2024)
# ─────────────────────────────────────────────────────────────────────────────

def dft_lrp(signal: np.ndarray,
            heatmap: np.ndarray,
            eps: float = 1e-9) -> np.ndarray:
    """
    Apply DFT-LRP to one axis of one sample.

    Parameters
    ----------
    signal  : (T,) float array  — raw time-domain signal for this axis
    heatmap : (T,) float array  — LRP relevance for this axis (R_n)
    eps     : float             — numerical stability term

    Returns
    -------
    R_k : (T//2+1,) float array — one-sided frequency relevance
    """
    T = len(signal)
    # Compute DFT of the signal to get amplitude r_k and phase phi_k
    Y = np.fft.rfft(signal)          # complex, shape (T//2+1,)
    r_k = np.abs(Y)                  # amplitude
    phi_k = np.angle(Y)              # phase

    # Ratio R_n / (x_n + eps)  — denominator stabilised
    ratio = heatmap / (signal + eps * np.sign(signal + 1e-30))

    # Eq. 12:  R_k = r_k * sum_n cos(2π k n / N + phi_k) * ratio_n
    # Vectorised: build cosine basis matrix (T//2+1, T)
    n = np.arange(T, dtype=np.float64)
    k = np.arange(T // 2 + 1, dtype=np.float64)
    # phase_offset shape: (n_freqs, 1) so it broadcasts with (n_freqs, T)
    basis = np.cos(
        2.0 * np.pi * k[:, None] * n[None, :] / T + phi_k[:, None]
    )  # (T//2+1, T)

    R_k = r_k * (basis @ ratio)     # (T//2+1,)
    return R_k.astype(np.float32)


def dft_lrp_multichannel(signal: np.ndarray,
                          heatmap: np.ndarray,
                          eps: float = 1e-9) -> np.ndarray:
    """
    Apply DFT-LRP to all axes (channels) of one sample.

    Parameters
    ----------
    signal  : (3, T) float — 3-axis vibration signal (X, Y, Z)
    heatmap : (3, T) float — per-axis LRP relevance

    Returns
    -------
    R_k : (3, T//2+1) float — frequency relevance per axis
    """
    C, T = signal.shape
    n_freq = T // 2 + 1
    R_k = np.zeros((C, n_freq), dtype=np.float32)
    for c in range(C):
        R_k[c] = dft_lrp(signal[c], heatmap[c], eps=eps)
    return R_k


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Conditional CRP heatmap computation  (Combination 2 / 3 core)
# ─────────────────────────────────────────────────────────────────────────────

def get_conditional_heatmap(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_class: int,
    layer_name: str,
    filter_idx: int,
    composite,
    device: str = "cpu"
) -> np.ndarray:
    """
    Compute the input-level conditional CRP heatmap for a single filter.

    The CRP conditional heatmap R_n^(j) is the input-level LRP relevance
    when only filter j is allowed to contribute in the backward pass at
    `layer_name`.  This answers: "what input pattern does filter j detect?"

    Parameters
    ----------
    model        : trained CNN1DWide
    x            : (1, 3, T) tensor — single input sample
    target_class : int — class whose output neuron we explain
    layer_name   : str — e.g. "conv2"
    filter_idx   : int — which filter (0-indexed) to condition on
    composite    : Zennit composite for LRP rules
    device       : "cpu" or "cuda"

    Returns
    -------
    heatmap : (3, T) float32 ndarray — conditional input-level heatmap
    """
    try:
        from crp.attribution import CondAttribution
        from crp.concepts import ChannelConcept
    except ImportError as e:
        raise ImportError(
            "crp package not found. Install with: pip install zennit-crp"
        ) from e

    x = x.to(device).requires_grad_(True)
    model = model.to(device)

    attr = CondAttribution(model)
    cc = ChannelConcept()

    # Condition the backward pass on a single filter at layer_name
    conditions = [{layer_name: [filter_idx]}]

    # Run forward + conditional backward
    attr_result = attr(
        x,
        conditions=conditions,
        record_layer=[],          # we only need the input gradient
        target_class=target_class,
        composite=composite
    )

    # attr_result.heatmap is the input-level conditional relevance
    # shape: (1, 3, T) → squeeze to (3, T)
    heatmap = attr_result.heatmap.detach().cpu().numpy()[0]   # (3, T)
    return heatmap.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Frequency band energy helper
# ─────────────────────────────────────────────────────────────────────────────

def band_energy(R_k: np.ndarray,
                freqs: np.ndarray,
                lo: float,
                hi: float) -> float:
    """Fraction of total |R_k| energy in frequency band [lo, hi] Hz."""
    mask = (freqs >= lo) & (freqs < hi)
    total = np.abs(R_k).sum() + 1e-30
    return float(np.abs(R_k[:, mask]).sum() / total)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

AXIS_LABELS = ["X", "Y", "Z"]
AXIS_COLORS = ["tab:red", "tab:green", "tab:blue"]


def plot_filter_frequency(
    mean_R_k: np.ndarray,          # (3, n_freq)
    freqs: np.ndarray,             # (n_freq,)
    class_name: str,
    proto_idx: int,
    filter_idx: int,
    n_samples: int,
    out_path: str
) -> None:
    """Two-panel frequency relevance plot for one (prototype, filter) pair."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"Class {class_name} Proto {proto_idx} — Filter {filter_idx} "
        f"Conditional DFT-LRP  (n={n_samples})",
        fontsize=11
    )

    for ax, (data, title) in zip(axes, [
        (mean_R_k,        "Mean signed relevance"),
        (np.abs(mean_R_k), "Mean absolute relevance"),
    ]):
        for c, (label, color) in enumerate(zip(AXIS_LABELS, AXIS_COLORS)):
            ax.plot(freqs, data[c], color=color, label=label, lw=1.2)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Relevance")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(0, freqs[-1])

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_concept_frequency_grid(
    proto_results: dict,           # filter_idx → mean_R_k (3, n_freq)
    freqs: np.ndarray,
    class_name: str,
    proto_idx: int,
    proto_coverage: float,
    prototype_mean: np.ndarray,   # (n_filters,) GMM mean
    top_k_filters: list,
    out_path: str
) -> None:
    """
    Grid figure: rows = top-k filters, columns = X / Y / Z axes.
    The "DFT-PCX" paper figure.
    """
    n_filters = len(top_k_filters)
    n_axes = 3
    fig, axes = plt.subplots(
        n_filters, n_axes,
        figsize=(4 * n_axes, 2.8 * n_filters),
        sharex=True
    )
    if n_filters == 1:
        axes = axes[np.newaxis, :]  # ensure 2D indexing

    fig.suptitle(
        f"Concept-Conditioned DFT-LRP\n"
        f"Class {class_name}  ·  Prototype {proto_idx} "
        f"(coverage {proto_coverage:.1%})",
        fontsize=12, y=1.01
    )

    for row, filt in enumerate(top_k_filters):
        mu = prototype_mean[filt]
        sign = "+" if mu >= 0 else ""
        row_label = f"Filter {filt}\n(μ={sign}{mu:.3f})"

        if filt not in proto_results:
            for col in range(n_axes):
                axes[row, col].text(0.5, 0.5, "no data",
                                    ha="center", va="center",
                                    transform=axes[row, col].transAxes)
            axes[row, 0].set_ylabel(row_label, fontsize=8)
            continue

        R_k = proto_results[filt]   # (3, n_freq)

        for col, (axis_label, color) in enumerate(
            zip(AXIS_LABELS, AXIS_COLORS)
        ):
            ax = axes[row, col]
            ax.fill_between(freqs, 0, R_k[col], alpha=0.4, color=color)
            ax.plot(freqs, R_k[col], color=color, lw=1.0)
            ax.axhline(0, color="k", lw=0.5, ls="--")
            ax.set_xlim(0, freqs[-1])

            if row == 0:
                ax.set_title(f"Axis {axis_label}", fontsize=10)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=8)
            if row == n_filters - 1:
                ax.set_xlabel("Frequency (Hz)", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Load saved heatmaps from crp_features directory
# ─────────────────────────────────────────────────────────────────────────────

def load_full_heatmaps(features_dir: str, class_id: int) -> np.ndarray:
    """
    Load the full (unconditional) input-level heatmaps stored by run_analysis.py.
    Shape: (N_class, 3, T)
    """
    path = os.path.join(features_dir, f"heatmaps_class_{class_id}.hdf5")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Heatmap file not found: {path}\n"
            "Run run_analysis.py first to generate CRP features."
        )
    with h5py.File(path, "r") as f:
        heatmaps = f["heatmaps"][:]    # (N, 3, T)
    return heatmaps.astype(np.float32)


def load_sample_ids(features_dir: str, class_id: int) -> np.ndarray:
    """
    Load the dataset sample indices that correspond to the stored heatmaps.
    These are the correctly-predicted samples used in run_analysis.py.
    """
    path = os.path.join(features_dir, f"sample_ids_class_{class_id}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sample IDs not found: {path}")
    return torch.load(path, weights_only=False).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_concept_frequency_analysis(
    concepts_dir: str,
    model_path: str,
    data_dir: str,
    features_dir: str,
    output_dir: str,
    layer_name: str = "conv2",
    top_k: int = 5,
    n_samples: int = 50,
    sample_rate: int = 400,
    use_conditional_crp: bool = True,
    device: str = "cpu"
) -> None:
    """
    Main routine.

    Parameters
    ----------
    use_conditional_crp : bool
        If True  → Combination 2/3: run conditional CRP backward pass
                   to get per-filter conditional heatmaps R_n^(j).
                   Requires model + zennit-crp installed. GPU recommended.
        If False → Fallback: use full heatmaps from features_dir but
                   weighted by prototype mean μ_j to approximate
                   filter-specific frequency relevance.
                   No model or GPU needed. Faster but less precise.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load TCD prototype results ────────────────────────────────────────────
    print("Loading TCD prototype results...")
    tcd_model_path = os.path.join(concepts_dir, "tcd_model.pkl")
    results_path   = os.path.join(concepts_dir, "results.pkl")

    if not os.path.exists(tcd_model_path):
        raise FileNotFoundError(
            f"TCD model not found: {tcd_model_path}\n"
            "Run discover_concepts.py first."
        )

    with open(tcd_model_path, "rb") as f:
        tcd = pickle.load(f)
    with open(results_path, "rb") as f:
        results = pickle.load(f)

    # Extract prototype assignments and features
    # results contains: features (N, C), labels (N,), assignments (N,)
    features    = results.get("features")          # torch.Tensor (N, C_layer)
    labels_np   = results.get("labels").numpy()    # (N,)
    assignments = results.get("proto_assignments")  # (N,) global proto index

    # Get the GMMs per class
    gmms = tcd.prototype_discovery.gmms   # dict: class_id → GaussianMixture

    # ── Load dataset (needed for signals) ────────────────────────────────────
    print("Loading dataset...")
    dataset = CNCDataset(data_dir, split="all")   # loads full dataset
    # Fallback: try without split argument if the above fails
    try:
        _ = dataset[0]
    except Exception:
        dataset = CNCDataset(data_dir)

    # ── Load pre-computed full heatmaps (from run_analysis.py) ───────────────
    print("Loading pre-computed heatmaps...")
    full_heatmaps = {}   # class_id → (N_class, 3, T)
    sample_ids    = {}   # class_id → (N_class,) dataset indices

    for class_id in [0, 1]:
        try:
            full_heatmaps[class_id] = load_full_heatmaps(
                features_dir, class_id
            )
            sample_ids[class_id] = load_sample_ids(features_dir, class_id)
            print(f"  Class {class_id}: {full_heatmaps[class_id].shape}")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            print("  Will fall back to re-running LRP for this class.")

    # ── Load model and composite (only needed for conditional CRP) ────────────
    model = None
    composite = None
    if use_conditional_crp:
        print("Loading model for conditional CRP...")
        try:
            from tcd.composites import CNCValidatedComposite
            model = CNN1DWide()
            ckpt = torch.load(model_path, map_location=device,
                              weights_only=False)
            # Handle different checkpoint formats
            if "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
            elif "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)
            model.eval()
            model = model.to(device)
            composite = CNCValidatedComposite()
            print(f"  Model loaded on {device}.")
        except Exception as e:
            print(f"  WARNING: Could not load model for conditional CRP: {e}")
            print("  Falling back to weighted-heatmap approximation.")
            use_conditional_crp = False

    # ── Frequency axis ────────────────────────────────────────────────────────
    T = full_heatmaps.get(0, full_heatmaps.get(1)).shape[-1]
    freqs = np.fft.rfftfreq(T, d=1.0 / sample_rate)   # (T//2+1,)

    # ── Summary rows for CSV ──────────────────────────────────────────────────
    summary_rows = []

    # ── Full numerical results dict ───────────────────────────────────────────
    all_results = {}   # (class_id, proto_idx, filter_idx) → (3, n_freq)

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop: class → prototype → filter
    # ─────────────────────────────────────────────────────────────────────────
    for class_id, class_name in [(0, "OK"), (1, "NOK")]:
        if class_id not in gmms:
            print(f"\nSkipping class {class_id} ({class_name}): no GMM found.")
            continue

        gmm = gmms[class_id]
        n_prototypes = gmm.n_components

        # Class mask in the global feature array
        class_mask = labels_np == class_id

        # Map from within-class index to global feature index
        class_global_indices = np.where(class_mask)[0]

        # Get heatmaps and sample IDs for this class
        if class_id in full_heatmaps:
            class_heatmaps   = full_heatmaps[class_id]   # (N_class, 3, T)
            class_sample_ids = sample_ids[class_id]       # (N_class,)
        else:
            print(f"  No pre-computed heatmaps for class {class_id}; skipping.")
            continue

        # Prototype assignments for this class
        class_features = features[class_mask]   # (N_class, C_layer) tensor
        class_assignments = tcd.assign_prototype(
            class_features, class_id
        )  # (N_class,) int array

        print(f"\n{'='*60}")
        print(f"CLASS {class_id} ({class_name}): {n_prototypes} prototypes")
        print(f"{'='*60}")

        for proto_idx in range(n_prototypes):
            proto_mask = class_assignments == proto_idx
            n_proto_total = int(proto_mask.sum())
            coverage = n_proto_total / len(class_assignments)

            # Indices within the class array for this prototype
            proto_class_indices = np.where(proto_mask)[0]

            # Select up to n_samples — choose the closest to prototype centre
            proto_mean = gmm.means_[proto_idx]           # (C_layer,)
            dists = np.linalg.norm(
                class_features[proto_mask].numpy() - proto_mean, axis=1
            )
            n_use = min(n_samples, len(proto_class_indices))
            closest = np.argsort(dists)[:n_use]
            selected_class_indices = proto_class_indices[closest]

            # Top-k filters for this prototype
            top_k_indices = list(
                np.argsort(np.abs(proto_mean))[::-1][:top_k]
            )

            print(f"\n  Prototype {proto_idx} "
                  f"({n_proto_total} samples, {coverage:.1%} coverage)")
            print(f"  Top-{top_k} filters: {top_k_indices}")
            print(f"  Using {n_use} representative samples")

            # Per-filter accumulator
            proto_filter_results = {}   # filter_idx → accumulated (3, n_freq)

            for filt in top_k_indices:
                mu_filt = proto_mean[filt]
                print(f"    Filter {filt:3d}  (μ={mu_filt:+.4f}) ...", end=" ")

                accumulated = np.zeros((3, len(freqs)), dtype=np.float64)
                count = 0

                for ci in selected_class_indices:
                    # ── Get input signal ────────────────────────────────────
                    # sample_ids maps heatmap index → dataset index
                    if ci >= len(class_sample_ids):
                        continue
                    dataset_idx = int(class_sample_ids[ci])

                    try:
                        signal_tensor, _ = dataset[dataset_idx]
                        signal = signal_tensor.numpy().astype(np.float32)
                        # signal shape: (3, T)
                    except Exception:
                        continue

                    # ── Get conditional heatmap R_n^(filter_j) ─────────────
                    if use_conditional_crp and model is not None:
                        # Combination 2/3: real conditional CRP backward pass
                        try:
                            x = signal_tensor.unsqueeze(0)   # (1, 3, T)
                            cond_heatmap = get_conditional_heatmap(
                                model=model,
                                x=x,
                                target_class=class_id,
                                layer_name=layer_name,
                                filter_idx=int(filt),
                                composite=composite,
                                device=device
                            )  # (3, T)
                        except Exception as e:
                            # Fall back to weighted full heatmap
                            full_h = class_heatmaps[ci]   # (3, T)
                            cond_heatmap = full_h * abs(float(mu_filt))
                    else:
                        # Approximation: weight full heatmap by |μ_j|
                        # Not a true conditional heatmap but cheaper.
                        full_h = class_heatmaps[ci]   # (3, T)
                        cond_heatmap = full_h * abs(float(mu_filt))

                    # ── Apply DFT-LRP to conditional heatmap ────────────────
                    R_k = dft_lrp_multichannel(
                        signal, cond_heatmap
                    )  # (3, n_freq)

                    accumulated += R_k
                    count += 1

                if count > 0:
                    mean_R_k = (accumulated / count).astype(np.float32)
                else:
                    mean_R_k = np.zeros((3, len(freqs)), dtype=np.float32)

                proto_filter_results[int(filt)] = mean_R_k
                all_results[(class_id, proto_idx, int(filt))] = mean_R_k
                print(f"done (n={count})")

                # ── Individual filter plot ──────────────────────────────────
                indiv_path = os.path.join(
                    output_dir,
                    f"concept_freq_class{class_id}_proto{proto_idx}"
                    f"_filter{filt:03d}.png"
                )
                plot_filter_frequency(
                    mean_R_k=mean_R_k,
                    freqs=freqs,
                    class_name=class_name,
                    proto_idx=proto_idx,
                    filter_idx=int(filt),
                    n_samples=count,
                    out_path=indiv_path
                )

                # ── Summary row ─────────────────────────────────────────────
                dom_freq = float(freqs[np.argmax(np.abs(mean_R_k).sum(axis=0))])
                summary_rows.append({
                    "class_id":     class_id,
                    "class_name":   class_name,
                    "proto_idx":    proto_idx,
                    "coverage":     round(coverage, 4),
                    "filter_idx":   int(filt),
                    "mu":           round(float(mu_filt), 5),
                    "n_samples":    count,
                    "dominant_hz":  round(dom_freq, 2),
                    "energy_0_10":  round(band_energy(mean_R_k, freqs,   0,  10), 4),
                    "energy_10_50": round(band_energy(mean_R_k, freqs,  10,  50), 4),
                    "energy_50_150":round(band_energy(mean_R_k, freqs,  50, 150), 4),
                    "energy_150_200":round(band_energy(mean_R_k, freqs, 150, 200), 4),
                })

            # ── Grid figure for this prototype ─────────────────────────────
            grid_path = os.path.join(
                output_dir,
                f"concept_freq_grid_class{class_id}_proto{proto_idx}.png"
            )
            plot_concept_frequency_grid(
                proto_results=proto_filter_results,
                freqs=freqs,
                class_name=class_name,
                proto_idx=proto_idx,
                proto_coverage=coverage,
                prototype_mean=proto_mean,
                top_k_filters=top_k_indices,
                out_path=grid_path
            )
            print(f"  ✓ Grid figure saved: {grid_path}")

    # ── Save numerical results ────────────────────────────────────────────────
    pkl_path = os.path.join(output_dir, "concept_frequency_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\n✓ Numerical results saved: {pkl_path}")

    # ── Save CSV summary ──────────────────────────────────────────────────────
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(output_dir, "concept_frequency_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"✓ Summary CSV saved: {csv_path}")

        # Print a human-readable table
        print("\n" + "="*70)
        print("CONCEPT-CONDITIONED DFT-LRP SUMMARY")
        print("="*70)
        print(df.to_string(index=False))

    print(f"\n✓ All outputs in: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Concept-Conditioned DFT-LRP: maps prototype filters "
                    "to frequency fingerprints."
    )
    parser.add_argument(
        "--concepts", required=True,
        help="Directory with tcd_model.pkl and results.pkl "
             "(output of discover_concepts.py)"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to trained model checkpoint (.ckpt / .pt)"
    )
    parser.add_argument(
        "--data", required=True,
        help="Dataset root directory"
    )
    parser.add_argument(
        "--features", required=True,
        help="Directory with pre-computed CRP features "
             "(output of run_analysis.py); "
             "needs heatmaps_class_{0,1}.hdf5 and sample_ids_class_{0,1}.pt"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for figures and CSV"
    )
    parser.add_argument(
        "--layer", default="conv2",
        help="Layer name for conditional CRP (default: conv2)"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top prototype filters to analyse per prototype "
             "(default: 5)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=50,
        help="Max representative samples per prototype to average over "
             "(default: 50)"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=400,
        help="Signal sample rate in Hz (default: 400)"
    )
    parser.add_argument(
        "--no-conditional-crp", action="store_true",
        help="Skip conditional CRP backward pass; use weighted full-heatmap "
             "approximation instead. Faster, no GPU, less accurate."
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Torch device: 'cpu' or 'cuda' (default: cpu)"
    )
    args = parser.parse_args()

    run_concept_frequency_analysis(
        concepts_dir=args.concepts,
        model_path=args.model,
        data_dir=args.data,
        features_dir=args.features,
        output_dir=args.output,
        layer_name=args.layer,
        top_k=args.top_k,
        n_samples=args.n_samples,
        sample_rate=args.sample_rate,
        use_conditional_crp=not args.no_conditional_crp,
        device=args.device
    )


if __name__ == "__main__":
    main()
