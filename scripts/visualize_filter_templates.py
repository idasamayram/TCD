#!/usr/bin/env python3
"""
Visualize what decisive 1D CNN filters respond to.

The PCX/CRP prototype plots tell us which filters are decisive for a prototype.
This script answers the next question: "what signal pattern does filter F capture?"

For each requested filter/channel at a convolutional layer it:
1. records activations over the dataset,
2. finds the samples and activation positions with the largest response,
3. maps the activation position back to an input-time window,
4. saves a PCX-style reference gallery with raw windows, activation traces, and
   frequency spectra, and
5. writes a CSV summary with peak-frequency and time-window metadata.
"""

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from models.cnn1d_model import CNN1D_Wide, VibrationDataset


def parse_filter_ids(filters: str) -> List[int]:
    """Parse a comma-separated filter list such as ``"3,17,42"``."""
    parsed = []
    for item in filters.split(","):
        item = item.strip()
        if not item:
            continue
        parsed.append(int(item))
    if not parsed:
        raise ValueError("At least one filter id must be provided.")
    return parsed


def load_config(config_path: str) -> dict:
    """Load a YAML config, returning an empty dict if no config path is provided."""
    if not config_path:
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def get_config_value(config: dict, section: str, key: str, default=None):
    """Safely read ``config[section][key]`` with a default."""
    return config.get(section, {}).get(key, default)


def load_model(model_path: str, device: torch.device) -> CNN1D_Wide:
    """Load CNN1D_Wide, accepting plain or Lightning-style checkpoint dicts."""
    model = CNN1D_Wide()
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        state_dict = {
            key.replace("model.", "", 1): value
            for key, value in state_dict.items()
        }
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model.to(device)
    model.eval()
    return model


def select_filters_from_concepts(
    concepts_dir: str,
    layer_name: str,
    n_filters: int,
    prototype_class: Optional[int] = None,
    prototype_id: Optional[int] = None
) -> List[int]:
    """
    Select decisive filters directly from Variant C GMM prototype means.

    If ``prototype_class`` and ``prototype_id`` are provided, filters are selected
    from that single prototype. Otherwise the score for each filter is the max
    absolute prototype mean across all available class/prototype centers.
    """
    tcd_path = Path(concepts_dir) / "tcd_model.pkl"
    results_path = Path(concepts_dir) / "results.pkl"
    if not tcd_path.exists():
        raise FileNotFoundError(f"Missing Variant C model at {tcd_path}")

    if results_path.exists():
        with results_path.open("rb") as f:
            results = pickle.load(f)
        result_layer = results.get("layer_name")
        if result_layer and result_layer != layer_name:
            print(
                f"Warning: concepts were discovered on layer '{result_layer}', "
                f"but --layer is '{layer_name}'. Use the same layer for filter templates."
            )

    with tcd_path.open("rb") as f:
        tcd = pickle.load(f)

    gmms = getattr(getattr(tcd, "prototype_discovery", None), "gmms", {})
    if not gmms:
        raise ValueError("No per-class GMMs found in tcd_model.pkl. Run Variant C without --joint-gmm.")

    if prototype_class is not None:
        if prototype_class not in gmms:
            raise KeyError(f"Class {prototype_class} not found in GMMs. Available classes: {sorted(gmms)}")
        gmm = gmms[prototype_class]
        if prototype_id is None:
            scores = np.max(np.abs(gmm.means_), axis=0)
        else:
            if prototype_id >= gmm.means_.shape[0]:
                raise IndexError(
                    f"Prototype {prototype_id} out of range for class {prototype_class} "
                    f"with {gmm.means_.shape[0]} prototypes."
                )
            scores = np.abs(gmm.means_[prototype_id])
    else:
        if prototype_id is not None:
            raise ValueError("--prototype-id requires --prototype-class.")
        scores = np.max(
            np.vstack([np.abs(gmm.means_) for gmm in gmms.values()]),
            axis=0
        )

    top = np.argsort(scores)[-n_filters:][::-1]
    selected = [int(idx) for idx in top]
    print(f"Selected filters from {concepts_dir}: {selected}")
    return selected


def get_named_module(model: nn.Module, layer_name: str) -> nn.Module:
    """Return a module by name, raising a helpful error if it does not exist."""
    modules = dict(model.named_modules())
    if layer_name not in modules:
        available = [name for name, module in modules.items() if isinstance(module, nn.Conv1d)]
        raise KeyError(f"Layer '{layer_name}' not found. Available Conv1d layers: {available}")
    return modules[layer_name]


def iter_receptive_field_modules(model: CNN1D_Wide) -> Iterable[Tuple[str, nn.Module]]:
    """
    Yield modules in the CNN1D_Wide temporal order used for receptive-field math.

    Hooks in this workflow target convolution outputs before the following ReLU
    and pooling operation. Therefore the target layer itself is included, but
    the pool after the target layer is not included.
    """
    ordered_names = ["conv1", "pool1", "conv2", "pool2", "conv3", "pool3", "conv4", "pool4"]
    for name in ordered_names:
        yield name, getattr(model, name)


def compute_layer_geometry(model: CNN1D_Wide, layer_name: str) -> Tuple[float, float, float]:
    """
    Compute receptive-field size, temporal stride, and first-center offset.

    Returns:
        ``(receptive_field_samples, jump_samples, start_center_samples)``.
    """
    receptive_field = 1.0
    jump = 1.0
    start = 0.5

    for name, module in iter_receptive_field_modules(model):
        if isinstance(module, (nn.Conv1d, nn.MaxPool1d)):
            kernel = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
            stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
            dilation = module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation

            start = start + (((kernel - 1) * dilation) / 2.0 - padding) * jump
            receptive_field = receptive_field + (kernel - 1) * dilation * jump
            jump *= stride

        if name == layer_name:
            return receptive_field, jump, start

    raise KeyError(f"Layer '{layer_name}' is not part of CNN1D_Wide receptive-field order.")


def activation_index_to_input_center(time_idx: int, jump: float, start: float, signal_len: int) -> int:
    """Map a layer activation index back to the approximate input-time center."""
    center = int(round(start + time_idx * jump - 0.5))
    return int(np.clip(center, 0, signal_len - 1))


def extract_window(signal: np.ndarray, center: int, window_size: int) -> Tuple[np.ndarray, int, int]:
    """Extract a fixed-size input window around ``center`` from ``signal`` (C, T)."""
    n_timesteps = signal.shape[-1]
    half = window_size // 2
    start = max(0, center - half)
    end = min(n_timesteps, start + window_size)
    start = max(0, end - window_size)
    return signal[:, start:end], start, end


def zscore_channels(window: np.ndarray) -> np.ndarray:
    """Per-channel z-score for plotting shape independently of amplitude scale."""
    mean = window.mean(axis=1, keepdims=True)
    std = window.std(axis=1, keepdims=True) + 1e-8
    return (window - mean) / std


def spectrum(window: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute mean magnitude spectrum over channels and return peak frequency."""
    centered = window - window.mean(axis=1, keepdims=True)
    magnitude = np.abs(np.fft.rfft(centered, axis=1)).mean(axis=0)
    freqs = np.fft.rfftfreq(window.shape[-1], d=1.0 / sample_rate)
    if len(magnitude) > 1:
        peak_idx = int(np.argmax(magnitude[1:]) + 1)
    else:
        peak_idx = 0
    return freqs, magnitude, float(freqs[peak_idx])


def collect_top_filter_windows(
    model: CNN1D_Wide,
    dataset: VibrationDataset,
    layer_name: str,
    filter_ids: List[int],
    top_k: int,
    window_size: int,
    sample_rate: float,
    batch_size: int,
    device: torch.device,
    class_id: Optional[int] = None
) -> Dict[int, List[dict]]:
    """Find top-activating input windows for each requested filter."""
    layer = get_named_module(model, layer_name)
    captured = {}

    def hook(_module, _inputs, output):
        captured["activation"] = output.detach().cpu()

    handle = layer.register_forward_hook(hook)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    receptive_field, jump, start_center = compute_layer_geometry(model, layer_name)
    effective_window = max(window_size, int(np.ceil(receptive_field)))

    candidates: Dict[int, List[dict]] = {filter_id: [] for filter_id in filter_ids}
    sample_offset = 0

    try:
        with torch.no_grad():
            for signals, labels in loader:
                signals_device = signals.to(device)
                _ = model(signals_device)
                activations = captured["activation"]

                for batch_idx in range(signals.shape[0]):
                    label = int(labels[batch_idx].item())
                    if class_id is not None and label != class_id:
                        continue

                    signal_np = signals[batch_idx].cpu().numpy()
                    signal_len = signal_np.shape[-1]
                    dataset_idx = sample_offset + batch_idx

                    for filter_id in filter_ids:
                        if filter_id >= activations.shape[1]:
                            raise IndexError(
                                f"Filter {filter_id} is out of range for {layer_name} "
                                f"with {activations.shape[1]} channels."
                            )
                        trace = activations[batch_idx, filter_id].numpy()
                        time_idx = int(np.argmax(trace))
                        activation_value = float(trace[time_idx])
                        center = activation_index_to_input_center(time_idx, jump, start_center, signal_len)
                        window, start, end = extract_window(signal_np, center, effective_window)
                        freqs, magnitude, peak_freq = spectrum(window, sample_rate=sample_rate)

                        candidates[filter_id].append({
                            "activation": activation_value,
                            "dataset_idx": int(dataset_idx),
                            "label": label,
                            "activation_idx": time_idx,
                            "input_center": center,
                            "input_start": int(start),
                            "input_end": int(end),
                            "window": window,
                            "activation_trace": trace,
                            "freqs": freqs,
                            "spectrum": magnitude,
                            "peak_freq_hz": peak_freq,
                        })

                sample_offset += signals.shape[0]
    finally:
        handle.remove()

    return {
        filter_id: sorted(rows, key=lambda row: row["activation"], reverse=True)[:top_k]
        for filter_id, rows in candidates.items()
    }


def plot_filter_gallery(
    filter_id: int,
    layer_name: str,
    rows: List[dict],
    output_dir: Path,
    sample_rate: float,
    channel_names: Tuple[str, ...] = ("X", "Y", "Z")
) -> None:
    """Save a gallery that acts as a visual template for one filter."""
    if not rows:
        return

    n_rows = len(rows) + 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    windows = [row["window"] for row in rows]
    min_len = min(window.shape[-1] for window in windows)
    windows = [window[:, :min_len] for window in windows]
    template = np.mean(np.stack([zscore_channels(window) for window in windows]), axis=0)

    time_ms = np.arange(min_len) / sample_rate * 1000.0
    ax = axes[0, 0]
    for channel_idx, channel_name in enumerate(channel_names[:template.shape[0]]):
        ax.plot(time_ms, template[channel_idx] + 4 * channel_idx, label=channel_name)
    ax.set_title(f"Filter F{filter_id} mean top-window template")
    ax.set_ylabel("z-score + offset")
    ax.set_xlabel("Window time (ms)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)

    freqs, magnitude, peak_freq = spectrum(np.mean(np.stack(windows), axis=0), sample_rate)
    axes[0, 1].plot(freqs, magnitude, color="black")
    axes[0, 1].axvline(peak_freq, color="red", linestyle="--", label=f"peak={peak_freq:.1f} Hz")
    axes[0, 1].set_title("Mean-window spectrum")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Magnitude")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.25)

    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.0,
        1.0,
        "How to read this:\n"
        "• left: recurring raw signal shape\n"
        "• middle: dominant frequency content\n"
        "• rows below: top reference windows\n\n"
        "Use this as the 1D analogue of\n"
        "vision reference patches for a concept.",
        va="top",
        fontsize=10,
    )

    for plot_idx, row in enumerate(rows, start=1):
        window = row["window"]
        t_ms = np.arange(window.shape[-1]) / sample_rate * 1000.0

        ax_signal = axes[plot_idx, 0]
        window_z = zscore_channels(window)
        for channel_idx, channel_name in enumerate(channel_names[:window_z.shape[0]]):
            ax_signal.plot(t_ms, window_z[channel_idx] + 4 * channel_idx, label=channel_name)
        ax_signal.set_title(
            f"sample {row['dataset_idx']} label={row['label']} "
            f"act={row['activation']:.3f} input[{row['input_start']}:{row['input_end']}]"
        )
        ax_signal.set_ylabel("z-score + offset")
        ax_signal.grid(True, alpha=0.25)

        ax_spec = axes[plot_idx, 1]
        ax_spec.plot(row["freqs"], row["spectrum"], color="black")
        ax_spec.axvline(row["peak_freq_hz"], color="red", linestyle="--")
        ax_spec.set_title(f"peak={row['peak_freq_hz']:.1f} Hz")
        ax_spec.set_xlabel("Frequency (Hz)")
        ax_spec.grid(True, alpha=0.25)

        ax_trace = axes[plot_idx, 2]
        trace_time = np.arange(len(row["activation_trace"]))
        ax_trace.plot(trace_time, row["activation_trace"], color="tab:blue")
        ax_trace.axvline(row["activation_idx"], color="red", linestyle="--")
        ax_trace.set_title(f"{layer_name}/F{filter_id} activation trace")
        ax_trace.set_xlabel("Layer time index")
        ax_trace.grid(True, alpha=0.25)

    fig.suptitle(f"{layer_name} filter F{filter_id}: activation references", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / f"{layer_name}_filter_{filter_id}_templates.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_summary(rows_by_filter: Dict[int, List[dict]], layer_name: str, output_dir: Path) -> None:
    """Write machine-readable template metadata for later concept naming."""
    csv_path = output_dir / f"{layer_name}_filter_template_summary.csv"
    fieldnames = [
        "layer",
        "filter_id",
        "rank",
        "dataset_idx",
        "label",
        "activation",
        "activation_idx",
        "input_center",
        "input_start",
        "input_end",
        "peak_freq_hz",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for filter_id, rows in rows_by_filter.items():
            for rank, row in enumerate(rows, start=1):
                writer.writerow({
                    "layer": layer_name,
                    "filter_id": filter_id,
                    "rank": rank,
                    "dataset_idx": row["dataset_idx"],
                    "label": row["label"],
                    "activation": row["activation"],
                    "activation_idx": row["activation_idx"],
                    "input_center": row["input_center"],
                    "input_start": row["input_start"],
                    "input_end": row["input_end"],
                    "peak_freq_hz": row["peak_freq_hz"],
                })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create reference-window galleries for decisive 1D CNN filters."
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Config path for model/data/sample_rate defaults.")
    parser.add_argument("--model", default=None, help="Path to trained CNN1D_Wide checkpoint; defaults to config model.path.")
    parser.add_argument("--data", default=None, help="Dataset directory; defaults to config data.path.")
    parser.add_argument("--concepts", default=None, help="Variant C output directory containing tcd_model.pkl/results.pkl.")
    parser.add_argument("--layer", default="conv3", help="Convolutional layer to inspect, e.g. conv2/conv3/conv4.")
    parser.add_argument("--filters", default=None, help="Optional comma-separated filter IDs, e.g. 3,17,42.")
    parser.add_argument("--prototype-class", type=int, default=None, choices=[0, 1],
                        help="When --concepts is used, select filters from this class GMM only.")
    parser.add_argument("--prototype-id", type=int, default=None,
                        help="When --concepts and --prototype-class are used, select filters from one prototype.")
    parser.add_argument("--n-filters", type=int, default=6,
                        help="Number of decisive filters to auto-select from --concepts.")
    parser.add_argument("--output", default="results/filter_templates", help="Output directory.")
    parser.add_argument("--top-k", type=int, default=8,
                        help="Number of top-activating dataset samples/windows used to create each template.")
    parser.add_argument("--window-size", type=int, default=256, help="Minimum input-window size in samples.")
    parser.add_argument("--sample-rate", type=float, default=None, help="Sample rate in Hz; defaults to config data.sample_rate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for activation scan.")
    parser.add_argument("--class-id", type=int, default=None, choices=[0, 1], help="Optionally restrict to one class.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cpu or cuda.")
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = args.model or get_config_value(config, "model", "path")
    data_path = args.data or get_config_value(config, "data", "path")
    sample_rate = args.sample_rate or float(get_config_value(config, "data", "sample_rate", 400.0))

    if model_path is None:
        raise ValueError("No model path provided. Use --model or set model.path in the config.")
    if data_path is None:
        raise ValueError("No data path provided. Use --data or set data.path in the config.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.filters:
        filter_ids = parse_filter_ids(args.filters)
    elif args.concepts:
        filter_ids = select_filters_from_concepts(
            concepts_dir=args.concepts,
            layer_name=args.layer,
            n_filters=args.n_filters,
            prototype_class=args.prototype_class,
            prototype_id=args.prototype_id,
        )
    else:
        raise ValueError("Provide either --filters or --concepts to choose filters.")

    print(f"Using sample_rate={sample_rate} Hz")
    print(f"Using top_k={args.top_k} top-activating samples/windows per filter")

    model = load_model(model_path, device)
    dataset = VibrationDataset(data_path)

    rows_by_filter = collect_top_filter_windows(
        model=model,
        dataset=dataset,
        layer_name=args.layer,
        filter_ids=filter_ids,
        top_k=args.top_k,
        window_size=args.window_size,
        sample_rate=sample_rate,
        batch_size=args.batch_size,
        device=device,
        class_id=args.class_id,
    )

    for filter_id, rows in rows_by_filter.items():
        plot_filter_gallery(
            filter_id=filter_id,
            layer_name=args.layer,
            rows=rows,
            output_dir=output_dir,
            sample_rate=sample_rate,
        )

    write_summary(rows_by_filter, args.layer, output_dir)
    print(f"Saved filter template galleries and summary to {output_dir}")


if __name__ == "__main__":
    main()
