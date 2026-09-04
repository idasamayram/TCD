"""Minimal example of computing LRP relevance for the trained CNN1D model
using zennit. No canonizer is needed since the model has no BatchNorm.

    pip install zennit
    python explain_lrp.py --run-dir ./runs/exp1 --data-dir ./processed --index 0

This will load one test-set window, run LRP with the EpsilonPlusFlat
composite (Epsilon rule for the Linear layers, ZPlus for the Conv1d layers,
Flat rule for the very first layer - Flat is the right choice here because,
unlike images, our z-scored vibration input isn't bounded in [0, 1], so the
box-constrained rules used for images don't apply), and plot the relevance
overlaid on the raw signal.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import PUBearingWindows, load_meta
from model import CNN1D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True, help="Dir with best_model.pt + run_config.json")
    parser.add_argument("--data-dir", type=str, required=True, help="Dir with test.npz + meta.json")
    parser.add_argument("--index", type=int, default=0, help="Index into the test set to explain.")
    parser.add_argument("--target", type=int, default=None,
                         help="Class index to explain (default: the model's predicted class).")
    parser.add_argument("--out", type=str, default="relevance.png")
    args = parser.parse_args()

    from zennit.composites import EpsilonPlusFlat
    from zennit.attribution import Gradient

    run_dir = Path(args.run_dir)
    with open(run_dir / "run_config.json") as f:
        run_config = json.load(f)
    class_names = run_config["class_names"]

    model = CNN1D(in_channels=1, num_classes=len(class_names),
                   base_channels=run_config.get("base_channels", 16),
                   dropout=run_config.get("dropout", 0.3))
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location="cpu"))
    model.eval()

    test_ds = PUBearingWindows(Path(args.data_dir) / "test.npz")
    x, y_true = test_ds[args.index]
    x = x.unsqueeze(0)          # (1, 1, window_size)
    x.requires_grad_(True)

    with torch.no_grad():
        logits = model(x)
        pred_class = int(logits.argmax(dim=1).item())

    target_class = args.target if args.target is not None else pred_class

    composite = EpsilonPlusFlat()
    one_hot = torch.eye(len(class_names))[[target_class]]
    with Gradient(model=model, composite=composite) as attributor:
        output, relevance = attributor(x, one_hot)

    relevance = relevance.squeeze().detach().numpy()   # (window_size,)
    signal = x.squeeze().detach().numpy()

    print(f"Sample index {args.index}: true={class_names[int(y_true)]}, "
          f"predicted={class_names[pred_class]}, explained class={class_names[target_class]}")
    print(f"Relevance sum: {relevance.sum():.4f} (logit for that class: {logits[0, target_class].item():.4f})")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        axes[0].plot(signal, linewidth=0.7)
        axes[0].set_ylabel("signal")
        axes[1].plot(relevance, linewidth=0.7, color="crimson")
        axes[1].set_ylabel("LRP relevance")
        axes[1].set_xlabel("sample")
        fig.suptitle(f"true={class_names[int(y_true)]} | pred={class_names[pred_class]} | "
                     f"explained={class_names[target_class]}")
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        print(f"Saved plot to {args.out}")
    except ImportError:
        np.save(args.out.replace(".png", ".npy"), relevance)
        print("matplotlib not available, saved raw relevance array instead.")


if __name__ == "__main__":
    main()
