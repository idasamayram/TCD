import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def measure_activation_growth(model, data_loader, device=None, n_batches=5):
    """
    Measure activation statistics across all layers to detect vanishing/exploding activations.

    Args:
        model: PyTorch model (CNN1D_DS_Wide, CNN1D_Wide, EnhancedTCN, etc.)
        data_loader: DataLoader to sample inputs from
        device: torch device
        n_batches: number of batches to average over

    Returns:
        stats: dict of layer_name -> {mean, std, max, min, dead_ratio}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    activation_stats = defaultdict(lambda: {"values": []})
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            # output can be a tensor or tuple
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output

            out_np = out.detach().cpu().float().numpy()
            activation_stats[name]["values"].append(out_np)

        return hook_fn

    # Register hooks on all named modules that produce activations
    skip_types = (torch.nn.Dropout, torch.nn.Flatten, torch.nn.Sequential)
    for name, module in model.named_modules():
        if name == "":
            continue  # skip root
        if isinstance(module, skip_types):
            continue
        hook = module.register_forward_hook(make_hook(name))
        hooks.append(hook)

    # Run forward passes
    with torch.no_grad():
        for i, (inputs, _) in enumerate(data_loader):
            if i >= n_batches:
                break
            inputs = inputs.to(device)
            model(inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute statistics
    stats = {}
    for name, data in activation_stats.items():
        all_vals = np.concatenate([v.flatten() for v in data["values"]])
        abs_vals = np.abs(all_vals)

        stats[name] = {
            "mean": float(np.mean(all_vals)),
            "std": float(np.std(all_vals)),
            "abs_mean": float(np.mean(abs_vals)),
            "max": float(np.max(abs_vals)),
            "min": float(np.min(abs_vals)),
            # Fraction of activations that are effectively zero (dead neurons)
            "dead_ratio": float(np.mean(abs_vals < 1e-6)),
            "l2_norm": float(np.sqrt(np.mean(all_vals ** 2))),
        }

    return stats


def plot_activation_growth(stats, title="Activation Growth per Layer", figsize=(16, 10)):
    """
    Plot activation statistics per layer to visually detect vanishing/exploding activations.
    """
    layer_names = list(stats.keys())
    abs_means = [stats[l]["abs_mean"] for l in layer_names]
    stds = [stats[l]["std"] for l in layer_names]
    maxes = [stats[l]["max"] for l in layer_names]
    dead_ratios = [stats[l]["dead_ratio"] for l in layer_names]
    l2_norms = [stats[l]["l2_norm"] for l in layer_names]

    x = np.arange(len(layer_names))
    short_names = [n.split(".")[-1] + f"\n({n.split('.')[0]})" if "." in n else n
                   for n in layer_names]

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # --- Plot 1: Mean absolute activation + std ---
    axes[0].bar(x, abs_means, alpha=0.7, label="Mean |activation|", color="#3498db")
    axes[0].errorbar(x, abs_means, yerr=stds, fmt="none", color="black",
                     capsize=3, linewidth=1, label="±1 std")
    axes[0].plot(x, maxes, "r^", markersize=5, label="Max |activation|", alpha=0.7)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Activation magnitude (log scale)")
    axes[0].set_title(f"{title} — Mean |activation| ± std")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, fontsize=7, rotation=45, ha="right")
    axes[0].legend(fontsize=9)
    axes[0].axhline(y=1e-4, color="orange", linestyle="--", linewidth=1,
                    label="Vanishing threshold (1e-4)")
    axes[0].axhline(y=1e4, color="red", linestyle="--", linewidth=1,
                    label="Exploding threshold (1e4)")
    axes[0].grid(True, alpha=0.3, axis="y")

    # --- Plot 2: L2 norm per layer ---
    axes[1].plot(x, l2_norms, "o-", color="#2ecc71", linewidth=2, markersize=5)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("L2 norm (log scale)")
    axes[1].set_title("L2 Norm per Layer")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, fontsize=7, rotation=45, ha="right")
    axes[1].axhline(y=1e-4, color="orange", linestyle="--", linewidth=1)
    axes[1].axhline(y=1e4, color="red", linestyle="--", linewidth=1)
    axes[1].grid(True, alpha=0.3)

    # --- Plot 3: Dead neuron ratio ---
    axes[2].bar(x, dead_ratios, color="#e74c3c", alpha=0.7)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Dead neuron ratio (|act| < 1e-6)")
    axes[2].set_title("Dead Neurons per Layer")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(short_names, fontsize=7, rotation=45, ha="right")
    axes[2].axhline(y=0.5, color="red", linestyle="--", linewidth=1,
                    label="50% dead threshold")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()
    return fig


def print_activation_report(stats):
    """Print a summary table flagging problematic layers."""
    print(f"\n{'Layer':<40} {'AbsMean':>10} {'Std':>10} {'Max':>10} "
          f"{'L2Norm':>10} {'Dead%':>8} {'Status':>12}")
    print("-" * 105)

    for name, s in stats.items():
        abs_mean = s["abs_mean"]
        dead_pct = s["dead_ratio"] * 100

        if abs_mean < 1e-4:
            status = "⚠ VANISHING"
        elif abs_mean > 1e4:
            status = "⚠ EXPLODING"
        elif dead_pct > 50:
            status = "⚠ DEAD"
        else:
            status = "✓ OK"

        print(f"{name:<40} {abs_mean:>10.4f} {s['std']:>10.4f} {s['max']:>10.2f} "
              f"{s['l2_norm']:>10.4f} {dead_pct:>7.1f}% {status:>12}")



if __name__ == "__main__":

    from Classification.cnn1D_model import CNN1D_Wide
    from utils.dataloader import stratified_group_split

    # Load data and model
    data_dir = "../data/final/new_selection/less_bad/normalized_windowed_downsampled_data_lessBAD"
    train_loader, val_loader, test_loader, dataset = stratified_group_split(data_dir)

    model = CNN1D_Wide()
    model.load_state_dict(torch.load("../cnn1d_model_new.ckpt", map_location="cpu"))

    # Measure on validation set (avoids training noise)
    stats = measure_activation_growth(model, val_loader, n_batches=10)

    # Report + plot
    print_activation_report(stats)
    fig = plot_activation_growth(stats, title="CNN1D_Wide Activation Growth")