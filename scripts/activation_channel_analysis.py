import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# ============================================================
# ACTIVATION + GRADIENT + FILTER ANALYSIS
# ============================================================

class CNNInspector:

    def __init__(self, model, device=None):

        self.model = model

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model.to(device)

        self.activation_storage = defaultdict(list)
        self.gradient_storage = defaultdict(list)

        self.forward_hooks = []
        self.backward_hooks = []

    # ========================================================
    # HOOKS
    # ========================================================

    def register_hooks(self):

        valid_types = (
            nn.Conv1d,
            nn.Linear,
            nn.LeakyReLU,
            nn.MaxPool1d,
            nn.AdaptiveAvgPool1d
        )



        for name, module in self.model.named_modules():

            if name == "":
                continue

            if not isinstance(module, valid_types):
                continue

            # ---------------- FORWARD HOOK ----------------
            fhook = module.register_forward_hook(
                self.make_forward_hook(name)
            )

            # ---------------- BACKWARD HOOK ----------------
            bhook = module.register_full_backward_hook(
                self.make_backward_hook(name)
            )

            self.forward_hooks.append(fhook)
            self.backward_hooks.append(bhook)

    def remove_hooks(self):

        for h in self.forward_hooks:
            h.remove()

        for h in self.backward_hooks:
            h.remove()

    def make_forward_hook(self, name):

        def hook(module, inp, output):

            if isinstance(output, tuple):
                output = output[0]

            out = output.detach().cpu().float()

            self.activation_storage[name].append(out)

        return hook

    def make_backward_hook(self, name):

        def hook(module, grad_input, grad_output):

            if grad_output[0] is None:
                return

            grad = grad_output[0].detach().cpu().float()

            self.gradient_storage[name].append(grad)

        return hook

    # ========================================================
    # RUN ANALYSIS
    # ========================================================

    def analyze(self, data_loader, n_batches=5):

        self.model.eval()

        self.register_hooks()

        criterion = nn.CrossEntropyLoss()

        with torch.enable_grad():

            for i, (x, y) in enumerate(data_loader):

                if i >= n_batches:
                    break

                x = x.to(self.device)
                y = y.to(self.device)

                self.model.zero_grad()

                out = self.model(x)

                loss = criterion(out, y)

                loss.backward()

        self.remove_hooks()

        return self.compute_statistics()

    # ========================================================
    # COMPUTE STATISTICS
    # ========================================================

    def compute_statistics(self):

        stats = {}

        for layer_name in self.activation_storage.keys():

            acts = torch.cat(
                [a.flatten() for a in self.activation_storage[layer_name]]
            )

            abs_acts = acts.abs()

            # ---------------- CHANNEL ANALYSIS ----------------

            # ---------------- CHANNEL ANALYSIS ----------------

            channel_std = None
            dominant_channels = None

            first_tensor = self.activation_storage[layer_name][0]

            try:

                # ==========================================
                # Conv outputs:
                # [batch, channels, time]
                # ==========================================

                if first_tensor.ndim == 3:

                    per_channel = []

                    for tensor in self.activation_storage[layer_name]:
                        # average over batch + temporal dimension
                        ch_mean = tensor.abs().mean(dim=(0, -1))

                        per_channel.append(ch_mean)

                    per_channel = torch.stack(per_channel).mean(dim=0)

                    channel_std = per_channel.std().item()

                    dominant_channels = (
                        (per_channel > 2 * per_channel.mean())
                        .sum()
                        .item()
                    )

                # ==========================================
                # Linear outputs:
                # [batch, features]
                # ==========================================

                elif first_tensor.ndim == 2:

                    per_channel = []

                    for tensor in self.activation_storage[layer_name]:
                        feat_mean = tensor.abs().mean(dim=0)

                        per_channel.append(feat_mean)

                    per_channel = torch.stack(per_channel).mean(dim=0)

                    channel_std = per_channel.std().item()

                    dominant_channels = (
                        (per_channel > 2 * per_channel.mean())
                        .sum()
                        .item()
                    )

            except Exception as e:

                print(f"Channel analysis skipped for {layer_name}: {e}")


            # ---------------- GRADIENTS ----------------

            if layer_name in self.gradient_storage:

                grads = torch.cat(
                    [g.flatten() for g in self.gradient_storage[layer_name]]
                )

                grad_norm = torch.sqrt((grads ** 2).mean()).item()
                grad_mean = grads.abs().mean().item()

            else:
                grad_norm = 0
                grad_mean = 0

            # ---------------- SPARSITY ----------------

            sparsity = (abs_acts < 1e-5).float().mean().item()

            # ---------------- STATS ----------------

            stats[layer_name] = {

                "mean": acts.mean().item(),
                "std": acts.std().item(),
                "abs_mean": abs_acts.mean().item(),
                "max": abs_acts.max().item(),
                "min": abs_acts.min().item(),

                "l2_norm": torch.sqrt((acts ** 2).mean()).item(),

                "sparsity": sparsity,

                "grad_norm": grad_norm,
                "grad_mean": grad_mean,

                "channel_std": channel_std,
                "dominant_channels": dominant_channels
            }

        return stats


# ============================================================
# REPORT
# ============================================================

def print_detailed_report(stats):

    print("\n" + "=" * 140)

    print(
        f"{'Layer':<25}"
        f"{'AbsMean':>12}"
        f"{'Std':>12}"
        f"{'Max':>12}"
        f"{'Sparsity%':>12}"
        f"{'GradNorm':>14}"
        f"{'GradMean':>14}"
        f"{'ChStd':>12}"
        f"{'DominantCh':>14}"
    )

    print("=" * 140)

    for layer, s in stats.items():

        print(
            f"{layer:<25}"
            f"{s['abs_mean']:>12.4f}"
            f"{s['std']:>12.4f}"
            f"{s['max']:>12.2f}"
            f"{100*s['sparsity']:>11.2f}"
            f"{s['grad_norm']:>14.4f}"
            f"{s['grad_mean']:>14.6f}"
            f"{str(round(s['channel_std'],4)) if s['channel_std'] is not None else '-':>12}"
            f"{str(s['dominant_channels']) if s['dominant_channels'] is not None else '-':>14}"
        )


# ============================================================
# ACTIVATION HISTOGRAMS
# ============================================================

def plot_activation_histograms(inspector, max_layers=6):

    layers = list(inspector.activation_storage.keys())[:max_layers]

    fig, axes = plt.subplots(len(layers), 1, figsize=(10, 3 * len(layers)))

    if len(layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):

        vals = torch.cat([
            a.flatten()
            for a in inspector.activation_storage[layer]
        ]).numpy()

        ax.hist(vals, bins=100)

        ax.set_title(f"{layer} Activation Distribution")

        ax.set_yscale("log")

    plt.tight_layout()
    plt.show()


# ============================================================
# GRADIENT FLOW
# ============================================================

def plot_gradient_flow(stats):

    layers = list(stats.keys())

    grad_norms = [stats[l]["grad_norm"] for l in layers]

    plt.figure(figsize=(12, 5))

    plt.plot(grad_norms, marker="o")

    plt.yscale("log")

    plt.xticks(
        range(len(layers)),
        layers,
        rotation=45
    )

    plt.ylabel("Gradient Norm")

    plt.title("Gradient Flow Through Layers")

    plt.grid(True)

    plt.tight_layout()

    plt.show()


# ============================================================
# FILTER FREQUENCY ANALYSIS
# ============================================================

def analyze_conv_filters(model):

    conv_layers = []

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv1d):

            conv_layers.append((name, module))

    for name, conv in conv_layers:

        weights = conv.weight.detach().cpu().numpy()

        print(f"\n{name} FILTER FREQUENCY ANALYSIS")

        plt.figure(figsize=(12, 4))

        for i in range(min(8, weights.shape[0])):

            kernel = weights[i, 0]

            fft_mag = np.abs(np.fft.rfft(kernel))

            freqs = np.fft.rfftfreq(len(kernel))

            plt.plot(freqs, fft_mag)

        plt.title(f"{name} Kernel Frequency Responses")

        plt.xlabel("Normalized Frequency")

        plt.ylabel("Magnitude")

        plt.grid(True)

        plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    from Classification.cnn1D_model import CNN1D_Wide
    from utils.dataloader import stratified_group_split


    # ========================================================
    # LOAD DATA
    # ========================================================

    data_dir = "../data/final/new_selection/less_bad/normalized_windowed_downsampled_data_lessBAD"

    train_loader, val_loader, test_loader, dataset = \
        stratified_group_split(data_dir)


    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = CNN1D_Wide()

    model.load_state_dict(
        torch.load(
            "../cnn1d_model_new.ckpt",
            map_location="cpu"
        )
    )


    # ========================================================
    # ANALYSIS
    # ========================================================

    inspector = CNNInspector(model)

    stats = inspector.analyze(
        val_loader,
        n_batches=10
    )


    # ========================================================
    # REPORT
    # ========================================================

    print_detailed_report(stats)


    # ========================================================
    # PLOTS
    # ========================================================

    plot_activation_histograms(inspector)

    plot_gradient_flow(stats)

    analyze_conv_filters(model)

