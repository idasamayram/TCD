"""
Output Layer Analysis for CNN1D_Wide.

Analyzes the model's final ``nn.Linear(64, 2)`` layer (``fc2``) to understand
the decision geometry learned by the network.
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score


class OutputAnalyzer:
    """
    Analyze the output geometry of CNN1D_Wide's ``fc2`` layer.

    The final layer ``fc2: nn.Linear(64, 2)`` has weight matrix
    ``W ∈ R^{2×64}``; rows ``w0`` and ``w1`` define the two class directions.
    """

    # ------------------------------------------------------------------
    # Weight geometry
    # ------------------------------------------------------------------

    def analyze_weights(self, model: nn.Module) -> dict:
        """
        Compute geometric statistics of the output weight vectors.

        Args:
            model: CNN1D_Wide instance.

        Returns:
            Dict with keys:
            ``w0``, ``w1``, ``cosine_similarity``, ``angle_degrees``,
            ``norm_w0``, ``norm_w1``, ``decision_direction``,
            ``single_neuron_sufficient``.
        """
        fc2: nn.Linear = model.fc2
        W = fc2.weight.detach().cpu()   # (2, 64)
        w0 = W[0]
        w1 = W[1]

        cos_sim = float(
            torch.nn.functional.cosine_similarity(w0.unsqueeze(0), w1.unsqueeze(0)).item()
        )
        angle_deg = float(np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0))))
        norm_w0 = float(w0.norm().item())
        norm_w1 = float(w1.norm().item())
        decision_dir = w0 - w1  # direction that drives binary decision

        return {
            'w0': w0.numpy(),
            'w1': w1.numpy(),
            'cosine_similarity': cos_sim,
            'angle_degrees': angle_deg,
            'norm_w0': norm_w0,
            'norm_w1': norm_w1,
            'decision_direction': decision_dir.numpy(),
            # If cos_sim < -0.9 the two weight vectors are nearly anti-parallel,
            # meaning a single neuron with weight (w0 - w1) could replace fc2.
            'single_neuron_sufficient': cos_sim < -0.9,
        }

    # ------------------------------------------------------------------
    # Filter importance via output gradient
    # ------------------------------------------------------------------

    def analyze_filter_importance_via_output(
        self,
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        layer_names: Optional[list] = None,
        n_batches: int = 10,
        batch_size: int = 32,
        device: str = 'cpu',
    ) -> dict:
        """
        Estimate per-filter decision importance via gradient of (out0 − out1).

        For each conv layer, computes the mean absolute gradient of the scalar
        ``(logit_0 − logit_1)`` with respect to the layer's feature map,
        then averages over the spatial dimension to get per-filter importance.

        Args:
            model: CNN1D_Wide instance.
            dataset: Dataset to sample batches from.
            layer_names: Conv layers to probe (default: all four).
            n_batches: Number of mini-batches to average over.
            batch_size: Batch size for gradient computation.
            device: Torch device.

        Returns:
            Dict mapping layer name → 1-D numpy array of importance scores.
        """
        if layer_names is None:
            layer_names = ['conv1', 'conv2', 'conv3', 'conv4']

        model.eval().to(device)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        # Accumulators: layer → list of (n_filters,) arrays
        accum: dict = {ln: [] for ln in layer_names}

        for batch_idx, (x, _) in enumerate(loader):
            if batch_idx >= n_batches:
                break
            x = x.to(device)

            # Register forward hooks to capture activations
            activations: dict = {}
            hooks = []
            for ln in layer_names:
                layer = getattr(model, ln)

                def _make_hook(name):
                    def _hook(module, inp, out):
                        out.retain_grad()
                        activations[name] = out
                    return _hook

                hooks.append(layer.register_forward_hook(_make_hook(ln)))

            out = model(x)          # (B, 2)
            score = (out[:, 0] - out[:, 1]).mean()
            model.zero_grad()
            score.backward()

            for ln in layer_names:
                act = activations.get(ln)
                if act is not None and act.grad is not None:
                    # Average over batch (0) and spatial/time dimension (2)
                    # to get per-filter importance independent of sequence length
                    importance = act.grad.abs().mean(dim=(0, 2))  # (n_filters,)
                    accum[ln].append(importance.detach().cpu().numpy())

            for h in hooks:
                h.remove()

        result = {}
        for ln in layer_names:
            if accum[ln]:
                result[ln] = np.mean(np.stack(accum[ln]), axis=0)
            else:
                result[ln] = np.array([])
        return result

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_output_geometry(
        self,
        model: nn.Module,
        output_dir: str,
        dataset: Optional[torch.utils.data.Dataset] = None,
        device: str = 'cpu',
    ) -> None:
        """
        Generate and save output geometry plots.

        Files written to *output_dir*:
        - ``output_weights.png``   — bar chart of w0 / w1 magnitudes
        - ``output_weight_scatter.png`` — w0 vs w1 scatter
        - ``output_space.png``     — test samples projected into output space

        Args:
            model: CNN1D_Wide instance.
            output_dir: Directory where plots are saved.
            dataset: Optional dataset for sample projection.
            device: Torch device.
        """
        os.makedirs(output_dir, exist_ok=True)
        info = self.analyze_weights(model)
        w0, w1 = info['w0'], info['w1']
        dims = np.arange(len(w0))

        # --- bar chart of w0/w1 magnitudes ---
        fig, ax = plt.subplots(figsize=(max(8, len(w0) // 4), 4))
        width = 0.4
        ax.bar(dims - width / 2, w0, width=width, label='w0 (OK)', alpha=0.75)
        ax.bar(dims + width / 2, w1, width=width, label='w1 (NOK)', alpha=0.75)
        ax.set_xlabel('Hidden dimension')
        ax.set_ylabel('Weight value')
        ax.set_title(
            f'Output Weights  |  cos_sim={info["cosine_similarity"]:.3f}'
            f'  angle={info["angle_degrees"]:.1f}°'
        )
        ax.legend()
        plt.tight_layout()
        path = os.path.join(output_dir, 'output_weights.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {path}")

        # --- scatter w0 vs w1 ---
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(w0, w1, alpha=0.7, s=30)
        lim = max(np.abs(w0).max(), np.abs(w1).max()) * 1.1
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=0.8, label='y=x')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel('w0 (OK class)')
        ax.set_ylabel('w1 (NOK class)')
        ax.set_title('Output Weight Scatter')
        ax.legend()
        plt.tight_layout()
        path = os.path.join(output_dir, 'output_weight_scatter.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {path}")

        # --- project samples into 2D output space ---
        if dataset is not None:
            self._plot_output_space(model, dataset, output_dir, device)

    def plot_decision_axis_analysis(
        self,
        model: nn.Module,
        output_dir: str,
        dataset: Optional[torch.utils.data.Dataset] = None,
        device: str = 'cpu',
        top_k: int = 20,
    ) -> None:
        """
        Analyze the single decision axis w_decision = w0 - w1.

        Since w0 ⊥ w1 the binary classification collapses to the scalar
        ``score = w_decision · h + b_decision``.  This method:

        - Projects all hidden activations onto the decision axis
        - Plots the 1D score distribution for OK vs NOK
        - Computes Fisher discriminant ratio, ROC-AUC and overlap
        - Shows the top-k filter weights on the decision axis
        - Saves a multi-panel figure to ``output_dir/decision_axis_analysis.png``

        Args:
            model: CNN1D_Wide instance.
            output_dir: Directory where the figure is saved.
            dataset: Dataset used to collect hidden activations.
            device: Torch device.
            top_k: Number of top decision-axis components to visualise.
        """
        os.makedirs(output_dir, exist_ok=True)
        info = self.analyze_weights(model)
        w0 = info['w0']
        w1 = info['w1']
        w_dec = w0 - w1  # (hidden_dim,)

        fc2: nn.Linear = model.fc2
        b0, b1 = 0.0, 0.0
        if fc2.bias is not None:
            b0 = float(fc2.bias.detach().cpu()[0].item())
            b1 = float(fc2.bias.detach().cpu()[1].item())
        b_dec = b0 - b1

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # ---- (a) 1D score distribution ----
        ax_a = axes[0]
        if dataset is not None:
            scores_ok, scores_nok = self._collect_decision_scores(
                model, dataset, w_dec, b_dec, device
            )
            all_scores = np.concatenate([scores_ok, scores_nok])
            all_labels_bin = np.concatenate([
                np.zeros(len(scores_ok)), np.ones(len(scores_nok))
            ])

            # KDE curves
            x_range = np.linspace(all_scores.min() - 1, all_scores.max() + 1, 500)
            for scores, color, lbl in [
                (scores_ok, 'steelblue', f'OK (n={len(scores_ok)})'),
                (scores_nok, 'tomato', f'NOK (n={len(scores_nok)})'),
            ]:
                if len(scores) > 1 and scores.std() > 0:
                    kde = gaussian_kde(scores)
                    ax_a.fill_between(x_range, kde(x_range), alpha=0.4, color=color)
                    ax_a.plot(x_range, kde(x_range), color=color, label=lbl)
                ax_a.axvline(scores.mean(), color=color, linestyle='--', linewidth=1)

            ax_a.axvline(0, color='k', linestyle='-', linewidth=1.2, label='threshold (0)')

            # Metrics
            mu_ok, mu_nok = scores_ok.mean(), scores_nok.mean()
            s_ok, s_nok = scores_ok.std(), scores_nok.std()
            fisher = float((mu_ok - mu_nok) ** 2 / (s_ok ** 2 + s_nok ** 2 + 1e-8))
            try:
                auc = float(roc_auc_score(all_labels_bin, -all_scores))
            except Exception:
                auc = float('nan')
            ax_a.set_title(
                f'Decision Score Distribution\nFisher={fisher:.2f}  AUC={auc:.3f}'
            )
        else:
            ax_a.set_title('Decision Score Distribution\n(no dataset)')
        ax_a.set_xlabel('score = w_dec · h + b_dec')
        ax_a.set_ylabel('density')
        ax_a.legend(fontsize=8)

        # ---- (b) Top-k decision axis filter weights ----
        ax_b = axes[1]
        k = min(top_k, len(w_dec))
        top_idx = np.argsort(np.abs(w_dec))[::-1][:k]
        top_idx_sorted = top_idx[np.argsort(top_idx)]
        ax_b.bar(
            np.arange(len(top_idx_sorted)),
            w_dec[top_idx_sorted],
            color=['tomato' if v > 0 else 'steelblue' for v in w_dec[top_idx_sorted]],
            alpha=0.8,
        )
        ax_b.set_xticks(np.arange(len(top_idx_sorted)))
        ax_b.set_xticklabels(top_idx_sorted, rotation=90, fontsize=7)
        ax_b.axhline(0, color='k', linewidth=0.5)
        ax_b.set_xlabel('Filter index')
        ax_b.set_ylabel('w_dec weight')
        ax_b.set_title(f'Top-{k} Decision Axis Filters\n(red=OK↑, blue=NOK↑)')

        # ---- (c) Fixed output space scatter ----
        ax_c = axes[2]
        if dataset is not None:
            all_logits, all_lbls = self._collect_logits(model, dataset, device)
            colors_map = ['steelblue', 'tomato']
            label_names = ['OK', 'NOK']
            counts = [int((all_lbls == c).sum()) for c in [0, 1]]
            for cls, (alpha, s) in zip([0, 1], [(0.3, 20), (1.0, 60)]):
                mask = all_lbls == cls
                ax_c.scatter(
                    all_logits[mask, 0],
                    all_logits[mask, 1],
                    label=f'{label_names[cls]} (n={counts[cls]})',
                    color=colors_map[cls],
                    alpha=alpha,
                    s=s,
                    zorder=cls + 1,
                )
            lims = [
                min(all_logits[:, 0].min(), all_logits[:, 1].min()) - 1,
                max(all_logits[:, 0].max(), all_logits[:, 1].max()) + 1,
            ]
            ax_c.plot(lims, lims, 'k--', linewidth=1.0, label='decision boundary', zorder=0)
            ax_c.set_xlim(lims)
            ax_c.set_ylim(lims)
        ax_c.set_xlabel('Logit 0 (OK)')
        ax_c.set_ylabel('Logit 1 (NOK)')
        ax_c.set_title('Output Space (logits)\nNOK plotted on top')
        ax_c.legend(fontsize=8)

        plt.tight_layout()
        path = os.path.join(output_dir, 'decision_axis_analysis.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _plot_output_space(
        self,
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        output_dir: str,
        device: str = 'cpu',
        max_samples: int = 10000,
    ) -> None:
        """Project samples to logit space and save scatter plot."""
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=False, num_workers=0
        )
        model.eval().to(device)

        all_logits, all_labels = [], []
        collected = 0
        with torch.no_grad():
            for x, y in loader:
                if collected >= max_samples:
                    break
                x = x.to(device)
                logits = model(x).cpu().numpy()
                all_logits.append(logits)
                all_labels.append(y.numpy())
                collected += len(x)

        all_logits = np.concatenate(all_logits)[:max_samples]
        all_labels = np.concatenate(all_labels)[:max_samples]

        colors = ['steelblue', 'tomato']
        label_names = ['OK', 'NOK']
        counts = [int((all_labels == cls).sum()) for cls in [0, 1]]
        fig, ax = plt.subplots(figsize=(6, 5))
        # Plot OK first (background), then NOK on top so minority class is visible
        for cls, (alpha, s) in zip([0, 1], [(0.3, 20), (1.0, 60)]):
            mask = all_labels == cls
            ax.scatter(
                all_logits[mask, 0],
                all_logits[mask, 1],
                label=f'{label_names[cls]} (n={counts[cls]})',
                color=colors[cls],
                alpha=alpha,
                s=s,
                zorder=cls + 1,
            )
        # Draw decision boundary: Logit_OK = Logit_NOK diagonal
        lims = [
            min(all_logits[:, 0].min(), all_logits[:, 1].min()) - 1,
            max(all_logits[:, 0].max(), all_logits[:, 1].max()) + 1,
        ]
        ax.plot(lims, lims, 'k--', linewidth=1.0, label='decision boundary', zorder=0)
        ax.set_xlabel('Logit 0 (OK)')
        ax.set_ylabel('Logit 1 (NOK)')
        ax.set_title('Output Space (logits)')
        ax.legend()
        plt.tight_layout()
        path = os.path.join(output_dir, 'output_space.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {path}")

    def _collect_logits(
        self,
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        device: str = 'cpu',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect all model logits and labels from dataset."""
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=False, num_workers=0
        )
        model.eval().to(device)
        all_logits, all_labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                all_logits.append(model(x).cpu().numpy())
                all_labels.append(y.numpy())
        return np.concatenate(all_logits), np.concatenate(all_labels)

    def _collect_decision_scores(
        self,
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        w_dec: np.ndarray,
        b_dec: float,
        device: str = 'cpu',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project fc2 hidden activations onto the decision axis.

        Returns arrays of scores for OK and NOK samples separately.
        """
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=False, num_workers=0
        )
        model.eval().to(device)
        w_t = torch.tensor(w_dec, dtype=torch.float32, device=device)

        scores_ok, scores_nok = [], []
        activations: dict = {}

        def _hook(module, inp, out):
            activations['h'] = inp[0].detach()

        handle = model.fc2.register_forward_hook(_hook)
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                model(x)
                h = activations['h']              # (B, hidden_dim)
                s = (h @ w_t + b_dec).cpu().numpy()
                labels = y.numpy()
                scores_ok.append(s[labels == 0])
                scores_nok.append(s[labels == 1])
        handle.remove()

        return np.concatenate(scores_ok), np.concatenate(scores_nok)
