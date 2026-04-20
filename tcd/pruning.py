"""
LRP-Based Relevance Pruning for CNN1D_Wide.

Uses CRP concept relevance scores to identify and prune non-informative
filters from the CNN1D_Wide model.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import h5py
import copy
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# Layer name → (in_channels, out_channels, kernel_size, stride, padding)
_LAYER_SPECS = {
    'conv1': (3, 16, 25, 1, 12),
    'conv2': (16, 32, 15, 1, 7),
    'conv3': (32, 64, 9, 1, 4),
    'conv4': (64, 128, 5, 1, 2),
}

# Maps conv layer name to the *next* layer name in the model
_NEXT_LAYER = {
    'conv1': 'conv2',
    'conv2': 'conv3',
    'conv3': 'conv4',
    'conv4': None,  # next is fc1 via global_avg_pool
}


class RelevancePruner:
    """
    Prune CNN1D_Wide filters based on CRP relevance importance scores.

    Workflow:
        1. Load pre-computed CRP relevance vectors from HDF5 files.
        2. Compute per-filter importance as mean absolute relevance.
        3. Rank filters by importance (descending).
        4. Prune layers by keeping only the top-k filters.
        5. Optionally run iterative pruning to find the best keep_ratio.
    """

    def __init__(self, features_dir: str):
        """
        Initialize the pruner.

        Args:
            features_dir: Path to directory containing CRP feature HDF5 files
                (``eps_relevances_class_{class_id}.hdf5`` with datasets named
                after layer names).
        """
        self.features_dir = Path(features_dir)
        self.filter_importance: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Relevance loading & importance computation
    # ------------------------------------------------------------------

    def load_relevances(self, layer_name: str) -> np.ndarray:
        """
        Load raw CRP relevance vectors for a given layer.

        Args:
            layer_name: One of ``conv1``, ``conv2``, ``conv3``, ``conv4``.

        Returns:
            Array of shape ``(N_total, n_filters)`` combining both classes.
        """
        rel_list = []
        for class_id in [0, 1]:
            h5_path = self.features_dir / f"eps_relevances_class_{class_id}.hdf5"
            if not h5_path.exists():
                raise FileNotFoundError(
                    f"CRP features not found at {h5_path}. "
                    "Run scripts/run_analysis.py first."
                )
            with h5py.File(h5_path, 'r') as f:
                dataset_key = layer_name  # keys are plain layer names
                if dataset_key not in f:
                    raise KeyError(
                        f"Layer '{layer_name}' not found in {h5_path}. "
                        f"Available: {list(f.keys())}"
                    )
                rel_list.append(np.array(f[dataset_key]))
        return np.concatenate(rel_list, axis=0)

    def compute_filter_importance(self, layer_name: str) -> np.ndarray:
        """
        Compute per-filter importance as mean absolute relevance.

        Args:
            layer_name: Conv layer name.

        Returns:
            Array of shape ``(n_filters,)`` with importance scores.
        """
        relevances = self.load_relevances(layer_name)  # (N, n_filters)
        importance = np.mean(np.abs(relevances), axis=0)
        self.filter_importance[layer_name] = importance
        return importance

    def compute_prototype_importance(
        self,
        concepts_dir: str,
        layer_name: str,
    ) -> np.ndarray:
        """
        Compute per-filter importance from prototype means |μ|.

        Args:
            concepts_dir: Path to Variant C output folder containing ``tcd_model.pkl``.
            layer_name: Layer name the concepts were discovered on.

        Returns:
            Array of shape ``(n_filters,)`` with prototype-based importance.
        """
        tcd_path = Path(concepts_dir) / "tcd_model.pkl"
        results_path = Path(concepts_dir) / "results.pkl"
        if not tcd_path.exists() or not results_path.exists():
            raise FileNotFoundError(
                f"Could not find tcd_model/results in {concepts_dir}. "
                "Run scripts/discover_concepts.py --variant C first."
            )

        with open(results_path, "rb") as f:
            results = pickle.load(f)
        discovered_layer = results.get("layer_name", None)
        if discovered_layer is not None and discovered_layer != layer_name:
            raise ValueError(
                f"Concepts were discovered at layer '{discovered_layer}', "
                f"but pruning layer requested is '{layer_name}'."
            )

        with open(tcd_path, "rb") as f:
            tcd = pickle.load(f)

        gmms = getattr(tcd.prototype_discovery, "gmms", {})
        if not gmms:
            raise ValueError("No per-class GMMs found in tcd_model.pkl")

        means = []
        for class_id, gmm in gmms.items():
            if hasattr(gmm, "means_"):
                means.append(np.abs(gmm.means_))
        if not means:
            raise ValueError("No prototype means found in GMMs")

        # Mean absolute prototype magnitude across all classes/components.
        cat = np.concatenate(means, axis=0)  # (n_total_components, n_filters)
        proto_importance = cat.mean(axis=0)
        return proto_importance

    def compute_merged_importance(
        self,
        layer_name: str,
        concepts_dir: str,
        merge_alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Merge CRP relevance importance with prototype-mean importance.

        merged = alpha * norm(relevance) + (1-alpha) * norm(prototype)

        Args:
            layer_name: Conv layer name.
            concepts_dir: Variant C output directory (for prototype means).
            merge_alpha: Weight for relevance importance in [0, 1].
        """
        if not (0.0 <= merge_alpha <= 1.0):
            raise ValueError("merge_alpha must be in [0, 1]")

        rel = self.compute_filter_importance(layer_name)
        proto = self.compute_prototype_importance(concepts_dir, layer_name)

        # Min-max normalization for stable weighting.
        rel_norm = (rel - rel.min()) / (rel.max() - rel.min() + 1e-10)
        proto_norm = (proto - proto.min()) / (proto.max() - proto.min() + 1e-10)
        merged = merge_alpha * rel_norm + (1.0 - merge_alpha) * proto_norm

        self.filter_importance[layer_name] = merged
        return merged

    def compute_all_layer_importance(self) -> Dict[str, np.ndarray]:
        """
        Compute importance for all four conv layers.

        Returns:
            Dict mapping layer name → importance array.
        """
        for layer_name in ['conv1', 'conv2', 'conv3', 'conv4']:
            self.compute_filter_importance(layer_name)
        return self.filter_importance

    # ------------------------------------------------------------------
    # Single-layer pruning
    # ------------------------------------------------------------------

    def get_keep_indices(self, layer_name: str, keep_ratio: float) -> np.ndarray:
        """
        Return the indices of filters to keep (sorted by original position).

        Args:
            layer_name: Conv layer name.
            keep_ratio: Fraction of filters to keep (0 < keep_ratio ≤ 1).

        Returns:
            Sorted array of filter indices to keep.
        """
        if layer_name not in self.filter_importance:
            self.compute_filter_importance(layer_name)
        importance = self.filter_importance[layer_name]
        n_keep = max(1, int(np.ceil(len(importance) * keep_ratio)))
        top_indices = np.argsort(importance)[::-1][:n_keep]
        return np.sort(top_indices)

    def get_keep_indices_from_importance(
        self, importance: np.ndarray, keep_ratio: float
    ) -> np.ndarray:
        """Return kept filter indices from an externally provided importance vector."""
        n_keep = max(1, int(np.ceil(len(importance) * keep_ratio)))
        top_indices = np.argsort(importance)[::-1][:n_keep]
        return np.sort(top_indices)

    def prune_layer(
        self,
        model: nn.Module,
        layer_name: str,
        keep_ratio: float,
    ) -> nn.Module:
        """
        Create a new model with fewer filters in one conv layer.

        The next layer's input channels are adjusted accordingly, and
        corresponding weight slices are copied from the original model.

        Args:
            model: Original ``CNN1D_Wide`` instance.
            layer_name: Conv layer to prune (``conv1``–``conv4``).
            keep_ratio: Fraction of filters to retain.

        Returns:
            A deep-copied model with the specified layer pruned.
        """
        keep_indices = self.get_keep_indices(layer_name, keep_ratio)
        new_model = copy.deepcopy(model)        # Prune output channels of the target conv layer
        old_conv: nn.Conv1d = getattr(model, layer_name)
        n_keep = len(keep_indices)
        new_conv = nn.Conv1d(
            in_channels=old_conv.in_channels,
            out_channels=n_keep,
            kernel_size=old_conv.kernel_size[0],
            stride=old_conv.stride[0],
            padding=old_conv.padding[0],
        )
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight[keep_indices])
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias[keep_indices])
        setattr(new_model, layer_name, new_conv)

        # Update the next layer's input channels
        next_layer_name = _NEXT_LAYER[layer_name]
        if next_layer_name is not None:
            old_next: nn.Conv1d = getattr(model, next_layer_name)
            new_next = nn.Conv1d(
                in_channels=n_keep,
                out_channels=old_next.out_channels,
                kernel_size=old_next.kernel_size[0],
                stride=old_next.stride[0],
                padding=old_next.padding[0],
            )
            with torch.no_grad():
                new_next.weight.copy_(old_next.weight[:, keep_indices, :])
                if old_next.bias is not None:
                    new_next.bias.copy_(old_next.bias)
            setattr(new_model, next_layer_name, new_next)
        else:
            # conv4 → fc1: fc1 input size equals conv4 out_channels (via global avg pool)
            old_fc1: nn.Linear = model.fc1
            new_fc1 = nn.Linear(n_keep, old_fc1.out_features)
            with torch.no_grad():
                new_fc1.weight.copy_(old_fc1.weight[:, keep_indices])
                new_fc1.bias.copy_(old_fc1.bias)
            new_model.fc1 = new_fc1

        return new_model

    def evaluate_projection_pruning(
        self,
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        layer_name: str,
        importance: np.ndarray,
        keep_ratios: Optional[List[float]] = None,
        device: str = "cpu",
        batch_size: int = 64,
    ) -> List[Dict]:
        """
        Evaluate projection-based concept pruning (no structural weight deletion).

        At inference, project the chosen layer activations onto the subspace spanned
        by the top-k kept filters (implemented as zeroing all dropped channels).
        """
        if keep_ratios is None:
            keep_ratios = [0.9, 0.8, 0.7, 0.5, 0.3]

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        model.eval().to(device)

        baseline = self._evaluate(model, dataset, device=device, batch_size=batch_size)
        results: List[Dict] = []

        # Find layer module once.
        target_module = dict(model.named_modules()).get(layer_name, None)
        if target_module is None:
            raise ValueError(f"Layer {layer_name} not found in model.")

        n_filters = len(importance)

        for ratio in keep_ratios:
            keep_idx = self.get_keep_indices_from_importance(importance, ratio)
            keep_mask = torch.zeros(n_filters, dtype=torch.bool, device=device)
            keep_mask[torch.tensor(keep_idx, dtype=torch.long, device=device)] = True

            def _projection_hook(module, inp, out):
                mod = out.clone()
                mod[:, ~keep_mask, :] = 0.0
                return mod

            handle = target_module.register_forward_hook(_projection_hook)
            correct = total = 0
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    preds = model(x).argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.numel()
            handle.remove()

            acc = correct / max(total, 1)
            results.append(
                {
                    "keep_ratio": ratio,
                    "accuracy": acc,
                    "accuracy_drop": baseline - acc,
                    "n_kept_filters": int(len(keep_idx)),
                    "n_total_filters": int(n_filters),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Iterative pruning
    # ------------------------------------------------------------------

    def iterative_prune(
        self,
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        keep_ratios: Optional[List[float]] = None,
        device: str = 'cpu',
    ) -> List[Dict]:
        """
        Iteratively prune all four conv layers and measure accuracy.

        Args:
            model: Trained ``CNN1D_Wide`` instance.
            dataset: Full dataset for evaluation.
            keep_ratios: Fractions to try (default: [0.9, 0.8, 0.7, 0.5, 0.3]).
            device: Torch device string.

        Returns:
            List of dicts with keys:
            ``keep_ratio``, ``n_params``, ``accuracy``, ``accuracy_drop``.
        """
        if keep_ratios is None:
            keep_ratios = [0.9, 0.8, 0.7, 0.5, 0.3]

        # Precompute all layer importances
        self.compute_all_layer_importance()

        baseline_acc = self._evaluate(model, dataset, device)
        print(f"Baseline accuracy: {baseline_acc:.4f}")

        results = []
        for ratio in keep_ratios:
            pruned = self.prune_all_layers(model, ratio)
            n_params = sum(p.numel() for p in pruned.parameters() if p.requires_grad)
            acc = self._evaluate(pruned, dataset, device)
            drop = baseline_acc - acc
            results.append({
                'keep_ratio': ratio,
                'n_params': n_params,
                'accuracy': acc,
                'accuracy_drop': drop,
            })
            print(
                f"  keep_ratio={ratio:.2f}  "
                f"n_params={n_params:,}  "
                f"accuracy={acc:.4f}  "
                f"drop={drop:+.4f}"
            )

        return results

    def find_knee_point(self, results: List[Dict], threshold: float = 0.01) -> float:
        """
        Find the largest keep_ratio where accuracy drop first exceeds *threshold*.

        Args:
            results: Output of :meth:`iterative_prune`.
            threshold: Accuracy drop threshold (default 1 %).

        Returns:
            The keep_ratio at the knee point, or the last ratio if none found.
        """
        for row in results:
            if row['accuracy_drop'] > threshold:
                return row['keep_ratio']
        return results[-1]['keep_ratio']

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_pruned_model(self, model: nn.Module, path: str) -> None:
        """
        Save a pruned model to disk.

        Args:
            model: Pruned model instance.
            path: File path for the saved checkpoint.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"Pruned model saved to {path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def prune_all_layers(self, model: nn.Module, keep_ratio: float) -> nn.Module:
        """Prune all four conv layers sequentially (public API)."""
        pruned = model
        for layer_name in ['conv1', 'conv2', 'conv3', 'conv4']:
            pruned = self.prune_layer(pruned, layer_name, keep_ratio)
        return pruned

    def _prune_all_layers(self, model: nn.Module, keep_ratio: float) -> nn.Module:
        """Prune all four conv layers sequentially (kept for backward compatibility)."""
        return self.prune_all_layers(model, keep_ratio)

    @staticmethod
    def _evaluate(
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        device: str = 'cpu',
        batch_size: int = 64,
    ) -> float:
        """Evaluate accuracy of *model* on *dataset*."""
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        model.eval().to(device)
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0
