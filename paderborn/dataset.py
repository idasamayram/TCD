"""PyTorch Dataset for the windowed, pre-split Paderborn bearing data
produced by preprocessing.py.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PUBearingWindows(Dataset):
    """Loads one of train.npz / val.npz / test.npz produced by preprocessing.py.

    X is stored as (N, 1, window_size) float32, y as (N,) int64.
    """

    def __init__(self, npz_path):
        npz_path = Path(npz_path)
        data = np.load(npz_path, allow_pickle=True)
        self.X = data["X"]
        self.y = data["y"]
        self.bearing = data["bearing"] if "bearing" in data else None
        self.condition = data["condition"] if "condition" in data else None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def load_meta(processed_dir):
    with open(Path(processed_dir) / "meta.json") as f:
        return json.load(f)
