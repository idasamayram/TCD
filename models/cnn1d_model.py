"""
CNN1D_Wide — CNC thesis architecture with flat attributes for trained weight compatibility.

CRITICAL: This uses flat attributes (conv1, conv2, etc.) NOT nn.Sequential.
The trained weights from CNC repo use parameter names like 'conv1.weight', 'conv2.weight'.
DO NOT restructure to nn.Sequential - this would break weight loading.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List
import h5py
import os
from pathlib import Path


class CNN1D_Wide(nn.Module):
    """
    CNN1D_Wide architecture from CNC thesis for vibration fault classification.
    
    Input: (batch, 3, 2000) - 3 accelerometer axes, 2000 timesteps (5s @ 400Hz)
    Output: (batch, 2) - binary classification logits (OK vs NOK)
    
    Architecture uses flat attributes (conv1, conv2, etc.) to match trained weights.
    4 convolutional layers with progressively wider kernels for multi-scale feature extraction.
    No BatchNorm - intentional for clean LRP gradient flow.
    """
    
    def __init__(self):
        super(CNN1D_Wide, self).__init__()
        # Wider kernels to increase receptive field
        self.conv1 = nn.Conv1d(3, 16, kernel_size=25, stride=1, padding=12)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=15, stride=1, padding=7)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout3 = nn.Dropout(0.2)

        self.conv4 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(0.2)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)  # Binary classification

        self.dropout = nn.Dropout(0.4)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.dropout1(self.pool1(self.relu(self.conv1(x))))
        x = self.dropout2(self.pool2(self.relu(self.conv2(x))))
        x = self.dropout3(self.pool3(self.relu(self.conv3(x))))
        x = self.dropout4(self.pool4(self.relu(self.conv4(x))))

        x = self.global_avg_pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def get_layer_names() -> List[str]:
    """
    Return conv layer names for CRP hooks.
    
    These correspond to the flat attribute names in CNN1D_Wide.
    Use these names when calling record_layer in CRP attribution.
    
    Returns:
        List of layer names: ["conv1", "conv2", "conv3", "conv4"]
    """
    return ["conv1", "conv2", "conv3", "conv4"]


class VibrationDataset(Dataset):
    """
    Dataset for CNC vibration signals stored as HDF5 files.
    
    Data structure:
    - Root directory contains 'good/' and 'bad/' subdirectories
    - Each subdirectory contains individual .h5 files
    - Each .h5 file has a 'vibration' dataset of shape (3, 2000)
    - good/ → label 0 (OK), bad/ → label 1 (NOK)
    
    Expected counts:
    - ~5606 OK samples (label=0)
    - ~777 NOK samples (label=1)
    
    Args:
        root_dir: Path to directory containing 'good/' and 'bad/' subdirectories
        split: 'train', 'val', or 'test' (currently loads all, TODO: implement splits)
        transform: Optional transform to apply to samples
    """
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Load file paths and labels
        self.samples = []
        
        # Load OK samples (label=0) from good/ subdirectory
        good_dir = self.root_dir / 'good'
        if good_dir.exists():
            for h5_file in good_dir.glob('*.h5'):
                self.samples.append((str(h5_file), 0))
        
        # Load NOK samples (label=1) from bad/ subdirectory
        bad_dir = self.root_dir / 'bad'
        if bad_dir.exists():
            for h5_file in bad_dir.glob('*.h5'):
                self.samples.append((str(h5_file), 1))
        
        if len(self.samples) == 0:
            raise ValueError(f"No .h5 files found in {root_dir}/good or {root_dir}/bad")
        
        # Count samples per class
        ok_count = sum(1 for _, label in self.samples if label == 0)
        nok_count = sum(1 for _, label in self.samples if label == 1)
        print(f"Loaded {len(self.samples)} samples: {ok_count} OK, {nok_count} NOK")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Load and return a sample.
        
        Returns:
            signal: Tensor of shape (3, 2000) - tri-axial vibration
            label: 0 (OK) or 1 (NOK)
        """
        h5_path, label = self.samples[idx]
        
        # Load vibration signal from HDF5
        with h5py.File(h5_path, 'r') as f:
            signal = f['vibration'][:]  # Shape: (3, 2000)
        
        # Convert to tensor
        signal = torch.from_numpy(signal).float()
        
        # Apply transform if provided
        if self.transform is not None:
            signal = self.transform(signal)
        
        return signal, label