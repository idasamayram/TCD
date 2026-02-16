"""
CNN1D Model Architecture and VibrationDataset for 1D time-series vibration data.

Adapted from idasamayram/CNC repository for binary fault detection:
- Input: (batch, 3, 2000) — 3 accelerometer axes (X, Y, Z), 2000 timesteps at 400Hz
- NO BatchNorm/GroupNorm — intentional for clean LRP gradient flow
- LeakyReLU activations
- Binary classification: OK (0) vs NOK (1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List


class CNN1D_Wide(nn.Module):
    """
    Wide 1D CNN for vibration signal classification.
    
    Architecture:
    - 3 Conv1d blocks with increasing channels (16→32→64)
    - LeakyReLU activations (no BatchNorm for clean LRP)
    - Global average pooling
    - 2-layer FC classifier
    
    Input shape: (B, 3, 2000) where 3 = X/Y/Z accelerometer channels
    Output shape: (B, 2) logits for binary classification
    """
    
    def __init__(self, num_classes: int = 2, num_channels: int = 3):
        super(CNN1D_Wide, self).__init__()
        
        # Conv block 1: 3 -> 16 channels
        self.conv1 = nn.Conv1d(num_channels, 16, kernel_size=7, padding=3)
        self.relu1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool1d(2)
        
        # Conv block 2: 16 -> 32 channels
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.relu2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool1d(2)
        
        # Conv block 3: 32 -> 64 channels
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.LeakyReLU(0.1)
        self.pool3 = nn.MaxPool1d(2)
        
        # Global average pooling is done in forward()
        # FC layers
        self.fc1 = nn.Linear(64, 32)
        self.relu4 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(32, num_classes)
        
        # Sequential feature extractor for easier layer access
        self.features = nn.Sequential(
            self.conv1, self.relu1, self.pool1,      # features.0, 1, 2
            self.conv2, self.relu2, self.pool2,      # features.3, 4, 5
            self.conv3, self.relu3, self.pool3,      # features.6, 7, 8
        )
        
        self.classifier = nn.Sequential(
            self.fc1, self.relu4, self.fc2
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, 2000)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        x = self.features(x)
        # Global average pooling over time dimension
        x = x.mean(dim=-1)  # (B, 64)
        x = self.classifier(x)
        return x


class VibrationDataset(Dataset):
    """
    Dataset for vibration time-series data stored in HDF5 format.
    
    Expected directory structure:
        data_path/
            good/
                sample_001.h5
                sample_002.h5
                ...
            bad/
                sample_001.h5
                sample_002.h5
                ...
    
    Each .h5 file contains a dataset 'vibration' with shape (3, 2000)
    representing X, Y, Z accelerometer readings over time.
    """
    
    def __init__(
        self, 
        data_path: str, 
        split: str = 'train',
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Root directory containing good/ and bad/ folders
            split: Dataset split ('train', 'val', or 'test')
            class_names: List of class names, defaults to ['OK', 'NOK']
        """
        self.data_path = Path(data_path)
        self.split = split
        self.class_names = class_names or ['OK', 'NOK']
        
        self.samples = []
        self.labels = []
        
        # Load file paths and labels
        for label_idx, class_name in enumerate(['good', 'bad']):
            class_dir = self.data_path / class_name
            if not class_dir.exists():
                continue
            
            for h5_file in sorted(class_dir.glob('*.h5')):
                self.samples.append(h5_file)
                self.labels.append(label_idx)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (signal, label) where signal has shape (3, 2000)
        """
        h5_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load signal from HDF5
        with h5py.File(h5_path, 'r') as f:
            signal = np.array(f['vibration'])  # (3, 2000)
        
        # Convert to tensor
        signal = torch.from_numpy(signal).float()
        
        return signal, label
    
    def preprocessing(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocessing function for FeatureVisualization compatibility.
        
        Args:
            x: Input tensor
            
        Returns:
            Preprocessed tensor (identity for now)
        """
        return x


def get_layer_names(model: CNN1D_Wide, prefix: str = 'features') -> List[str]:
    """
    Get layer names for CRP analysis.
    
    Args:
        model: CNN1D_Wide model
        prefix: Prefix for layer names (default: 'features')
        
    Returns:
        List of layer names (e.g., ['features.0', 'features.3', 'features.6'])
    """
    layer_names = []
    for i, module in enumerate(model.features):
        if isinstance(module, nn.Conv1d):
            layer_names.append(f"{prefix}.{i}")
    return layer_names


if __name__ == "__main__":
    # Test model instantiation
    model = CNN1D_Wide()
    x = torch.randn(4, 3, 2000)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Layer names: {get_layer_names(model)}")
