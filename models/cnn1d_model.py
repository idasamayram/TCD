"""
CNN1D_Wide — Sequential version of the CNC thesis architecture for CRP compatibility.
"""

import torch
import torch.nn as nn
from typing import List


class CNN1D_Wide(nn.Module):
    def __init__(self):
        super(CNN1D_Wide, self).__init__()
        # Wider kernels to increase receptive field
        self.conv1 = nn.Conv1d(3, 16, kernel_size=25, stride=1, padding=12)  # Increased kernel size
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)  # Increased pooling
        self.dropout1 = nn.Dropout(0.2)  # Add dropout after first layer

        self.conv2 = nn.Conv1d(16, 32, kernel_size=15, stride=1, padding=7)  # Increased kernel size
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)  # Increased pooling
        self.dropout2 = nn.Dropout(0.2)  # Add dropout after second layer

        self.conv3 = nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4)  # Increased kernel size
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)  # Increased pooling
        self.dropout3 = nn.Dropout(0.2)  # Add dropout after third layer

        # NEW: Add a fourth convolutional layer for deeper network
        self.conv4 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(0.2)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)  # Changed input size to match conv4 output
        self.fc2 = nn.Linear(64, 2)  # Binary classification

        self.dropout = nn.Dropout(0.4)  # Increased dropout for final layer
        self.relu = nn.LeakyReLU(0.1)  # Using LeakyReLU for better gradient flow

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.dropout1(self.pool1(self.relu(self.conv1(x))))
        x = self.dropout2(self.pool2(self.relu(self.conv2(x))))
        x = self.dropout3(self.pool3(self.relu(self.conv3(x))))
        x = self.dropout4(self.pool4(self.relu(self.conv4(x))))

        x = self.global_avg_pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # No activation (we use CrossEntropyLoss)

        return x


def get_layer_names(model: CNN1D_Wide) -> List[str]:
    """Return conv layer names for CRP hooks."""
    return [f"features.{i}" for i, m in enumerate(model.features) if isinstance(m, nn.Conv1d)]


def load_from_original_weights(checkpoint_path: str, device: str = 'cpu') -> CNN1D_Wide:
    """
    Load weights saved with the original flat-attribute CNN1D_Wide
    (conv1, conv2, ...) into this Sequential version (features.0, features.4, ...).
    """
    model = CNN1D_Wide()
    state = torch.load(checkpoint_path, map_location=device)

    if any(k.startswith('features.') for k in state.keys()):
        model.load_state_dict(state)
        return model

    key_map = {
        'conv1.weight': 'features.0.weight',  'conv1.bias': 'features.0.bias',
        'conv2.weight': 'features.4.weight',  'conv2.bias': 'features.4.bias',
        'conv3.weight': 'features.8.weight',  'conv3.bias': 'features.8.bias',
        'conv4.weight': 'features.12.weight', 'conv4.bias': 'features.12.bias',
        'fc1.weight': 'classifier.2.weight',  'fc1.bias': 'classifier.2.bias',
        'fc2.weight': 'classifier.5.weight',  'fc2.bias': 'classifier.5.bias',
    }

    new_state = {key_map[k]: v for k, v in state.items() if k in key_map}
    model.load_state_dict(new_state, strict=False)
    return model