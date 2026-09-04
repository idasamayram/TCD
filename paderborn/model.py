"""1D CNN for bearing fault classification, deliberately kept LRP-friendly:

- No BatchNorm / GroupNorm / LayerNorm anywhere (these break standard LRP
  rules and need extra canonizers to handle correctly; easiest to just not
  use them).
- No residual/skip connections (additions of two branches need explicit
  rule handling in zennit; a plain feed-forward stack avoids that entirely).
- ReLU is used with inplace=False everywhere. In-place ops can overwrite
  activations that hooks need to read during the backward/relevance pass.
- Only layer types zennit handles natively out of the box: Conv1d, Linear,
  ReLU, MaxPool1d, AdaptiveAvgPool1d, Flatten, Dropout (identity at eval
  time anyway).

Architecture is a small "wide first kernel" CNN (à la WDCNN), which works
well for raw vibration signals: a large kernel in the first layer acts as
a learnable band-pass-ish filter, subsequent layers get progressively
narrower kernels.
"""
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 3,
                 base_channels: int = 16, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # wide first kernel, large stride to quickly reduce the ~4096-sample input
            nn.Conv1d(in_channels, base_channels, kernel_size=64, stride=8, padding=28),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(base_channels, base_channels * 2, kernel_size=16, stride=2, padding=7),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=8, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool1d(4),  # fixes the flattened size regardless of input length
        )

        flat_features = base_channels * 4 * 4

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_features, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # quick smoke test
    m = CNN1D(num_classes=3)
    x = torch.randn(8, 1, 4096)
    out = m(x)
    print("output shape:", out.shape)
    assert out.shape == (8, 3)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params: {n_params:,}")
    for name, mod in m.named_modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
            raise RuntimeError(f"Found a normalization layer at {name}, not LRP-friendly!")
    print("No BatchNorm/GroupNorm/LayerNorm layers found - OK for zennit.")
