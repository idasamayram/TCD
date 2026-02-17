# TCD Phase 1-4 Updates - Usage Guide

This document provides usage examples for the new features added in the Phase 1-4 updates to the TCD (Temporal Concept Discovery) framework.

## Quick Start

All new features are enabled by default in `configs/default.yaml`. To use the new pipeline:

```bash
# Step 1: Run CRP analysis with class weights
python scripts/run_analysis.py --config configs/default.yaml --output results/crp_features

# Step 2: Discover concepts using Variant C (with class weights) or Variant D (vibration features)
# Variant C - Learned Clusters
python scripts/discover_concepts.py --variant C --features results/crp_features --output results/concepts_C

# Variant D - Comprehensive Vibration Features
python scripts/discover_concepts.py --variant D --features results/crp_features --output results/concepts_D
```

---

## 1. Class Weight Support

### Overview
Class weights help address imbalanced datasets by giving minority class samples proportionally higher influence on the concept space.

### Configuration
```yaml
# configs/default.yaml
analysis:
  use_class_weights: true  # Enable class weighting
```

### What it does
- **In `run_analysis.py`**: Scales concept relevances by `dataset.weights[class_id]` before storing
- **In `discover_concepts.py` (Variant C)**: Uses oversampling during GMM fitting to balance class influence
- **Effect**: Minority class (NOK/bad) concepts are amplified, majority class (OK/good) concepts are balanced

### Example
```python
from models.cnn1d_model import VibrationDataset

# Dataset automatically computes weights
dataset = VibrationDataset('./data')
print(f"Class weights: {dataset.weights}")
# Output: Class weights: tensor([1.2, 3.5])  # More weight to minority class
```

---

## 2. Default Layer Changed to Conv3

### Overview
Conv3 (64 filters, mid-level features) provides more semantically meaningful fault patterns compared to Conv1 (16 filters, low-level edges).

### Configuration
```yaml
# configs/default.yaml
intervention:
  target_layer: "conv3"  # Default for interventions

# In code, Variant C also defaults to conv3
tcd = LearnedClusterTCD(layer_name='conv3')  # Default
```

### Layer Comparison
| Layer | Filters | Kernel | Features Learned |
|-------|---------|--------|------------------|
| conv1 | 16 | 25 | Edge detection, basic smoothing |
| conv2 | 32 | 15 | Low-level patterns |
| **conv3** | **64** | **9** | **Mid-level temporal patterns** |
| conv4 | 128 | 5 | High-level fault signatures |

### Override Example
```bash
# Use conv4 for highest-level features
python scripts/discover_concepts.py --variant C --layer conv4 --features results/crp_features --output results/concepts_C_conv4
```

---

## 3. CNC-Validated LRP Composite

### Overview
The `cnc_validated` composite uses attribution rules validated on CNC vibration data from the thesis work. It provides superior attribution quality compared to generic composites.

### Configuration
```yaml
# configs/default.yaml
analysis:
  composite: "cnc_validated"  # Thesis-validated rules
```

### Rule Details
- **First Conv1d layer**: AlphaBeta(α=2, β=1) - Preserves positive and negative contributions
- **Other Conv1d layers**: Gamma(γ=0.25) - Emphasizes important activations
- **Linear layers**: Epsilon(ε=1e-6) - Numerical stability
- **Pooling layers**: Norm() - Conserves relevance
- **Activations/Dropout**: Pass() - No modification

### Available Composites
```python
from tcd.composites import get_composite

# All available composites
composites = ['epsilon_plus', 'epsilon', 'gradient', 'custom_cnn1d', 'cnc_validated']

for name in composites:
    composite = get_composite(name)
    # Use with attribution...
```

### When to Use
- **cnc_validated**: Best for CNC vibration fault detection (default)
- **epsilon_plus**: General-purpose, good baseline
- **gradient**: Fast but less accurate
- **custom_cnn1d**: Alternative validated rules

---

## 4. Variant D - Comprehensive Vibration Features

### Overview
Variant D automatically extracts 50+ vibration-relevant features from heatmaps and signals, enabling maximally automated concept discovery without domain expertise.

### Configuration
```yaml
# configs/default.yaml
tcd:
  variant: "D"  # Set as default variant (optional)
  vibration_features:
    window_size: 100
    n_concepts: 30  # Auto-select top 30 features
    use_feature_selection: true
    selection_method: "mutual_info"  # or "fisher"
    n_prototypes: 4
    gmm_covariance: "full"
```

### Feature Categories

#### Time-Domain Features (13 features per channel)
- RMS, Peak, Crest Factor, Kurtosis, Skewness
- Peak-to-Average Ratio, Zero-Crossing Rate
- Waveform Factor, Impulse Factor, Clearance Factor
- Standard Deviation, Variance

#### Frequency-Domain Features (12 features per channel)
- Spectral Centroid, Entropy, Dominant Frequency
- Spectral Kurtosis, Skewness, Rolloff, Flatness
- Band Energy Ratios (4 bands: 0-10Hz, 10-50Hz, 50-100Hz, 100-200Hz)

#### Vibration-Specific Features
- Envelope Analysis (5 features): Mean, Std, Peak, RMS, Kurtosis
- Inter-Axis Correlation (3 features): XY, XZ, YZ
- Energy Ratios (3 features): Per-axis energy distribution

### Usage Example
```python
from tcd.variants import VibrationFeatureTCD
import torch

# Initialize
tcd = VibrationFeatureTCD(
    sample_rate=400,
    n_concepts=20,  # Select top 20 discriminative features
    use_feature_selection=True,
    selection_method='mutual_info'
)

# Fit on training data
tcd.fit(heatmaps, labels=labels, signals=signals)

# Extract concepts
concept_vectors = tcd.extract_concepts(heatmaps, signals=signals)

# Get concept labels
labels = tcd.get_concept_labels()
# Example output: ['heatmap_ch0_rms', 'heatmap_ch1_kurtosis', 'signal_ch2_spectral_centroid', ...]

# Compute importance
importance = tcd.compute_concept_importance(heatmaps, signals=signals)
```

### Command-Line Usage
```bash
# Run Variant D concept discovery
python scripts/discover_concepts.py \
    --variant D \
    --features results/crp_features \
    --output results/concepts_D \
    --config configs/default.yaml

# Output includes:
# - Top 15 most important concepts
# - Per-class importance (OK vs NOK)
# - Concept relevance vectors
# - Trained TCD model (tcd_model.pkl)
```

### Feature Selection Methods

**Mutual Information (default)**
- Measures how much knowing a feature value reduces uncertainty about the class
- Good for capturing non-linear relationships
- More robust to outliers

**Fisher Score**
- Measures (between-class variance) / (within-class variance)
- Good for linear separability
- Faster computation

```yaml
# Switch to Fisher score
vibration_features:
  selection_method: "fisher"
```

### Customization

#### Using Raw Signals (if available)
```python
# Load raw signals from dataset
from models.cnn1d_model import VibrationDataset

dataset = VibrationDataset('./data')
signals = []
for i in range(len(dataset)):
    data, label = dataset[i]
    signals.append(data)
signals = torch.stack(signals)

# Fit with both heatmaps and signals for richer features
tcd.fit(heatmaps, labels=labels, signals=signals)
```

#### Adjusting Number of Features
```python
# Use all features (no selection)
tcd = VibrationFeatureTCD(n_concepts=None, use_feature_selection=False)

# Or select specific number
tcd = VibrationFeatureTCD(n_concepts=50, use_feature_selection=True)
```

---

## Integration with Existing Workflow

### Complete Pipeline Example
```bash
#!/bin/bash
# Complete TCD pipeline with new features

# 1. Run CRP analysis with class weights and CNC-validated composite
python scripts/run_analysis.py \
    --config configs/default.yaml \
    --model cnn1d_model_final.ckpt \
    --data ./data \
    --output results/crp_features

# 2a. Variant A - Frequency bands (legacy)
python scripts/discover_concepts.py \
    --variant A \
    --features results/crp_features \
    --output results/concepts_A

# 2b. Variant C - Learned clusters with class weights
python scripts/discover_concepts.py \
    --variant C \
    --features results/crp_features \
    --output results/concepts_C \
    --layer conv3 \
    --data ./data

# 2c. Variant D - Comprehensive vibration features (NEW)
python scripts/discover_concepts.py \
    --variant D \
    --features results/crp_features \
    --output results/concepts_D \
    --data ./data

# 3. Evaluate concepts (works with all variants)
python scripts/evaluate_concepts.py \
    --config configs/default.yaml \
    --concepts results/concepts_D \
    --output results/evaluation
```

---

## Troubleshooting

### Class Weights Not Applied
- Ensure `use_class_weights: true` in config
- Check that `VibrationDataset` has `weights` attribute
- Verify dataset has at least 2 classes

### Variant D Feature Extraction Issues
- Check that heatmaps have shape `(N, n_channels, n_timesteps)`
- Ensure sample_rate matches your data (default: 400 Hz)
- For NaN/Inf values, features are automatically replaced with 0

### GMM Fitting Warnings
- "Class has only X samples, need at least Y for GMM"
  - Increase training data or decrease `n_prototypes`
- GMM convergence issues
  - Try different `gmm_covariance` types ('spherical', 'diag', 'tied')
  - Increase `gmm_n_init` for more stable initialization

---

## Performance Tips

1. **Class Weights**: Most beneficial when class imbalance > 2:1
2. **Layer Selection**: 
   - Use conv3 for balanced performance
   - Use conv4 for highest-level abstractions (may need more data)
3. **Composite Selection**: 
   - `cnc_validated` is recommended for CNC vibration data
   - For other domains, try `epsilon_plus` first
4. **Variant D**:
   - Start with `n_concepts=20-30` for good balance
   - Use `mutual_info` for complex datasets
   - Use `fisher` for faster computation on linear problems

---

## References

- Original PCX Paper: [Link to paper if available]
- CNC Thesis: idasamayram/CNC repository
- Zennit Documentation: https://zennit.readthedocs.io/

---

*Last Updated: 2026-02-17*
