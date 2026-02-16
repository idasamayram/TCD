# TCD — Temporal Concept Discovery

**Temporal Concept Discovery** for 1D time-series explainability, extending CRP (Concept Relevance Propagation) and PCX (Prototypical Concept-based Explanations) from 2D images to vibration-based industrial fault detection.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

TCD adapts state-of-the-art concept-based XAI methods from computer vision to 1D time-series analysis. It discovers interpretable temporal concepts in neural network predictions for vibration signal classification.

**Key Features:**
- 🔧 **Fixed CRP for 1D**: Preserves multi-channel structure (X/Y/Z accelerometer axes) unlike standard CRP
- 📊 **Three TCD Variants**: Filterbank (frequency bands), temporal descriptors, and learned GMM prototypes
- 🎯 **Causal Testing**: Concept intervention to measure faithfulness
- 📈 **Prototype Discovery**: GMM-based clustering following PCX paper methodology
- 🔍 **1D Visualization**: Custom time-series plotting (no image dependencies)

## Architecture

### Core Framework (`tcd/`)
- **`attribution.py`**: `TimeSeriesCondAttribution` — Critical fix for CRP's heatmap_modifier to preserve channel structure
- **`concepts.py`**: `ChannelConcept` wrapper for 1D
- **`composites.py`**: LRP composites (Epsilon+, Epsilon, Gradient) for Conv1d
- **`prototypes.py`**: GMM-based prototype discovery (adapted from PCX)
- **`intervention.py`**: Concept suppression/amplification for causal testing
- **`evaluation.py`**: Faithfulness, stability, purity metrics
- **`visualization.py`**: 1D signal plotting with heatmap overlays
- **`feature_visualization.py`**: Adapted FeatureVisualization for time series

### TCD Variants (`tcd/variants/`)

#### **Variant A: Filterbank** (✅ Fully Implemented)
Frequency-band concepts using physics-informed filters:
- 0-10 Hz: Normal operation signatures
- 10-50 Hz: Transition band
- 50-100 Hz: Mid-frequency
- 100-200 Hz: Fault signatures (~150 Hz)

**Algorithm**: Apply bandpass filters to relevance heatmaps, compute energy per band, assign soft concept weights.

#### **Variant B: Temporal Descriptors** (🚧 Skeleton)
Extract temporal patterns via descriptors:
- Slope statistics (rise/fall rates)
- Peak/burst characteristics
- Autocorrelation structure
- Spectral density

**TODO**: Implement descriptor extraction and clustering.

#### **Variant C: Learned Clusters / PCX-Style** (🚧 Skeleton with working GMM)
Direct PCX adaptation for 1D:
1. Collect CRP concept relevance vectors ν^rel at chosen layer
2. Filter to correctly predicted samples, fit GMM per class
3. Each Gaussian component = one prediction sub-strategy (prototype)
4. Assign new samples via log-likelihood
5. Compute deviations Δ = ν - μ for interpretation

**Status**: GMM core (`prototypes.py`) fully functional. TODO: Intervention pipeline, visualization.

## Installation

```bash
# Clone repository
git clone https://github.com/idasamayram/TCD.git
cd TCD

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**
- PyTorch ≥1.13
- zennit ≥0.5 (LRP framework)
- crp ≥0.6 (CRP framework)
- scikit-learn ≥1.0
- scipy, numpy, matplotlib, h5py, pyyaml, tqdm

## Quick Start

### 1. Model and Data

**Model**: `CNN1D_Wide` — 3-layer Conv1d architecture (no BatchNorm for clean LRP)
```python
from models.cnn1d_model import CNN1D_Wide

model = CNN1D_Wide(num_classes=2, num_channels=3)
# Input: (batch, 3, 2000) — 3 accelerometer axes, 2000 timesteps (5s @ 400Hz)
# Output: (batch, 2) — binary classification logits
```

**Dataset**: `VibrationDataset` — Loads .h5 files from `good/` and `bad/` folders
```python
from models.cnn1d_model import VibrationDataset

dataset = VibrationDataset('path/to/data', split='train')
# Each sample: (3, 2000) tensor from HDF5 'vibration' dataset
```

### 2. Run CRP Analysis

Collect concept features across dataset:
```bash
python scripts/run_analysis.py \
    --config configs/default.yaml \
    --model path/to/model.ckpt \
    --data path/to/data \
    --output results/crp_features
```

**Outputs:**
- `eps_relevances_class_{0,1}.hdf5` — Per-layer concept relevance vectors
- `outputs_class_{0,1}.pt` — Model logits
- `sample_ids_class_{0,1}.pt` — Sample identifiers
- `heatmaps_class_{0,1}.hdf5` — Input-level relevance

### 3. Discover Concepts

**Variant A (Filterbank):**
```bash
python scripts/discover_concepts.py \
    --variant A \
    --features results/crp_features \
    --output results/concepts_A
```

**Variant C (Learned Clusters):**
```bash
python scripts/discover_concepts.py \
    --variant C \
    --features results/crp_features \
    --output results/concepts_C \
    --layer conv1
```

### 4. Evaluate Concepts

Test faithfulness via intervention:
```bash
python scripts/evaluate_concepts.py \
    --concepts results/concepts_C \
    --model path/to/model.ckpt \
    --data path/to/data \
    --output results/evaluation
```

**Metrics:**
- **Faithfulness**: Correlation between relevance and intervention effect
- **Stability**: Consistency across similar samples
- **Purity**: Distinctiveness of concepts
- **Coverage**: Prototype assignment statistics

## Configuration

See `configs/default.yaml` for all options:
```yaml
model:
  path: "path/to/model.ckpt"
  name: "cnn1d_wide"

data:
  path: "path/to/data"
  num_channels: 3
  seq_length: 2000
  sample_rate: 400

tcd:
  variant: "A"  # A, B, or C
  n_concepts: 4
  filterbank_bands: [[0, 10], [10, 50], [50, 100], [100, 200]]
  n_prototypes: 4
  top_k_samples: 6
```

## Testing

Run tests to verify core functionality:
```bash
# Test attribution (heatmap shape preservation)
python tests/test_attribution.py

# Test concept extraction
python tests/test_concepts.py
```

**Note**: Tests require dependencies to be installed first.

## Project Structure

```
TCD/
├── configs/
│   └── default.yaml              # Experiment configuration
├── models/
│   ├── __init__.py
│   └── cnn1d_model.py            # CNN1D_Wide + VibrationDataset
├── tcd/
│   ├── __init__.py
│   ├── attribution.py            # TimeSeriesCondAttribution (fixed heatmap)
│   ├── concepts.py               # ChannelConcept for 1D
│   ├── composites.py             # LRP composites
│   ├── visualization.py          # 1D signal plotting
│   ├── feature_visualization.py  # Adapted FeatureVisualization
│   ├── prototypes.py             # GMM prototype discovery
│   ├── intervention.py           # Concept suppression/amplification
│   ├── evaluation.py             # Faithfulness, stability metrics
│   └── variants/
│       ├── filterbank.py         # Variant A (FULL)
│       ├── temporal_descriptors.py  # Variant B (SKELETON)
│       └── learned_clusters.py   # Variant C (SKELETON)
├── scripts/
│   ├── run_analysis.py           # Step 1: CRP feature collection
│   ├── discover_concepts.py      # Step 2: TCD pipeline
│   └── evaluate_concepts.py      # Step 3: Intervention + validation
├── notebooks/
│   └── tcd_demo.ipynb            # End-to-end demo (TODO)
└── tests/
    ├── test_attribution.py       # Verify heatmap shape preservation
    └── test_concepts.py          # Verify concept extraction
```

## Critical Technical Details

### 1. Heatmap Modifier Fix

**Problem**: Default CRP `CondAttribution.heatmap_modifier()` collapses spatial dimensions, destroying multi-channel structure.

**Solution**: `TimeSeriesCondAttribution` preserves full (batch, channels, timesteps) shape:
```python
class TimeSeriesCondAttribution(CondAttribution):
    def heatmap_modifier(self, data, on_device=None):
        heatmap = data.grad.detach()
        # DO NOT collapse — keep (batch, 3, 2000) shape
        return heatmap.to(on_device) if on_device else heatmap
```

### 2. ChannelConcept Works for 1D

The base `ChannelConcept.attribute()` correctly handles 1D:
- `.view(*shape[:2], -1)` flattens (B, C, T) → (B, C, T) ✓
- Returns per-channel relevance: (B, num_filters)

### 3. LRP Composites

Use `zennit.types.Convolution` (covers Conv1d and Conv2d), not `torch.nn.Conv2d` specifically.

### 4. 1D Visualization

All visualization is signal-based (no images):
- `plot_ts_heatmap()`: Signal with bwr colormap overlay
- `plot_concept_relevance()`: Per-concept traces
- `plot_prototype_grid()`: Grid of prototype signals
- `plot_deviation_matrix()`: Heatmap of Δ vectors

## References

This work adapts methods from:

1. **PCX** (Dreyer et al., CVPR 2024): Prototypical Concept-based Explanations
   - Repository: `idasamayram/pcx_codes`
   - Key patterns: GMM prototype discovery, concept relevance collection, conditional heatmaps

2. **CRP** (Achtibat et al.): Concept Relevance Propagation
   - Repository: `idasamayram/zennit-crp` (user fork with 1D adaptations)
   - Key fix: `heatmap_modifier` for channel preservation

3. **User's Thesis**: `idasamayram/CNC`
   - CNN1D_Wide architecture, VibrationDataset
   - Binary fault detection: OK (0) vs NOK (1)
   - 99.8% test accuracy on industrial bearing data

## Citation

If you use this code, please cite:

```bibtex
@software{tcd2024,
  author = {Samay, Ida},
  title = {TCD: Temporal Concept Discovery for 1D Time Series},
  year = {2024},
  url = {https://github.com/idasamayram/TCD}
}
```

## License

MIT License — see LICENSE file for details.

## Contributing

Contributions welcome! Priority areas:
- Complete Variant B (temporal descriptors)
- Complete Variant C intervention pipeline
- End-to-end demo notebook
- Additional evaluation metrics
- Support for multi-class classification

## Acknowledgments

Built on top of:
- [zennit](https://github.com/chr5tphr/zennit) — LRP framework
- [zennit-crp](https://github.com/rachtibat/zennit-crp) — CRP implementation
- PCX paper methodology — CVPR 2024 SAIAD Workshop

---

**Status**: Core framework complete ✅ | Variant A complete ✅ | Variants B/C partial 🚧