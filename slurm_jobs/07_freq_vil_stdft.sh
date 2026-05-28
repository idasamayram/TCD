#!/bin/bash
#SBATCH --job-name=tcd_07_vil_stdft
#SBATCH --output=logs/tcd/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Step 7: VIL STDFT prototype-conditioned frequency relevance
# Prerequisite: 01_crp_features.sh + 02_discover_concepts.sh
# Outputs: results/pipeline/frequency_relevance_vil_stdft/
# Runtime: ~30 min with 50 samples per prototype
# WARNING: Using all samples (--max-samples-per-prototype 0) takes many hours.
#          Keep at 50 unless you have a specific reason to use more.

set -euo pipefail
source "$(dirname "$0")/00_common.sh"
preflight
require_file "${CRP_FEATURES}/heatmaps_class_0.hdf5"
require_file "${CONCEPTS_CONV3}/tcd_model.pkl"

echo "[$(date)] Step 7: VIL STDFT frequency relevance"
echo "[$(date)] Output: ${FREQ_STDFT}"

apptainer exec --nv \
  --bind "${PROJECT_DIR}:/workspace/TCD" \
  --bind "${DATA_DIR}:/workspace/data" \
  --bind "${RESULTS_DIR}:/workspace/out" \
  "${CONTAINER_SIF}" \
  bash -lc "
    set -euo pipefail
    cd /workspace/TCD

    python scripts/analyze_frequency_relevance.py \
      --data     /workspace/data \
      --features /workspace/out/crp_features \
      --concepts /workspace/out/variantC_conv4 \
      --output   /workspace/out/frequency_relevance_vil_stdft \
      --method   vil_stdft \
      --max-samples-per-prototype 50 \
      --vil-window-width  128 \
      --vil-window-shift   64 \
      --vil-window-shape  rectangle
  "

echo "[$(date)] ✓ Step 7 complete. Results: ${FREQ_STDFT}"
