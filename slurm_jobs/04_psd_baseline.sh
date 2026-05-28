#!/bin/bash
#SBATCH --job-name=tcd_04_psd
#SBATCH --output=logs/tcd/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Step 4: PSD baseline (Welch power spectral density per prototype)
# Prerequisite: 02_discover_concepts.sh
# Outputs: results/pipeline/frequency_psd/
# Runtime: ~10 min
# Note: does NOT need --features, only --data and --concepts

set -euo pipefail
source "$(dirname "$0")/00_common.sh"
preflight
require_file "${CONCEPTS_CONV3}/tcd_model.pkl"

echo "[$(date)] Step 4: PSD baseline analysis"
echo "[$(date)] Output: ${FREQ_PSD}"

apptainer exec --nv \
  --bind "${PROJECT_DIR}:/workspace/TCD" \
  --bind "${DATA_DIR}:/workspace/data" \
  --bind "${RESULTS_DIR}:/workspace/out" \
  "${CONTAINER_SIF}" \
  bash -lc "
    set -euo pipefail
    cd /workspace/TCD

    python scripts/analyze_frequency.py \
      --data     /workspace/data \
      --concepts /workspace/out/variantC_conv3 \
      --output   /workspace/out/frequency_psd
  "

echo "[$(date)] ✓ Step 4 complete. Results: ${FREQ_PSD}"
