#!/bin/bash
#SBATCH --job-name=tcd_06_vil_idft
#SBATCH --output=logs/tcd/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Step 6: VIL IDFT prototype-conditioned frequency relevance
# Prerequisite: 01_crp_features.sh + 02_discover_concepts.sh
# Outputs: results/pipeline/frequency_relevance_vil_idft/
# Runtime: ~15 min with 50 samples per prototype

set -euo pipefail
source "$(dirname "$0")/00_common.sh"
preflight
require_file "${CRP_FEATURES}/heatmaps_class_0.hdf5"
require_file "${CONCEPTS_CONV3}/tcd_model.pkl"

echo "[$(date)] Step 6: VIL IDFT frequency relevance"
echo "[$(date)] Output: ${FREQ_IDFT}"

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
      --concepts /workspace/out/variantC_conv3 \
      --output   /workspace/out/frequency_relevance_vil_idft \
      --method   vil_idft \
      --max-samples-per-prototype 50
  "

echo "[$(date)] ✓ Step 6 complete. Results: ${FREQ_IDFT}"
