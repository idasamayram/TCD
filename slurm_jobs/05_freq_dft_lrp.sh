#!/bin/bash
#SBATCH --job-name=tcd_05_dft_lrp
#SBATCH --output=logs/tcd/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Step 5: DFT-LRP prototype-conditioned frequency relevance
# Prerequisite: 01_crp_features.sh + 02_discover_concepts.sh
# Outputs: results/pipeline/frequency_relevance_dft_lrp/
# Runtime: ~15 min with 50 samples per prototype

set -euo pipefail
source "$(dirname "$0")/00_common.sh"
preflight
require_file "${CRP_FEATURES}/heatmaps_class_0.hdf5"
require_file "${CONCEPTS_CONV3}/tcd_model.pkl"

echo "[$(date)] Step 5: DFT-LRP frequency relevance"
echo "[$(date)] Output: ${FREQ_DFT}"

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
      --output   /workspace/out/frequency_relevance_dft_lrp \
      --method   dft_lrp \
      --max-samples-per-prototype 50
  "

echo "[$(date)] ✓ Step 5 complete. Results: ${FREQ_DFT}"
