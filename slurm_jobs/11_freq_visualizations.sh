#!/bin/bash
#SBATCH --job-name=tcd_11_freq_vis
#SBATCH --output=logs/tcd/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0
#SBATCH --mem=16G
#SBATCH --time=00:20:00

# Step 11: Frequency analysis visualizations
# Prerequisite: 05_freq_dft_lrp.sh + 06_freq_vil_idft.sh + 07_freq_vil_stdft.sh
# Outputs: results/pipeline/frequency_visualizations/
# Runtime: ~5 min

set -euo pipefail
source "$(dirname "$0")/00_common.sh"
preflight
require_file "${FREQ_DFT}/prototype_frequency_relevance_dft_lrp.csv"
require_file "${FREQ_IDFT}/prototype_frequency_relevance_vil_idft.csv"
require_file "${FREQ_STDFT}/prototype_frequency_relevance_vil_stdft.csv"
require_file "${CONCEPTS_CONV3}/tcd_model.pkl"

echo "[$(date)] Step 11: Frequency analysis visualizations"
echo "[$(date)] Output: ${FREQ_VIS}"

apptainer exec --nv \
  --bind "${PROJECT_DIR}:/workspace/TCD" \
  --bind "${DATA_DIR}:/workspace/data" \
  --bind "${RESULTS_DIR}:/workspace/out" \
  "${CONTAINER_SIF}" \
  bash -lc "
    set -euo pipefail
    cd /workspace/TCD

    python scripts/visualize_frequency_analysis.py \
      --methods-dir \
        /workspace/out/frequency_relevance_dft_lrp \
        /workspace/out/frequency_relevance_vil_idft \
        /workspace/out/frequency_relevance_vil_stdft \
      --concepts /workspace/out/variantC_conv3 \
      --output   /workspace/out/frequency_visualizations
  "

echo "[$(date)] ✓ Step 11 complete. Results: ${FREQ_VIS}"
