#!/bin/bash
#SBATCH --job-name=tcd_10_pruning
#SBATCH --output=logs/tcd/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Step 10: Relevance-based pruning analysis
# Prerequisite: 01_crp_features.sh
# Outputs: results/pipeline/pruning_relevance/
# Runtime: ~20 min

set -euo pipefail
source "$(dirname "$0")/00_common.sh"
preflight
require_file "${CRP_FEATURES}/eps_relevances_class_0.hdf5"

echo "[$(date)] Step 10: Pruning analysis"
echo "[$(date)] Output: ${PRUNING}"

apptainer exec --nv \
  --bind "${PROJECT_DIR}:/workspace/TCD" \
  --bind "${DATA_DIR}:/workspace/data" \
  --bind "${RESULTS_DIR}:/workspace/out" \
  "${CONTAINER_SIF}" \
  bash -lc "
    set -euo pipefail
    cd /workspace/TCD

    python scripts/prune_model.py \
      --model    ${MODEL} \
      --features /workspace/out/crp_features \
      --data     /workspace/data \
      --output   /workspace/out/pruning_relevance
  "

echo "[$(date)] ✓ Step 10 complete. Results: ${PRUNING}"
