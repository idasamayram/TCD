#!/bin/bash
#SBATCH --job-name=tcd_03_evaluate
#SBATCH --output=logs/tcd/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00

# Step 3: Concept evaluation (faithfulness, intervention, UMAP)
# Prerequisite: 02_discover_concepts.sh
# Outputs: results/pipeline/evaluation_conv3/
# Runtime: ~1-2 hours (incremental faithfulness is slow)
# Next step: 04_psd_baseline.sh (independent, can run in parallel)

set -euo pipefail
source "$(dirname "$0")/00_common.sh"
preflight
require_file "${CONCEPTS_CONV3}/tcd_model.pkl"

echo "[$(date)] Step 3: Concept evaluation"
echo "[$(date)] Input:  ${CONCEPTS_CONV3}"
echo "[$(date)] Output: ${EVALUATION}"

apptainer exec --nv \
  --bind "${PROJECT_DIR}:/workspace/TCD" \
  --bind "${DATA_DIR}:/workspace/data" \
  --bind "${RESULTS_DIR}:/workspace/out" \
  "${CONTAINER_SIF}" \
  bash -lc "
    set -euo pipefail
    cd /workspace/TCD

    python scripts/evaluate_concepts.py \
      --config   ${CONFIG} \
      --concepts /workspace/out/variantC_conv3 \
      --model    ${MODEL} \
      --data     /workspace/data \
      --output   /workspace/out/evaluation_conv3
  "

echo "[$(date)] ✓ Step 3 complete. Results: ${EVALUATION}"
