#!/bin/bash
#SBATCH --job-name=tcd_08_layer_sweep
#SBATCH --output=logs/tcd/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Step 8: Layer sweep (runs discover_concepts for conv1/conv2/conv3/conv4)
# Prerequisite: 01_crp_features.sh
# Outputs: results/pipeline/layer_sweep/  +  layer_sweep_summary.csv
# Runtime: ~1 hour (4 x discover_concepts)

set -euo pipefail
source "$(dirname "$0")/00_common.sh"
preflight
require_file "${CRP_FEATURES}/eps_relevances_class_0.hdf5"

echo "[$(date)] Step 8: Layer sweep analysis"
echo "[$(date)] Output: ${LAYER_SWEEP}"

apptainer exec --nv \
  --bind "${PROJECT_DIR}:/workspace/TCD" \
  --bind "${DATA_DIR}:/workspace/data" \
  --bind "${RESULTS_DIR}:/workspace/out" \
  "${CONTAINER_SIF}" \
  bash -lc "
    set -euo pipefail
    cd /workspace/TCD

    python scripts/run_layer_sweep.py \
      --config       ${CONFIG} \
      --features     /workspace/out/crp_features \
      --data         /workspace/data \
      --output-root  /workspace/out/layer_sweep \
      --summary      /workspace/out/layer_sweep_summary.csv \
      --run-discovery
  "

echo "[$(date)] ✓ Step 8 complete. Results: ${LAYER_SWEEP}"
