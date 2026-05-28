#!/bin/bash
#SBATCH --job-name=tcd_02_concepts
#SBATCH --output=logs/tcd/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Step 2: Variant C concept discovery (GMM prototypes)
# Prerequisite: 01_crp_features.sh
# Outputs: results/pipeline/variantC_conv3/
# Runtime: ~20 min (CPU only)
# Next step: 03_evaluate_concepts.sh

set -euo pipefail
source "$(dirname "$0")/00_common.sh"
preflight
require_file "${CRP_FEATURES}/eps_relevances_class_0.hdf5"
require_file "${CRP_FEATURES}/eps_relevances_class_1.hdf5"

echo "[$(date)] Step 2: Variant C concept discovery"
echo "[$(date)] Input:  ${CRP_FEATURES}"
echo "[$(date)] Output: ${CONCEPTS_CONV3}"

apptainer exec --nv \
  --bind "${PROJECT_DIR}:/workspace/TCD" \
  --bind "${DATA_DIR}:/workspace/data" \
  --bind "${RESULTS_DIR}:/workspace/out" \
  "${CONTAINER_SIF}" \
  bash -lc "
    set -euo pipefail
    cd /workspace/TCD

    python scripts/discover_concepts.py \
      --config   ${CONFIG} \
      --variant  C \
      --features /workspace/out/crp_features \
      --output   /workspace/out/variantC_conv3 \
      --layer    conv4 \
      --data     /workspace/data

    if [ ! -f /workspace/out/variantC_conv3/tcd_model.pkl ]; then
      echo 'ERROR: tcd_model.pkl not created — concept discovery failed' >&2; exit 1
    fi
    echo '  ✓ tcd_model.pkl'
    echo '  ✓ results.pkl'
  "

echo "[$(date)] ✓ Step 2 complete. Results: ${CONCEPTS_CONV3}"
