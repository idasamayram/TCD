#!/bin/bash
#SBATCH --job-name=tcd_09_metadata
#SBATCH --output=logs/tcd/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Step 9: Metadata analysis (machine/operation breakdown per prototype)
# Prerequisite: 02_discover_concepts.sh
# Outputs: results/pipeline/metadata/
# Runtime: ~10 min

set -euo pipefail
source "$(dirname "$0")/00_common.sh"
preflight
require_file "${CONCEPTS_CONV3}/tcd_model.pkl"

echo "[$(date)] Step 9: Metadata analysis"
echo "[$(date)] Output: ${METADATA}"

apptainer exec --nv \
  --bind "${PROJECT_DIR}:/workspace/TCD" \
  --bind "${DATA_DIR}:/workspace/data" \
  --bind "${RESULTS_DIR}:/workspace/out" \
  "${CONTAINER_SIF}" \
  bash -lc "
    set -euo pipefail
    cd /workspace/TCD

    python scripts/analyze_metadata.py \
      --data     /workspace/data \
      --concepts /workspace/out/variantC_conv3 \
      --output   /workspace/out/metadata
  "

echo "[$(date)] ✓ Step 9 complete. Results: ${METADATA}"
