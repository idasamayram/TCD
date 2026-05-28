#!/bin/bash
#SBATCH --job-name=tcd_01_crp
#SBATCH --output=logs/tcd/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Step 1: CRP feature extraction
# Outputs: results/pipeline/crp_features/
# Runtime: ~30 min on GPU
# Next step: 02_discover_concepts.sh

set -euo pipefail
source "$(dirname "$0")/00_common.sh"
preflight

echo "[$(date)] Step 1: CRP feature extraction"
echo "[$(date)] Output: ${CRP_FEATURES}"

apptainer exec --nv \
  --bind "${PROJECT_DIR}:/workspace/TCD" \
  --bind "${DATA_DIR}:/workspace/data" \
  --bind "${RESULTS_DIR}:/workspace/out" \
  "${CONTAINER_SIF}" \
  bash -lc "
    set -euo pipefail
    cd /workspace/TCD

    python scripts/run_analysis.py \
      --config  ${CONFIG} \
      --model   ${MODEL} \
      --data    /workspace/data \
      --output  /workspace/out/crp_features

    echo '[$(date)] Verifying output files...'
    for class_id in 0 1; do
      for f in \
        eps_relevances_class_\${class_id}.hdf5 \
        heatmaps_class_\${class_id}.hdf5 \
        sample_ids_class_\${class_id}.pt; do
        if [ ! -f /workspace/out/crp_features/\${f} ]; then
          echo 'ERROR: Missing \${f}' >&2; exit 1
        fi
        echo '  ✓ \${f}'
      done
    done
  "

echo "[$(date)] ✓ Step 1 complete. Results: ${CRP_FEATURES}"
