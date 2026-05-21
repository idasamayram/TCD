#!/bin/bash
#SBATCH --job-name=tcd_full_pipeline
#SBATCH --output=logs/tcd/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

echo "[$(date)] Starting job ${SLURM_JOB_ID} on ${HOSTNAME}"

# ---------- USER PATHS ----------
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
CONTAINER_SIF="${CONTAINER_SIF:-${PROJECT_DIR}/container/tcd.sif}"

# Prefer the shared dataset in datapool3 for reproducibility and to avoid accidental drift.
DATA_DIR="/data/datapool3/datasets/asadi/TCD/data"
# Alternative local copy (not recommended as default):
# DATA_DIR="${PROJECT_DIR}/data"

# Output root (per-job folder will be created automatically)
OUTPUT_BASE="${OUTPUT_BASE:-$SLURM_SUBMIT_DIR/results/slurm}"
OUTPUT_ROOT="${OUTPUT_BASE}/${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_ROOT}" "$SLURM_SUBMIT_DIR/logs/tcd"

# If your cluster provides a local scratch helper, use it; otherwise fallback to /tmp
if [ -f /etc/slurm/local_job_dir.sh ]; then
  # shellcheck disable=SC1091
  source /etc/slurm/local_job_dir.sh
  SCRATCH_DIR="${LOCAL_JOB_DIR}/tcd_${SLURM_JOB_ID}"
else
  SCRATCH_DIR="/tmp/tcd_${SLURM_JOB_ID}"
fi
mkdir -p "${SCRATCH_DIR}"

# Stage job outputs in scratch first, then copy back at the end.
SCRATCH_OUT="${SCRATCH_DIR}/out"
mkdir -p "${SCRATCH_OUT}"

# ---------- PRE-FLIGHT CHECKS ----------
if [ ! -d "${PROJECT_DIR}" ]; then
  echo "ERROR: PROJECT_DIR does not exist: ${PROJECT_DIR}" >&2
  exit 1
fi
if [ ! -f "${CONTAINER_SIF}" ]; then
  echo "ERROR: CONTAINER_SIF does not exist: ${CONTAINER_SIF}" >&2
  exit 1
fi
if [ ! -d "${DATA_DIR}" ]; then
  echo "ERROR: DATA_DIR does not exist: ${DATA_DIR}" >&2
  exit 1
fi

# ---------- CONTAINER COMMAND ----------
# Bind project and data read-write/read-only as needed.
# We bind datapool3 data into /workspace/data to keep CLI commands clean.
apptainer exec --nv \
  --bind "${PROJECT_DIR}:/workspace/TCD" \
  --bind "${DATA_DIR}:/workspace/data" \
  --bind "${SCRATCH_OUT}:/workspace/out" \
  "${CONTAINER_SIF}" \
  bash -lc '
    set -euo pipefail
    cd /workspace/TCD

    # 1) CRP feature extraction
    echo "[$(date)] Step 1/10: CRP feature extraction..."
    python scripts/run_analysis.py \
      --config configs/variantC_conv3_reference.yaml \
      --data /workspace/data \
      --output /workspace/out/crp_features

    # 2) Variant C concept discovery (reference conv3)
    echo "[$(date)] Step 2/10: Variant C concept discovery..."
    python scripts/discover_concepts.py \
      --config configs/variantC_conv3_reference.yaml \
      --variant C \
      --features /workspace/out/crp_features \
      --output /workspace/out/variantC_conv3 \
      --layer conv3 \
      --data /workspace/data

    # 3) Concept evaluation
    echo "[$(date)] Step 3/10: Concept evaluation..."
    python scripts/evaluate_concepts.py \
      --config configs/variantC_conv3_reference.yaml \
      --concepts /workspace/out/variantC_conv3 \
      --model ./cnn1d_model_new.ckpt \
      --data /workspace/data \
      --output /workspace/out/evaluation_conv3

    # 4) PSD baseline (prototype spectral energy)
    echo "[$(date)] Step 4/10: PSD baseline analysis..."
    python scripts/analyze_frequency.py \
      --data /workspace/data \
      --features /workspace/out/crp_features \
      --concepts /workspace/out/variantC_conv3 \
      --output /workspace/out/frequency_psd

    # 5) DFT-LRP prototype-conditioned relevance
    echo "[$(date)] Step 5/10: DFT-LRP frequency relevance..."
    python scripts/analyze_frequency_relevance.py \
      --data /workspace/data \
      --features /workspace/out/crp_features \
      --concepts /workspace/out/variantC_conv3 \
      --output /workspace/out/frequency_relevance_dft_lrp \
      --method dft_lrp \
      --max-samples-per-prototype 0

    # 6) VIL IDFT prototype-conditioned relevance (NEW: distance-to-centroid)
    echo "[$(date)] Step 6/10: VIL IDFT frequency relevance..."
    python scripts/analyze_frequency_relevance.py \
      --data /workspace/data \
      --features /workspace/out/crp_features \
      --concepts /workspace/out/variantC_conv3 \
      --output /workspace/out/frequency_relevance_vil_idft \
      --method vil_idft \
      --max-samples-per-prototype 0

    # 7) VIL STDFT prototype-conditioned relevance (NEW: time-frequency)
    echo "[$(date)] Step 7/10: VIL STDFT frequency relevance..."
    python scripts/analyze_frequency_relevance.py \
      --data /workspace/data \
      --features /workspace/out/crp_features \
      --concepts /workspace/out/variantC_conv3 \
      --output /workspace/out/frequency_relevance_vil_stdft \
      --method vil_stdft \
      --max-samples-per-prototype 0 \
      --vil-window-width 128 \
      --vil-window-shift 64 \
      --vil-window-shape rectangle

    # 8) Layer sweep summary (conv1..conv4)
    echo "[$(date)] Step 8/10: Layer sweep analysis..."
    python scripts/run_layer_sweep.py \
      --config configs/variantC_conv3_reference.yaml \
      --features /workspace/out/crp_features \
      --data /workspace/data \
      --output-root /workspace/out/layer_sweep \
      --summary /workspace/out/layer_sweep_summary.csv \
      --run-discovery

    # 9) Metadata analysis
    echo "[$(date)] Step 9/10: Metadata analysis..."
    python scripts/analyze_metadata.py \
      --data /workspace/data \
      --concepts /workspace/out/variantC_conv3 \
      --output /workspace/out/metadata

    # 10) Pruning: relevance-only baseline
    echo "[$(date)] Step 10/10: Pruning analysis..."
    python scripts/prune_model.py \
      --model ./cnn1d_model_new.ckpt \
      --features /workspace/out/crp_features \
      --data /workspace/data \
      --output /workspace/out/pruning_relevance

    echo "[$(date)] All pipeline steps complete."
  '

# ---------- COPY RESULTS BACK ----------
echo "[$(date)] Copying results back from scratch..."
rsync -a --delete "${SCRATCH_OUT}/" "${OUTPUT_ROOT}/"
echo "[$(date)] Copy complete."

# ---------- CLEANUP ----------
rm -rf "${SCRATCH_DIR}"
echo "[$(date)] Scratch directory cleaned up."

echo "[$(date)] Finished. Results copied to: ${OUTPUT_ROOT}"
