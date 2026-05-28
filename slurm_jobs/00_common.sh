# =============================================================================
# 00_common.sh — Shared variables sourced by every step script.
# Usage: source slurm_jobs/steps/00_common.sh
# =============================================================================

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
CONTAINER_SIF="${CONTAINER_SIF:-${PROJECT_DIR}/container/tcd.sif}"
DATA_DIR="/data/datapool3/datasets/asadi/TCD/data"
MODEL="/workspace/TCD/cnn1d_model_new.ckpt"
CONFIG="configs/variantC_conv3_reference.yaml"

# All steps write to the same output root on the shared filesystem.
# Override by setting RESULTS_DIR before submitting, e.g.:
#   RESULTS_DIR=/my/path sbatch slurm_jobs/steps/01_crp_features.sh
RESULTS_DIR="${RESULTS_DIR:-$SLURM_SUBMIT_DIR/results/pipeline}"
mkdir -p "${RESULTS_DIR}"

# Convenience paths used by multiple steps
CRP_FEATURES="${RESULTS_DIR}/crp_features"
CONCEPTS_CONV3="${RESULTS_DIR}/variantC_conv3"
EVALUATION="${RESULTS_DIR}/evaluation_conv3"
FREQ_PSD="${RESULTS_DIR}/frequency_psd"
FREQ_DFT="${RESULTS_DIR}/frequency_relevance_dft_lrp"
FREQ_IDFT="${RESULTS_DIR}/frequency_relevance_vil_idft"
FREQ_STDFT="${RESULTS_DIR}/frequency_relevance_vil_stdft"
LAYER_SWEEP="${RESULTS_DIR}/layer_sweep"
METADATA="${RESULTS_DIR}/metadata"
PRUNING="${RESULTS_DIR}/pruning_relevance"
FREQ_VIS="${RESULTS_DIR}/frequency_visualizations"

# Pre-flight checks common to every step
preflight() {
  if [ ! -d "${PROJECT_DIR}" ]; then
    echo "ERROR: PROJECT_DIR not found: ${PROJECT_DIR}" >&2; exit 1
  fi
  if [ ! -f "${CONTAINER_SIF}" ]; then
    echo "ERROR: Container not found: ${CONTAINER_SIF}" >&2; exit 1
  fi
  if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: DATA_DIR not found: ${DATA_DIR}" >&2; exit 1
  fi
  if [ ! -f "${PROJECT_DIR}/cnn1d_model_new.ckpt" ]; then
    echo "ERROR: Model checkpoint not found: ${PROJECT_DIR}/cnn1d_model_new.ckpt" >&2; exit 1
  fi
}

# Require a file to exist before proceeding
require_file() {
  local file="$1"
  if [ ! -f "${file}" ]; then
    echo "ERROR: Required input not found: ${file}" >&2
    echo "       Run the prerequisite step first." >&2
    exit 1
  fi
}
