#!/bin/bash
#SBATCH --job-name=tcd_full_pipeline
#SBATCH --output=logs/tcd/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -uo pipefail

echo "[$(date)] Starting job ${SLURM_JOB_ID} on ${HOSTNAME}"

# ---------- USER PATHS ----------
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
CONTAINER_SIF="${CONTAINER_SIF:-${PROJECT_DIR}/container/tcd.sif}"

DATA_DIR="/data/datapool3/datasets/asadi/TCD/data"

OUTPUT_BASE="${OUTPUT_BASE:-$SLURM_SUBMIT_DIR/results/slurm}"
OUTPUT_ROOT="${OUTPUT_BASE}/${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_ROOT}" "$SLURM_SUBMIT_DIR/logs/tcd"

if [ -f /etc/slurm/local_job_dir.sh ]; then
  source /etc/slurm/local_job_dir.sh
  SCRATCH_DIR="${LOCAL_JOB_DIR}/tcd_${SLURM_JOB_ID}"
else
  SCRATCH_DIR="/tmp/tcd_${SLURM_JOB_ID}"
fi
mkdir -p "${SCRATCH_DIR}"

SCRATCH_OUT="${SCRATCH_DIR}/out"
mkdir -p "${SCRATCH_OUT}"

# ---------- TRAP: always copy results back on exit, success or failure ----------
trap '
  EXIT_CODE=$?
  echo "[$(date)] Job exiting with code ${EXIT_CODE} — copying results back from scratch..."
  rsync -a "${SCRATCH_OUT}/" "${OUTPUT_ROOT}/"
  echo "[$(date)] Results saved to ${OUTPUT_ROOT}"
  rm -rf "${SCRATCH_DIR}"
  echo "[$(date)] Scratch directory cleaned up."
  exit ${EXIT_CODE}
' EXIT

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
if [ ! -f "${PROJECT_DIR}/cnn1d_model_new.ckpt" ]; then
  echo "ERROR: Model checkpoint not found at ${PROJECT_DIR}/cnn1d_model_new.ckpt" >&2
  exit 1
fi

# ---------- HELPER: run a pipeline step, log success/failure, never abort ----------
run_step() {
  local step_name="$1"
  shift
  echo ""
  echo "[$(date)] ===== ${step_name} ====="
  if "$@"; then
    echo "[$(date)] ✓ ${step_name} succeeded"
    return 0
  else
    local code=$?
    echo "[$(date)] ✗ ${step_name} FAILED (exit code: ${code}) — continuing pipeline"
    return ${code}
  fi
}

# ---------- CONTAINER COMMAND ----------
apptainer exec --nv \
  --bind "${PROJECT_DIR}:/workspace/TCD" \
  --bind "${DATA_DIR}:/workspace/data" \
  --bind "${SCRATCH_OUT}:/workspace/out" \
  "${CONTAINER_SIF}" \
  bash -lc '
    set -uo pipefail
    cd /workspace/TCD

    # Helper inside container (same as outer one)
    run_step() {
      local step_name="$1"
      shift
      echo ""
      echo "[$(date)] ===== ${step_name} ====="
      if "$@"; then
        echo "[$(date)] ✓ ${step_name} succeeded"
        return 0
      else
        local code=$?
        echo "[$(date)] ✗ ${step_name} FAILED (exit code: ${code}) — continuing pipeline"
        return ${code}
      fi
    }

    # 1) CRP feature extraction
    run_step "Step 1/11: CRP feature extraction" \
      python scripts/run_analysis.py \
        --config configs/variantC_conv3_reference.yaml \
        --data /workspace/data \
        --output /workspace/out/crp_features

    # Verify CRP feature files exist for BOTH classes before continuing
    echo "[$(date)] Verifying CRP feature files..."
    ALL_OK=true
    for class_id in 0 1; do
      for suffix in eps_relevances_class_${class_id}.hdf5 heatmaps_class_${class_id}.hdf5 sample_ids_class_${class_id}.pt; do
        if [ ! -f /workspace/out/crp_features/${suffix} ]; then
          echo "ERROR: Missing ${suffix}" >&2
          ALL_OK=false
        else
          echo "  ✓ ${suffix} verified"
        fi
      done
    done
    if [ "${ALL_OK}" = false ]; then
      echo "ERROR: CRP feature verification failed — cannot continue with concept discovery" >&2
      # Do not exit; remaining steps that dont need CRP features can still run
    else
      echo "  ✓ All CRP feature files verified"

      # 2) Variant C concept discovery (reference conv3)
      run_step "Step 2/11: Variant C concept discovery" \
        python scripts/discover_concepts.py \
          --config configs/variantC_conv3_reference.yaml \
          --variant C \
          --features /workspace/out/crp_features \
          --output /workspace/out/variantC_conv3 \
          --layer conv3 \
          --data /workspace/data

      # 3) Concept evaluation (depends on step 2)
      if [ -f /workspace/out/variantC_conv3/tcd_model.pkl ]; then
        run_step "Step 3/11: Concept evaluation" \
          python scripts/evaluate_concepts.py \
            --config configs/variantC_conv3_reference.yaml \
            --concepts /workspace/out/variantC_conv3 \
            --data /workspace/data \
            --output /workspace/out/evaluation_conv3
      else
        echo "[$(date)] ✗ Step 3/11: Skipping evaluation — concept discovery output missing"
      fi

      # 4) PSD baseline (prototype spectral energy)
      # Note: analyze_frequency.py does NOT take --features, only --data and --concepts
      if [ -f /workspace/out/variantC_conv3/tcd_model.pkl ]; then
        run_step "Step 4/11: PSD baseline analysis" \
          python scripts/analyze_frequency.py \
            --data /workspace/data \
            --concepts /workspace/out/variantC_conv3 \
            --output /workspace/out/frequency_psd
      else
        echo "[$(date)] ✗ Step 4/11: Skipping PSD analysis — concept discovery output missing"
      fi

      # 5) DFT-LRP prototype-conditioned relevance
      if [ -f /workspace/out/variantC_conv3/tcd_model.pkl ]; then
        run_step "Step 5/11: DFT-LRP frequency relevance" \
          python scripts/analyze_frequency_relevance.py \
            --data /workspace/data \
            --features /workspace/out/crp_features \
            --concepts /workspace/out/variantC_conv3 \
            --output /workspace/out/frequency_relevance_dft_lrp \
            --method dft_lrp \
            --max-samples-per-prototype 0
      else
        echo "[$(date)] ✗ Step 5/11: Skipping DFT-LRP — concept discovery output missing"
      fi

      # 6) VIL IDFT prototype-conditioned relevance
      if [ -f /workspace/out/variantC_conv3/tcd_model.pkl ]; then
        run_step "Step 6/11: VIL IDFT frequency relevance" \
          python scripts/analyze_frequency_relevance.py \
            --data /workspace/data \
            --features /workspace/out/crp_features \
            --concepts /workspace/out/variantC_conv3 \
            --output /workspace/out/frequency_relevance_vil_idft \
            --method vil_idft \
            --max-samples-per-prototype 0
      else
        echo "[$(date)] ✗ Step 6/11: Skipping VIL IDFT — concept discovery output missing"
      fi

      # 7) VIL STDFT prototype-conditioned relevance
      if [ -f /workspace/out/variantC_conv3/tcd_model.pkl ]; then
        run_step "Step 7/11: VIL STDFT frequency relevance" \
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
      else
        echo "[$(date)] ✗ Step 7/11: Skipping VIL STDFT — concept discovery output missing"
      fi

    fi  # end ALL_OK block

    # 8) Layer sweep (has its own --run-discovery, independent of step 2 output)
    run_step "Step 8/11: Layer sweep analysis" \
      python scripts/run_layer_sweep.py \
        --config configs/variantC_conv3_reference.yaml \
        --features /workspace/out/crp_features \
        --data /workspace/data \
        --output-root /workspace/out/layer_sweep \
        --summary /workspace/out/layer_sweep_summary.csv \
        --run-discovery

    # 9) Metadata analysis (depends on step 2)
    if [ -f /workspace/out/variantC_conv3/tcd_model.pkl ]; then
      run_step "Step 9/11: Metadata analysis" \
        python scripts/analyze_metadata.py \
          --data /workspace/data \
          --concepts /workspace/out/variantC_conv3 \
          --output /workspace/out/metadata
    else
      echo "[$(date)] ✗ Step 9/11: Skipping metadata analysis — concept discovery output missing"
    fi

    # 10) Pruning: relevance-only baseline
    run_step "Step 10/11: Pruning analysis" \
      python scripts/prune_model.py \
        --features /workspace/out/crp_features \
        --data /workspace/data \
        --output /workspace/out/pruning_relevance

    # 11) Frequency analysis visualizations (depends on steps 5-7)
    if [ -f /workspace/out/frequency_relevance_dft_lrp/prototype_frequency_relevance_dft_lrp.csv ]; then
      run_step "Step 11/11: Frequency analysis visualizations" \
        python scripts/visualize_frequency_analysis.py \
          --methods-dir /workspace/out/frequency_relevance_dft_lrp \
                        /workspace/out/frequency_relevance_vil_idft \
                        /workspace/out/frequency_relevance_vil_stdft \
          --concepts /workspace/out/variantC_conv3 \
          --output /workspace/out/frequency_visualizations
    else
      echo "[$(date)] ✗ Step 11/11: Skipping frequency visualizations — frequency relevance output missing"
    fi

    echo "[$(date)] All pipeline steps attempted."
  '