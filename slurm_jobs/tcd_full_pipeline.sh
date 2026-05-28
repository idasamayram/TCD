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

# OUTPUT_ROOT is on the shared filesystem (accessible from headnode in real time).
# We do NOT use a scratch/tmp directory — every python script writes directly here
# so results are visible immediately without waiting for job completion or rsync.
OUTPUT_BASE="${OUTPUT_BASE:-$SLURM_SUBMIT_DIR/results/slurm}"
OUTPUT_ROOT="${OUTPUT_BASE}/${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_ROOT}" "$SLURM_SUBMIT_DIR/logs/tcd"

echo "[$(date)] Results will be written live to: ${OUTPUT_ROOT}"
echo "[$(date)] Monitor progress from headnode:  tail -f $SLURM_SUBMIT_DIR/logs/tcd/tcd_full_pipeline_${SLURM_JOB_ID}.out"

# ---------- PRE-FLIGHT CHECKS ----------
if [ ! -d "${PROJECT_DIR}" ]; then
  echo "ERROR: PROJECT_DIR does not exist: ${PROJECT_DIR}" >&2; exit 1
fi
if [ ! -f "${CONTAINER_SIF}" ]; then
  echo "ERROR: Container not found: ${CONTAINER_SIF}" >&2; exit 1
fi
if [ ! -d "${DATA_DIR}" ]; then
  echo "ERROR: DATA_DIR does not exist: ${DATA_DIR}" >&2; exit 1
fi
if [ ! -f "${PROJECT_DIR}/cnn1d_model_new.ckpt" ]; then
  echo "ERROR: Model checkpoint not found: ${PROJECT_DIR}/cnn1d_model_new.ckpt" >&2; exit 1
fi

# ---------- CONTAINER COMMAND ----------
# /workspace/out is bound directly to OUTPUT_ROOT on the shared filesystem.
# Every write inside the container is immediately visible from the headnode.
apptainer exec --nv \
  --bind "${PROJECT_DIR}:/workspace/TCD" \
  --bind "${DATA_DIR}:/workspace/data" \
  --bind "${OUTPUT_ROOT}:/workspace/out" \
  "${CONTAINER_SIF}" \
  bash -lc '
    set -uo pipefail
    cd /workspace/TCD

    # Helper: run a step, log pass/fail, never abort the pipeline
    run_step() {
      local name="$1"; shift
      echo ""
      echo "[$(date)] ===== ${name} ====="
      if "$@"; then
        echo "[$(date)] ✓ ${name} succeeded"
        return 0
      else
        local code=$?
        echo "[$(date)] ✗ ${name} FAILED (exit code: ${code}) — continuing pipeline"
        return ${code}
      fi
    }

    # Helper: check a prerequisite file exists before running a step
    need() {
      local file="$1"; local step="$2"
      if [ ! -f "${file}" ]; then
        echo "[$(date)] ✗ Skipping ${step} — prerequisite missing: ${file}"
        return 1
      fi
      return 0
    }

    # -------------------------------------------------------------------------
    # Step 1: CRP feature extraction
    # -------------------------------------------------------------------------
    run_step "Step 1/11: CRP feature extraction" \
      python scripts/run_analysis.py \
        --config  configs/variantC_conv3_reference.yaml \
        --model   /workspace/TCD/cnn1d_model_new.ckpt \
        --data    /workspace/data \
        --output  /workspace/out/crp_features

    # Verify both classes were saved before anything downstream runs
    CRP_OK=true
    echo "[$(date)] Verifying CRP feature files..."
    for class_id in 0 1; do
      for f in \
        eps_relevances_class_${class_id}.hdf5 \
        heatmaps_class_${class_id}.hdf5 \
        sample_ids_class_${class_id}.pt; do
        if [ ! -f /workspace/out/crp_features/${f} ]; then
          echo "  ✗ Missing: ${f}" >&2
          CRP_OK=false
        else
          echo "  ✓ ${f}"
        fi
      done
    done

    if [ "${CRP_OK}" = false ]; then
      echo "[$(date)] ERROR: CRP verification failed — steps 2-11 that need CRP will be skipped"
    fi

    # -------------------------------------------------------------------------
    # Step 2: Variant C concept discovery
    # -------------------------------------------------------------------------
    if [ "${CRP_OK}" = true ]; then
      run_step "Step 2/11: Variant C concept discovery" \
        python scripts/discover_concepts.py \
          --config  configs/variantC_conv3_reference.yaml \
          --variant C \
          --features /workspace/out/crp_features \
          --output   /workspace/out/variantC_conv3 \
          --layer    conv3 \
          --data     /workspace/data
    else
      echo "[$(date)] ✗ Skipping Step 2/11 — CRP features missing"
    fi

    # -------------------------------------------------------------------------
    # Step 3: Concept evaluation
    # -------------------------------------------------------------------------
    if need /workspace/out/variantC_conv3/tcd_model.pkl "Step 3/11: Concept evaluation"; then
      run_step "Step 3/11: Concept evaluation" \
        python scripts/evaluate_concepts.py \
          --config   configs/variantC_conv3_reference.yaml \
          --concepts /workspace/out/variantC_conv3 \
          --model    /workspace/TCD/cnn1d_model_new.ckpt \
          --data     /workspace/data \
          --output   /workspace/out/evaluation_conv3
    fi

    # -------------------------------------------------------------------------
    # Step 4: PSD baseline
    # Note: analyze_frequency.py does NOT accept --features
    # -------------------------------------------------------------------------
    if need /workspace/out/variantC_conv3/tcd_model.pkl "Step 4/11: PSD baseline"; then
      run_step "Step 4/11: PSD baseline analysis" \
        python scripts/analyze_frequency.py \
          --data     /workspace/data \
          --concepts /workspace/out/variantC_conv3 \
          --output   /workspace/out/frequency_psd
    fi

    # -------------------------------------------------------------------------
    # Step 5: DFT-LRP frequency relevance
    # -------------------------------------------------------------------------
    if need /workspace/out/variantC_conv3/tcd_model.pkl "Step 5/11: DFT-LRP"; then
      run_step "Step 5/11: DFT-LRP frequency relevance" \
        python scripts/analyze_frequency_relevance.py \
          --data     /workspace/data \
          --features /workspace/out/crp_features \
          --concepts /workspace/out/variantC_conv3 \
          --output   /workspace/out/frequency_relevance_dft_lrp \
          --method   dft_lrp \
          --max-samples-per-prototype 50
    fi

    # -------------------------------------------------------------------------
    # Step 6: VIL IDFT frequency relevance
    # -------------------------------------------------------------------------
    if need /workspace/out/variantC_conv3/tcd_model.pkl "Step 6/11: VIL IDFT"; then
      run_step "Step 6/11: VIL IDFT frequency relevance" \
        python scripts/analyze_frequency_relevance.py \
          --data     /workspace/data \
          --features /workspace/out/crp_features \
          --concepts /workspace/out/variantC_conv3 \
          --output   /workspace/out/frequency_relevance_vil_idft \
          --method   vil_idft \
          --max-samples-per-prototype 50
    fi

    # -------------------------------------------------------------------------
    # Step 7: VIL STDFT frequency relevance
    # Capped at 50 samples per prototype — using all samples takes many hours.
    # -------------------------------------------------------------------------
    if need /workspace/out/variantC_conv3/tcd_model.pkl "Step 7/11: VIL STDFT"; then
      run_step "Step 7/11: VIL STDFT frequency relevance" \
        python scripts/analyze_frequency_relevance.py \
          --data     /workspace/data \
          --features /workspace/out/crp_features \
          --concepts /workspace/out/variantC_conv3 \
          --output   /workspace/out/frequency_relevance_vil_stdft \
          --method   vil_stdft \
          --max-samples-per-prototype 50 \
          --vil-window-width  128 \
          --vil-window-shift   64 \
          --vil-window-shape  rectangle
    fi

    # -------------------------------------------------------------------------
    # Step 8: Layer sweep (independent — runs its own discover_concepts internally)
    # -------------------------------------------------------------------------
    if [ "${CRP_OK}" = true ]; then
      run_step "Step 8/11: Layer sweep analysis" \
        python scripts/run_layer_sweep.py \
          --config       configs/variantC_conv3_reference.yaml \
          --features     /workspace/out/crp_features \
          --data         /workspace/data \
          --output-root  /workspace/out/layer_sweep \
          --summary      /workspace/out/layer_sweep_summary.csv \
          --run-discovery
    else
      echo "[$(date)] ✗ Skipping Step 8/11 — CRP features missing"
    fi

    # -------------------------------------------------------------------------
    # Step 9: Metadata analysis
    # -------------------------------------------------------------------------
    if need /workspace/out/variantC_conv3/tcd_model.pkl "Step 9/11: Metadata analysis"; then
      run_step "Step 9/11: Metadata analysis" \
        python scripts/analyze_metadata.py \
          --data     /workspace/data \
          --concepts /workspace/out/variantC_conv3 \
          --output   /workspace/out/metadata
    fi

    # -------------------------------------------------------------------------
    # Step 10: Pruning
    # -------------------------------------------------------------------------
    if [ "${CRP_OK}" = true ]; then
      run_step "Step 10/11: Pruning analysis" \
        python scripts/prune_model.py \
          --model    /workspace/TCD/cnn1d_model_new.ckpt \
          --features /workspace/out/crp_features \
          --data     /workspace/data \
          --output   /workspace/out/pruning_relevance
    else
      echo "[$(date)] ✗ Skipping Step 10/11 — CRP features missing"
    fi

    # -------------------------------------------------------------------------
    # Step 11: Frequency visualizations (needs steps 5-7 outputs)
    # -------------------------------------------------------------------------
    if need /workspace/out/frequency_relevance_dft_lrp/prototype_frequency_relevance_dft_lrp.csv \
       "Step 11/11: Frequency visualizations"; then
      run_step "Step 11/11: Frequency analysis visualizations" \
        python scripts/visualize_frequency_analysis.py \
          --methods-dir \
            /workspace/out/frequency_relevance_dft_lrp \
            /workspace/out/frequency_relevance_vil_idft \
            /workspace/out/frequency_relevance_vil_stdft \
          --concepts /workspace/out/variantC_conv3 \
          --output   /workspace/out/frequency_visualizations
    fi

    echo ""
    echo "[$(date)] ===== Pipeline complete ====="
    echo "[$(date)] All outputs in /workspace/out (= ${OUTPUT_ROOT} on host)"
  '

echo "[$(date)] Job finished. Results at: ${OUTPUT_ROOT}"