#!/bin/bash
#SBATCH --job-name=tcd_job
#SBATCH --output=%j_%x.out
#SBATCH --partition=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
source "/etc/slurm/local_job_dir.sh"
echo "PWD: $(pwd)"
echo "LOCAL_JOB_DIR=${LOCAL_JOB_DIR}"
echo "DATAPOOL3=${DATAPOOL3}"
# Host paths
CONTAINER=/data/cluster/users/asadi/projects/TCD/container/tcd.sif
REPO=/data/cluster/users/asadi/projects/TCD
DATASET_HOST=$DATAPOOL3/datasets/asadi/TCD/data
# ----- CHANGE ONLY THESE TWO LINES PER RUN -----
# 1) Entry script inside the repo
MAIN_SCRIPT=/workspace/scripts/analyze_frequency.py
# e.g.:
# MAIN_SCRIPT=/workspace/scripts/evaluate_concepts/run_concept.py
# MAIN_SCRIPT=/workspace/scripts/run_concept.py
# 2) Extra arguments for your script
EXTRA_ARGS="--data-root /data/TCD/data --output-dir /workspace/results"
# ----------------------------------------------
# Run inside container
apptainer exec --nv \
 --bind "${REPO}:/workspace" \
 --bind "${DATASET_HOST}:/data/TCD/data" \
"${CONTAINER}" \
env PYTHONPATH=/workspace:$PYTHONPATH \
 python "${MAIN_SCRIPT}" ${EXTRA_ARGS}
