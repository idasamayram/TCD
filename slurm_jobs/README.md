# TCD Pipeline — Individual Step Scripts

Each script is a self-contained SLURM job for one pipeline step.
All steps share `00_common.sh` for paths and helper functions.

## Setup

Copy the entire `steps/` folder into your `slurm_jobs/` directory:

```bash
cp -r steps/ slurm_jobs/steps/
mkdir -p logs/tcd
```

All outputs go to `results/pipeline/` by default (shared filesystem,
visible from headnode in real time). Override with:

```bash
RESULTS_DIR=/your/custom/path sbatch slurm_jobs/steps/01_crp_features.sh
```

Keep `RESULTS_DIR` consistent across all steps so they can find each
other's outputs.

---

## Dependency graph

```
01_crp_features
    ├── 02_discover_concepts
    │       ├── 03_evaluate_concepts
    │       ├── 04_psd_baseline
    │       ├── 05_freq_dft_lrp ──┐
    │       ├── 06_freq_vil_idft ─┼── 11_freq_visualizations
    │       ├── 07_freq_vil_stdft ┘
    │       └── 09_metadata
    ├── 08_layer_sweep   (independent of step 2)
    └── 10_pruning       (independent of step 2)
```

Steps 3-7 and 9 all depend on step 2.
Steps 8 and 10 only depend on step 1.
Step 11 depends on steps 5, 6, 7.

---

## Submitting

### Run everything in order (simplest):
```bash
sbatch slurm_jobs/steps/01_crp_features.sh
# wait for it to finish, then:
sbatch slurm_jobs/steps/02_discover_concepts.sh
# wait, then submit steps 3-11 as needed
```

### Run with SLURM dependencies (automated):
```bash
J1=$(sbatch --parsable slurm_jobs/steps/01_crp_features.sh)
J2=$(sbatch --parsable --dependency=afterok:$J1 slurm_jobs/steps/02_discover_concepts.sh)
J3=$(sbatch --parsable --dependency=afterok:$J2 slurm_jobs/steps/03_evaluate_concepts.sh)
J4=$(sbatch --parsable --dependency=afterok:$J2 slurm_jobs/steps/04_psd_baseline.sh)
J5=$(sbatch --parsable --dependency=afterok:$J2 slurm_jobs/steps/05_freq_dft_lrp.sh)
J6=$(sbatch --parsable --dependency=afterok:$J2 slurm_jobs/steps/06_freq_vil_idft.sh)
J7=$(sbatch --parsable --dependency=afterok:$J2 slurm_jobs/steps/07_freq_vil_stdft.sh)
J8=$(sbatch --parsable --dependency=afterok:$J1 slurm_jobs/steps/08_layer_sweep.sh)
J9=$(sbatch --parsable --dependency=afterok:$J2 slurm_jobs/steps/09_metadata.sh)
J10=$(sbatch --parsable --dependency=afterok:$J1 slurm_jobs/steps/10_pruning.sh)
J11=$(sbatch --parsable --dependency=afterok:$J5,afterok:$J6,afterok:$J7 slurm_jobs/steps/11_freq_visualizations.sh)
echo "All jobs submitted. Final job: $J11"
```

With `--dependency=afterok:$JN`, SLURM will only start a step after its
prerequisite succeeds. If a step fails, dependent steps are cancelled
automatically.

### Re-run a single step:
```bash
# e.g. re-run concept discovery with a new config, keeping existing CRP features:
sbatch slurm_jobs/steps/02_discover_concepts.sh
```

---

## Monitoring

```bash
# Watch job queue
watch squeue -u $USER

# Follow a specific job's log
tail -f logs/tcd/tcd_02_concepts_<JOBID>.out

# Check what's been saved so far
ls -lh results/pipeline/
```

---

## Step reference

| Script | GPU | Time | Input | Output |
|--------|-----|------|-------|--------|
| 01_crp_features | ✓ | 2h | data + model | crp_features/ |
| 02_discover_concepts | — | 1h | crp_features/ | variantC_conv3/ |
| 03_evaluate_concepts | ✓ | 3h | variantC_conv3/ | evaluation_conv3/ |
| 04_psd_baseline | — | 30m | variantC_conv3/ | frequency_psd/ |
| 05_freq_dft_lrp | — | 30m | crp_features/ + variantC_conv3/ | frequency_relevance_dft_lrp/ |
| 06_freq_vil_idft | — | 30m | crp_features/ + variantC_conv3/ | frequency_relevance_vil_idft/ |
| 07_freq_vil_stdft | — | 1h | crp_features/ + variantC_conv3/ | frequency_relevance_vil_stdft/ |
| 08_layer_sweep | — | 2h | crp_features/ | layer_sweep/ |
| 09_metadata | — | 30m | variantC_conv3/ | metadata/ |
| 10_pruning | — | 1h | crp_features/ | pruning_relevance/ |
| 11_freq_visualizations | — | 20m | freq_relevance_*/  | frequency_visualizations/ |
