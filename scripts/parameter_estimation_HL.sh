#!/bin/bash

# Array job: one task per replication r = 1, ..., 30 of the HL scenario.
# Task r runs optimize_several_parameters.jl on Y_obs = y_HL[:, r, :] and saves
# its estimates to furset26/simulation_study/results/HL/estimate_rXX.jld2.
#
# Submit from the repo root with:
#   mkdir -p logs && sbatch scripts/parameter_estimation_HL.sh
#
# Rerun only some replications (e.g. failed ones) by overriding the array range:
#   sbatch --array=3,17 scripts/parameter_estimation_HL.sh
#
# Before the first submission, instantiate and precompile once on a login node so the
# 30 tasks don't all try to precompile the same packages at the same time:
#   module load julia/1.12.2 && julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
#
# NOTE: --output/--error paths are resolved by SLURM before the job runs, so the
# `logs/` directory must already exist at submission time (the mkdir -p above).

#SBATCH --job-name=param_est_HL
#SBATCH --account=share-ie-imf
#SBATCH --array=1-30
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --mail-user=theodoros.xenakis.03@gmail.com
#SBATCH --mail-type=END,FAIL

set -euo pipefail

echo "Job $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID running on $(hostname) at $(date)"

# SLURM jobs start in the submission directory, but cd explicitly to be safe.
cd "$SLURM_SUBMIT_DIR"

module purge
module load Julia/1.12.2

# Use the allocated cores for Julia threads (the finite-difference gradient is parallelized over
# threads); keep BLAS single-threaded so the threads don't compete for the same cores.
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1

# The Julia script reads the replication index from SLURM_ARRAY_TASK_ID and the scenario from SCENARIO.
export SCENARIO=HL
julia --project=. furset26/simulation_study/optimize_several_parameters.jl

echo "Task $SLURM_ARRAY_TASK_ID finished at $(date)"
