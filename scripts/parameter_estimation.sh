#!/bin/bash

# Submit from the repo root with:
#   mkdir -p logs && sbatch src/scripts/ccdb/run_train_main.sh --model-id 3 --epochs 100 --lr 5e-4
#
# Everything after the script path is forwarded to train_main.jl's ArgParse CLI
# (run `julia src/models/train_main.jl --help` locally to see all flags).
#
# NOTE: --output/--error paths are resolved by SLURM before the job runs, so the
# `logs/` directory must already exist at submission time (the mkdir -p above).

#SBATCH --job-name=train_main
#SBATCH --account=share-ie-imf    
#SBATCH --time=08:00:00                
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-user=theodoros.xenakis.03@gmail.com
#SBATCH --mail-type=END,FAIL

set -euo pipefail

echo "Job $SLURM_JOB_ID running on $(hostname) at $(date)"

# SLURM jobs start in the submission directory, but cd explicitly to be safe.
cd "$SLURM_SUBMIT_DIR"

module purge
module load julia/1.12.2

julia --project=. furset26/simulation_study/optimize_several_parameters.jl

echo "Job $SLURM_JOB_ID finished at $(date)"
