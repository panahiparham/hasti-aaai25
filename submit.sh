#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu="4G"
#SBATCH --time=0:59:00
#SBATCH --cpus-per-task=1
#SBATCH --account=aip-amw8

mkdir -p $SLURM_TMPDIR/$SLURM_JOB_ID

module --force purge
module load StdEnv/2023

cd {cwd}

module load python/3.12
uv venv $SLURM_TMPDIR/$SLURM_JOB_ID/.venv --cache-dir $SLURM_TMPDIR/$SLURM_JOB_ID/.cache --python 3.12
source $SLURM_TMPDIR/$SLURM_JOB_ID/.venv/bin/activate

uv pip install . --cache-dir $SLURM_TMPDIR/$SLURM_JOB_ID/.cache

uv run run.py
