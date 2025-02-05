#!/bin/bash
#SBATCH --nodes=1          # Use 1 Node     (Unless code is multi-node parallelized)
#SBATCH --ntasks=1
#SBATCH --account=fconrad0
#SBATCH --time=23:20:00
#SBATCH --cpus-per-task=3
#SBATCH -o slurm-%j.out-%N
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:4
#SBATCH --mem=180000m
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maolee@umich.edu   # Your email address has to be set accordingly
#SBATCH --job-name=censusSocialMedia        # the job's name you want to be used

module load python3.11-anaconda clang gcc/13.2.0 openmpi cuda

export FILENAME=/nfs/turbo/isr-fconrad1/Mao/projects/information-diffusion/src/eval_topmod/eval_topic_model.py
srun python $FILENAME > $SLURM_JOBID.out

echo "End of program at `date`"
