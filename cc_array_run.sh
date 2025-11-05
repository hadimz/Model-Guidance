#!/bin/bash
#SBATCH --mail-user=hadimoazen@ymail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=4
#SBATCH --gpus=h100_2.20
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-59
#SBATCH --account=rrg-adurand
#SBATCH --output=logs/out/%x-%j.out


module load python/3.10
module load scipy-stack
source ../venv/model_guidance/bin/activate

time python my_script_parallel.py $SLURM_ARRAY_TASK_ID
