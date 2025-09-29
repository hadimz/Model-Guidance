#!/bin/bash
#SBATCH --mail-user=hadimoazen@ymail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=16 # change this parameter to 2,4,6,... to see the effect on performance
#SBATCH --gpus=h100:1 
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --array=0-863
#SBATCH --account=rrg-adurand


module load python/3.10
module load scipy-stack
source ../venv/model_guidance/bin/activate

time python my_script_parallel.py $SLURM_ARRAY_TASK_ID
