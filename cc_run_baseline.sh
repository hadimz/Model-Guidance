#!/bin/bash
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=16 # change this parameter to 2,4,6,... to see the effect on performance
#SBATCH --gpus=h100:1 
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --account=rrg-adurand

module load python/3.10
module load scipy-stack
source ../venv/model_guidance/bin/activate


time python train.py --model_backbone vanilla --dataset COCO2014 --learning_rate 1e-3 --train_batch_size 256 --total_epochs 60 > logs/Baselines/COCO2014_vanilla_lr_0.001.out 2> logs/Baselines/COCO2014_vanilla_lr_0.001.err
