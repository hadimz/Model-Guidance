#!/bin/bash
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=4 # change this parameter to 2,4,6,... to see the effect on performance
#SBATCH --gpus=h100_2.20
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --account=rrg-adurand

module load python/3.10
module load scipy-stack
source ../venv/model_guidance/bin/activate


time python train.py --model_backbone xdnn --dataset COCO2014 --learning_rate 1e-4 --train_batch_size 128 --eval_batch_size 64 --total_epochs 60 > logs/Baselines/COCO2014_xdnn_lr_0.0001.out 2> logs/Baselines/COCO2014_xdnn_lr_0.0001.err
