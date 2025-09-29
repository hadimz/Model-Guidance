#!/bin/bash
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=16 # change this parameter to 2,4,6,... to see the effect on performance
#SBATCH --gpus=h100:1 
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --account=rrg-adurand

module load python/3.10
module load scipy-stack
source ../venv/model_guidance/bin/activate


time python train.py --model_backbone vanilla --dataset COCO2014 --learning_rate 1e-4 --train_batch_size 512 --total_epochs 10 --optimize_explanations --model_path models/COCO2014/vanilla_standard_attrNone_loclossNone_origNone_resnet50_lr0.0001_sll1.0_layerInput/model_checkpoint_f1_best.pt --localization_loss_lambda 1e-3 --layer Final --localization_loss_fn Energy --pareto --attribution_method GradCam --feedback_type bbox  > logs/Vanilla/COCO2014_vanilla_lr_0.0001_opt_expl_GradCAM_batch_512.out 2> logs/Vanilla/out.err
