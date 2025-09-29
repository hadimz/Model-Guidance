#!/bin/bash
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=16 # change this parameter to 2,4,6,... to see the effect on performance
#SBATCH --gpus=h100:1 
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --account=rrg-adurand

model_path=models/COCO2014/vanilla_standard_attrNone_loclossNone_origNone_resnet50_lr0.0001_sll1.0_layerInput/model_checkpoint_f1_best.pt
lambda=1e-4
layer=Final
localization_loss_fn=Energy_Points
attribution_method=GradCam
feedback_type=points
similarity_threshold=0.99
adaptive_lambda=False
num_guiding_points=10
train_batch_size=512
eval_batch_size=256
n_epochs=10
learning_rate=1e-4

output=logs/${attribution_method}/noSim_sbatchRun_lr_${learning_rate}_Layer${layer}_adaptive_${adaptive_lambda}_lambda_${lambda}_threshold_${similarity_threshold}_NGuidingPoints_${num_guiding_points}_epochs_${n_epochs}.out
error=logs/${attribution_method}/noSim_sbatchRun_lr_${learning_rate}_Layer${layer}_adaptive_${adaptive_lambda}_lambda_${lambda}_threshold_${similarity_threshold}_NGuidingPoints_${num_guiding_points}_epochs_${n_epochs}.err

module load python/3.10
module load scipy-stack
source ../venv/model_guidance/bin/activate

time python train.py --model_backbone "vanilla" --dataset "COCO2014" --learning_rate "$learning_rate" --train_batch_size "$train_batch_size" --eval_batch_size "$eval_batch_size" --total_epochs "$n_epochs" --optimize_explanations --model_path "$model_path" --localization_loss_lambda "$lambda" --layer "$layer" --localization_loss_fn "$localization_loss_fn" --pareto --attribution_method "$attribution_method" --feedback_type "points" --similarity_threshold "$similarity_threshold" --num_guiding_points "$num_guiding_points" # > "$output" 2>"$error" --adaptive_lambda
