#!/bin/bash
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=16 # change this parameter to 2,4,6,... to see the effect on performance
#SBATCH --gpus=a100:1 
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --account=def-adurand

model_path=models/COCO2014/bcos_standard_attrNone_loclossNone_origNone_resnet50_lr0.0001_sll1.0_layerInput/model_checkpoint_f1_best.pt
lambda=0.001
layer=Final
localization_loss_fn="Energy_Points"
attribution_method=BCos
feedback_type=points
similarity_threshold=0.99
adaptive_lambda=True
num_guiding_points=10
train_batch_size=24
eval_batch_size=24
# Disable verbose printing of training progress for sbatch jobs
disable_verbose=False

output=logs/${attribution_method}/sbatchRun_Layer${layer}_adaptive_${adaptive_lambda}_lambda_${lambda}_threshold_${similarity_threshold}_NGuidingPoints_${num_guiding_points}_%N-%j.out
error=logs/${attribution_method}/sbatchRun_Layer${layer}_adaptive_${adaptive_lambda}_lambda_${lambda}_threshold_${similarity_threshold}_NGuidingPoints_${num_guiding_points}_%N-%j.err

module load python/3.10
module load scipy-stack
source ../venv/model_guidance/bin/activate

time python train.py --model_backbone "bcos" --dataset "COCO2014" --learning_rate "1e-4" --train_batch_size "$train_batch_size" --eval_batch_size "$eval_batch_size" --total_epochs "10" --optimize_explanations --model_path "$model_path" --localization_loss_lambda "$lambda" --layer "$layer" --localization_loss_fn "$localization_loss_fn" --pareto --attribution_method "BCos" --adaptive_lambda "$adaptive_lambda" --feedback_type "points" --similarity_threshold "$similarity_threshold" --num_guiding_points "$num_guiding_points" --disable_verbose "$disable_verbose" > "$output" 2>"$error"
