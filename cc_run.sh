#!/bin/bash
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=4 # change this parameter to 2,4,6,... to see the effect on performance
#SBATCH --gpus=h100_2.20:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --account=rrg-adurand

model_path=models/COCO2014/vanilla_standard_attrNone_loclossNone_origNone_resnet50_lr0.0001_sll1.0_layerInput/model_checkpoint_f1_best.pt
lambda=1e-3
layer=Final
localization_loss_fn=Energy_Points
attribution_method=GradCam
feedback_type=points_adaptive
similarity_threshold=0.99
adaptive_lambda=True
num_guiding_points=10
train_batch_size=128
eval_batch_size=64
n_epochs=1
learning_rate=1e-4

output=logs/points_adaptive/${feedback_type}_noProb_threshold_0.85_sbatchRun_lr_${learning_rate}_Layer${layer}_adaptive_${adaptive_lambda}_lambda_${lambda}_threshold_${similarity_threshold}_NGuidingPoints_${num_guiding_points}_epochs_${n_epochs}.out
error=logs/points_adaptive/${feedback_type}_noProb_threshold_0.85_sbatchRun_lr_${learning_rate}_Layer${layer}_adaptive_${adaptive_lambda}_lambda_${lambda}_threshold_${similarity_threshold}_NGuidingPoints_${num_guiding_points}_epochs_${n_epochs}.err

module load python/3.10
module load scipy-stack
source ../venv/model_guidance/bin/activate

time python train.py --model_backbone "vanilla" --dataset "COCO2014" --adaptive_lambda "$adaptive_lambda" --learning_rate "$learning_rate" --train_batch_size "$train_batch_size" --eval_batch_size "$eval_batch_size" --total_epochs "$n_epochs" --optimize_explanations --model_path "$model_path" --localization_loss_lambda "$lambda" --layer "$layer" --localization_loss_fn "$localization_loss_fn" --pareto --attribution_method "$attribution_method" --feedback_type "$feedback_type" --similarity_threshold "$similarity_threshold" --num_guiding_points "$num_guiding_points" > "$output" 2>"$error"
