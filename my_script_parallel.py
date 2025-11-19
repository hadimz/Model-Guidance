import os
import sys
# Define the constant hyperparameters
model_path="models/COCO2014/vanilla_standard_attrNone_loclossNone_origNone_resnet50_lr0.0001_sll1.0_layerInput/model_checkpoint_f1_best.pt"
train_batch_size=128
eval_batch_size=64
n_epochs=10

# Define the variable hyperparameters
lambda_=[0.001]
layer=["Final"]
localization_loss_fn=["Energy_Points"]
attribution_method=["GradCam"]
feedback_type=["mask"]
similarity_threshold=[0.5, 0.99, 0.999, 97, 0.95, 0.997, 0.995]
adaptive_lambda=[False, True]
num_guiding_points=[1, 25, 10, 50, 100]
learning_rates=[1e-4]

# Create a list of all combinations of hyperparameters
experiments = []
for l in layer:
    for ll in lambda_:
        for st in similarity_threshold:
            for al in adaptive_lambda:
                for ngp in num_guiding_points:
                    for lr in learning_rates:
                        experiments.append((l, ll, st, al, ngp, lr))
print(len(experiments), "experiments in total.")
if __name__ == "__main__":
    i = int(sys.argv[1]) #get the value of the $SLURM_ARRAY_TASK_ID
    layer, loss_lambda, st, al, ngp, lr = experiments[i]
    output_file = f'logs/GradCamSparseSampling/Sim_lr_${lr}_Layer{layer}_adaptive_{al}_lambda_{loss_lambda}_threshold_{st}_NGuidingPoints_{ngp}.out'
    error_file  = f'logs/GradCamSparseSampling/Sim_lr_${lr}_Layer{layer}_adaptive_{al}_lambda_{loss_lambda}_threshold_{st}_NGuidingPoints_{ngp}.err'

    if al:
        os.system(f"time python train.py --model_backbone vanilla --dataset COCO2014 --learning_rate {lr} --train_batch_size {train_batch_size} --eval_batch_size {eval_batch_size} --total_epochs {n_epochs} --optimize_explanations --model_path {model_path} --localization_loss_lambda {loss_lambda} --adaptive_lambda {al} --layer {l} --localization_loss_fn Energy_Points --pareto --attribution_method {attribution_method[0]} --feedback_type {feedback_type[0]} --similarity_threshold {st} --num_guiding_points {ngp}") # > {output_file} 2>{error_file}")
    else:
        os.system(f"time python train.py --model_backbone vanilla --dataset COCO2014 --learning_rate {lr} --train_batch_size {train_batch_size} --eval_batch_size {eval_batch_size} --total_epochs {n_epochs} --optimize_explanations --model_path {model_path} --localization_loss_lambda {loss_lambda} --layer {l} --localization_loss_fn Energy_Points --pareto --attribution_method {attribution_method[0]} --feedback_type {feedback_type[0]} --similarity_threshold {st} --num_guiding_points {ngp}") # > {output_file} 2>{error_file}")
