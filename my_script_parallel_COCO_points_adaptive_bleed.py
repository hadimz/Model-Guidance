import os
import sys
# Define the constant hyperparameters

train_batch_size=256
eval_batch_size=128
n_epochs=10

# Define the variable hyperparameters
lambda_=[1e-3, 1e-4, 5e-4, 5e-3]
layer=["Final"]
localization_loss_fn=["Energy_Points"]
feedback_type=["points_adaptive"]
similarity_threshold=[0.99, 0.999, 0.997, 0.995, 0.97]
adaptive_lambda=[False, True]
num_guiding_points=[100]
learning_rates=[1e-4]
adaptive_threshold = [ 0.75, 0.5, 0.25, 0.0]
backbone =[
    # ("vanilla","GradCam","models/COCO2014/vanilla_standard_attrNone_loclossNone_origNone_resnet50_lr0.0001_sll1.0_layerInput/model_checkpoint_f1_best.pt"),
    # ("bcos"   ,"BCos"   ,"models/COCO2014/bcos_standard_attrNone_loclossNone_origNone_resnet50_lr0.0001_sll1.0_layerInput/model_checkpoint_f1_best.pt"),
    ("xdnn"   ,"IxG"    ,"models/COCO2014/xdnn_standard_attrNone_loclossNone_origNone_resnet50_lr1e-05_sll1.0_layerInput/model_checkpoint_f1_best.pt"),
    ]
# Create a list of all combinations of hyperparameters
experiments = []
for l in layer:
    for bb in backbone:
        for ll in lambda_:
            for ftype in feedback_type:
                for locloss in localization_loss_fn:
                    for lr in learning_rates:
                        for ngp in num_guiding_points:
                            for threshold in similarity_threshold:
                                for al in adaptive_lambda:
                                    for at in adaptive_threshold:
                                        experiments.append((l, ll, bb, ftype, locloss, lr, ngp, threshold, al, at))
print(len(experiments), "experiments in total.")

if __name__ == "__main__":
    i = int(sys.argv[1]) #get the value of the $SLURM_ARRAY_TASK_ID
    layer, loss_lambda, (backbone, attribution_method, model_path), ftype, locloss, lr, ngp, threshold, al, at = experiments[i]
    output_file = f'logs/COCO_Final/points_adaptive/bleed/{backbone}/{backbone}_COCO_lr_{lr}_AttMethod_{attribution_method}_feedback_type_{ftype}_localization_loss_{locloss}_Lambda_{loss_lambda}_numGuidingPoints_{ngp}_AdaptiveLambda_{al}_SimilarityThreshold_{threshold}_adaptive_threshold_{at}_seed_0.out'
    error_file  = f'logs/COCO_Final/points_adaptive/bleed/{backbone}/errors/{backbone}_COCO_lr_{lr}_AttMethod_{attribution_method}_feedback_type_{ftype}_localization_loss_{locloss}_Lambda_{loss_lambda}_numGuidingPoints_{ngp}_AdaptiveLambda_{al}_SimilarityThreshold_{threshold}_adaptive_threshold_{at}_seed_0.err'
    
    if al:
        os.system(f"time python train_fir_adaptive.py --save_path /scratch/hadimz/ --model_backbone {backbone} --dataset COCO2014 --learning_rate {lr} --train_batch_size {train_batch_size} --eval_batch_size {eval_batch_size} --total_epochs {n_epochs} --optimize_explanations --model_path {model_path} --localization_loss_lambda {loss_lambda} --layer {l} --localization_loss_fn {locloss} --pareto --attribution_method {attribution_method} --feedback_type {ftype} --num_guiding_points {ngp} --similarity_threshold {threshold} --adaptive_points_threshold {at} --adaptive_lambda True > {output_file} 2>{error_file}")
    else:
        os.system(f"time python train_fir_adaptive.py --save_path /scratch/hadimz/ --model_backbone {backbone} --dataset COCO2014 --learning_rate {lr} --train_batch_size {train_batch_size} --eval_batch_size {eval_batch_size} --total_epochs {n_epochs} --optimize_explanations --model_path {model_path} --localization_loss_lambda {loss_lambda} --layer {l} --localization_loss_fn {locloss} --pareto --attribution_method {attribution_method} --feedback_type {ftype} --num_guiding_points {ngp} --similarity_threshold {threshold} --adaptive_points_threshold {at} > {output_file} 2>{error_file}")
