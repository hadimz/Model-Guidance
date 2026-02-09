import os
import sys
# Define the constant hyperparameters
model_path="models/COCO2014/vanilla_standard_attrNone_loclossNone_origNone_resnet50_lr0.0001_sll1.0_layerInput/model_checkpoint_f1_best.pt"
train_batch_size=64
eval_batch_size=32
n_epochs=10

# Define the variable hyperparameters
lambda_=[1e-4, 1e-3, 5e-4, 5e-3, 5e-5]
layer=["Final"]
localization_loss_fn=["L1", "PPCE", "Energy"]
backbone =[
    ("vanilla","GradCam","models/COCO2014/vanilla_standard_attrNone_loclossNone_origNone_resnet50_lr0.0001_sll1.0_layerInput/model_checkpoint_f1_best.pt"),
    ("bcos"   ,"BCos"   ,"models/COCO2014/bcos_standard_attrNone_loclossNone_origNone_resnet50_lr0.0001_sll1.0_layerInput/model_checkpoint_f1_best.pt"),
    ("xdnn"   ,"IxG"    ,"models/COCO2014/xdnn_standard_attrNone_loclossNone_origNone_resnet50_lr1e-05_sll1.0_layerInput/model_checkpoint_f1_best.pt"),
    ]
feedback_type=["mask", "bbox"]
learning_rates=[1e-4]

# Create a list of all combinations of hyperparameters
experiments = []
for l in layer:
    for bb in backbone:
        for ll in lambda_:
            for ftype in feedback_type:
                for locloss in localization_loss_fn:
                    for lr in learning_rates:
                        experiments.append((l, ll, bb, ftype, locloss, lr))
print(len(experiments), "experiments in total.")
if __name__ == "__main__":
    i = int(sys.argv[1]) #get the value of the $SLURM_ARRAY_TASK_ID
    layer, loss_lambda, (backbone, attribution_method, model_path), ftype, locloss, lr = experiments[i]
    
    output_file = f'logs/COCO_Final/GuidedBaselines/{backbone}/{backbone}_COCO_lr_{lr}_AttMethod_{attribution_method}_feedback_type_{ftype}_localization_loss_{locloss}_Lambda_{loss_lambda}_seed_0.out'
    error_file  = f'logs/COCO_Final/GuidedBaselines/{backbone}/errors/{backbone}_COCO_lr_{lr}_AttMethod_{attribution_method}_feedback_type_{ftype}_localization_loss_{locloss}_Lambda_{loss_lambda}_seed_0.err'

    os.system(f"time python train_fir.py --save_path /scratch/hadimz/ --model_backbone {backbone} --dataset COCO2014 --learning_rate {lr} --train_batch_size {train_batch_size} --eval_batch_size {eval_batch_size} --total_epochs {n_epochs} --optimize_explanations --model_path {model_path} --localization_loss_lambda {loss_lambda} --layer {l} --localization_loss_fn {locloss} --pareto --attribution_method {attribution_method} --feedback_type {ftype} > {output_file} 2>{error_file}")
