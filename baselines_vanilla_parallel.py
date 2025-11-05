import os
import sys
# Define the constant hyperparameters
model_path="models/COCO2014/vanilla_standard_attrNone_loclossNone_origNone_resnet50_lr0.0001_sll1.0_layerInput/model_checkpoint_f1_best.pt"
train_batch_size=128
eval_batch_size=64
n_epochs=10

# Define the variable hyperparameters
lambdas_=[0.001, 0.0001, 0.005, 0.0005, 0.01]
layers=["Final"]
attribution_methods=["GradCam", "IxG"]
localization_loss_fns=["RRR", "Energy", "L1", "PPCE"]
feedback_types=["mask", "bbox", "points"]
similarity_thresholds=[1.] # 0.5, 0.99, 0.999, 97, 0.95, 0.997, 0.995]
# adaptive_lambda=[False, True]
# num_guiding_points=[50]#, 25, 1, 10, 100]
ngp = 50
learning_rates=[1e-4]

# Create a list of all combinations of hyperparameters
experiments = []
for loc_lambda in lambdas_:
    for att_method in attribution_methods:
        for loc_loss in localization_loss_fns:
            for ftype in feedback_types:
                for lr in learning_rates:
                    experiments.append((loc_lambda, att_method, loc_loss, ftype, lr))

print(len(experiments), "experiments in total.")
if __name__ == "__main__":
    i = int(sys.argv[1]) #get the value of the $SLURM_ARRAY_TASK_ID
    loc_lambda, att_method, loc_loss, ftype, lr = experiments[i]
    print('running vanilla experiment:', i, loc_lambda, att_method, loc_loss, ftype, lr)
    if ftype == "points":
        output_file = f'logs/GuidedBaselines/out/vanilla_COCO_AttMethod_{att_method}_Layer_Final_feedback_type_{ftype}_localization_loss_{loc_loss}_lambda_{loc_lambda}_NGuidingPoints_{50}_threshold_{1.0}_seed_0.out'
        error_file  = f'logs/GuidedBaselines/err/vanilla_COCO_AttMethod_{att_method}_Layer_Final_feedback_type_{ftype}_localization_loss_{loc_loss}_lambda_{loc_lambda}_NGuidingPoints_{50}_threshold_{1.0}_seed_0.err'
        os.system(f"time python train.py --model_backbone vanilla --dataset COCO2014 --learning_rate {lr} --train_batch_size {train_batch_size} --eval_batch_size {eval_batch_size} --total_epochs 10 --optimize_explanations --model_path {model_path} --localization_loss_lambda {loc_lambda} --layer Final --localization_loss_fn {loc_loss} --pareto --attribution_method {att_method} --feedback_type {ftype} --similarity_threshold 1.0 --num_guiding_points {ngp}  > {output_file} 2> {error_file}")
    else:
        output_file = f'logs/GuidedBaselines/out/vanilla_COCO_AttMethod_{att_method}_Layer_Final_feedback_type_{ftype}_localization_loss_{loc_loss}_lambda_{loc_lambda}_seed_0.out'
        error_file  = f'logs/GuidedBaselines/err/vanilla_COCO_AttMethod_{att_method}_Layer_Final_feedback_type_{ftype}_localization_loss_{loc_loss}_lambda_{loc_lambda}_seed_0.err'
        os.system(f"time python train.py --model_backbone vanilla --dataset COCO2014 --learning_rate {lr} --train_batch_size {train_batch_size} --eval_batch_size {eval_batch_size} --total_epochs 10 --optimize_explanations --model_path {model_path} --localization_loss_lambda {loc_lambda} --layer Final --localization_loss_fn {loc_loss} --pareto --attribution_method {att_method} --feedback_type {ftype} > {output_file} 2> {error_file}")
