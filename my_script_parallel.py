import os
import sys


lambda_=[0.001, 0.0001]
layer=["Input","Final"]
localization_loss_fn=["Energy_Points"]
attribution_method=["BCos"]
feedback_type=["points"]
similarity_threshold=[0.99, 0.97, 0.95, 0.9]
adaptive_lambda=[True, False]
num_guiding_points=[10, 50, 0]

experiments = []
for l in layer:
    for ll in lambda_:
        for st in similarity_threshold:
            for al in adaptive_lambda:
                for ngp in num_guiding_points:
                    experiments.append((l, ll, st, al, ngp))

if __name__ == "__main__":
    model_path="models/COCO2014/bcos_standard_attrNone_loclossNone_origNone_resnet50_lr0.0001_sll1.0_layerInput/model_checkpoint_f1_best.pt"
    i = int(sys.argv[1]) #get the value of the $SLURM_ARRAY_TASK_ID
    layer, loss_lambda, st, al, ngp = experiments[i]
    output_file = f'logs/BCos/sbatchRun_Layer{layer}_adaptive_{al}_lambda_{loss_lambda}_threshold_{st}_NGuidingPoints_{ngp}.out'
    error_file  = f'logs/BCos/sbatchRun_Layer{layer}_adaptive_{al}_lambda_{loss_lambda}_threshold_{st}_NGuidingPoints_{ngp}.err'

    os.system(f"time python train.py --model_backbone bcos --dataset COCO2014 --learning_rate 1e-4 --train_batch_size 24 --eval_batch_size 24 --total_epochs 10 --optimize_explanations --model_path {model_path} --localization_loss_lambda {loss_lambda} --layer {l} --localization_loss_fn Energy_Points --pareto --attribution_method BCos --adaptive_lambda {al} --feedback_type points --similarity_threshold {st} --num_guiding_points {ngp} > {output_file} 2>{error_file}")
