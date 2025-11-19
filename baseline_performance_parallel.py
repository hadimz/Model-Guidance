import os
import sys

backbones = {'bcos': ['models/COCO2014/bcos_standard_attrNone_loclossNone_origNone_resnet50_lr0.0001_sll1.0_layerInput/model_checkpoint_f1_best.pt', ["BCos", "GradCam"]],
            'vanilla': ['models/COCO2014/vanilla_standard_attrNone_loclossNone_origNone_resnet50_lr0.0001_sll1.0_layerInput/model_checkpoint_f1_best.pt', ["GradCam", "IxG"]],
            'xdnn':['models/COCO2014/xdnn_standard_attrNone_loclossNone_origNone_resnet50_lr1e-05_sll1.0_layerInput/model_checkpoint_f1_best.pt', ["GradCam", "IxG"]]
            }
experiments = []
for backbone, params in backbones.items():
    for exp_method in params[1]:
        experiments.append((backbone, params[0], exp_method))

if __name__ == "__main__":
    i = int(sys.argv[1]) #get the value of the $SLURM_ARRAY_TASK_ID
    backbone, model_path, att_method = experiments[i]
    os.system(f"time python baseline_performance.py --model_backbone {backbone} --dataset COCO2014 --train_batch_size 4 --eval_batch_size 16 --pareto --attribution_method={att_method} --model_path={model_path} > logs/Baselines/baseline_performance_{backbone}_{att_method}.out")
    