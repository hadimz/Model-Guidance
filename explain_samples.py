import torch
import torchvision
import argparse
from tqdm import tqdm
import datasets
import argparse
import torch.utils.tensorboard
import utils
import copy
import losses
import metrics
import bcos.models
import model_activators
import attribution_methods
import hubconf
import bcos
import bcos.modules
import bcos.data
import bcos.data.transforms as bcostransforms
import fixup_resnet

import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import os

import matplotlib.pyplot as plt
import torchvision.transforms.functional as ff
import pickle

utils.set_seed(0)
batch_size=2
layer = 'Final'
data_path = "datasets/"
num_classes_dict = {"VOC2007": 20, "COCO2014":  80}
for dataset in ['COCO2014']:#, 'VOC2007']:
    num_classes = num_classes_dict[dataset]
    root = os.path.join(data_path, dataset, "processed")
    for model_backbone in ['bcos', 'xdnn', 'vanilla']:
        gts = {}
        is_bcos = (model_backbone == "bcos")
        is_xdnn = (model_backbone == "xdnn")
        is_vanilla = (model_backbone == "vanilla")
        
        if is_bcos:
            transformer = bcostransforms.AddInverse(dim=0)
        else:
            transformer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            invTrans = torchvision.transforms.Compose([ torchvision.transforms.Normalize(mean = [ 0., 0., 0. ],
                                                std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                            torchvision.transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                std = [ 1., 1., 1. ]),])
        
        print(f"Loading guiding points validation dataset from {root}")
        test_data = datasets.VOCDetectParsed(
            root=root, image_set="test_GuidingPoints", transform=transformer)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=datasets.VOCDetectParsed.collate_fn)
        for guidance in ['unguided', 'L1', 'PPCE', 'Energy', 'Points']:
            if is_bcos:
                print('loading bcos resnet50 model')
                model = hubconf.resnet50(pretrained=True)
                model[0].fc = bcos.modules.bcosconv2d.BcosConv2d(
                            in_channels=model[0].fc.in_channels, out_channels=num_classes)
                layer_dict = {"Input": None, "Mid1": 3,
                            "Mid2": 4, "Mid3": 5, "Final": 6}
            elif is_xdnn:
                model = fixup_resnet.xfixup_resnet50()
                imagenet_checkpoint = torch.load(os.path.join("weights/xdnn/xfixup_resnet50_model_best.pth.tar"))
                imagenet_state_dict = utils.remove_module(
                    imagenet_checkpoint["state_dict"])
                model.load_state_dict(imagenet_state_dict)
                model.fc = torch.nn.Linear(
                    in_features=model.fc.in_features, out_features=num_classes)
                layer_dict = {"Input": None, "Mid1": 3,
                            "Mid2": 4, "Mid3": 5, "Final": 6}
            elif is_vanilla:
                model = torchvision.models.resnet50(
                    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
                model.fc = torch.nn.Linear(
                        in_features=model.fc.in_features, out_features=num_classes)
                layer_dict = {"Input": None, "Mid1": 4,
                            "Mid2": 5, "Mid3": 6, "Final": 7}
            else:
                raise NotImplementedError

            layer_idx = layer_dict[layer]
            
            if guidance=='unguided':
                if is_vanilla:
                    checkpoint = torch.load('models/COCO2014/vanilla_standard_attrNone_loclossNone_origNone_resnet50_lr0.0001_sll1.0_layerInput/model_checkpoint_f1_best.pt')
                    model.load_state_dict(checkpoint["model"])
                elif is_bcos:
                    checkpoint = torch.load('models/COCO2014/bcos_standard_attrNone_loclossNone_origNone_resnet50_lr0.0001_sll1.0_layerInput/model_checkpoint_f1_best.pt')
                    model.load_state_dict(checkpoint["model"])
                elif is_xdnn:
                    checkpoint = torch.load('models/COCO2014/xdnn_standard_attrNone_loclossNone_origNone_resnet50_lr1e-05_sll1.0_layerInput/model_checkpoint_f1_best.pt')
                    model.load_state_dict(checkpoint["model"])
            elif guidance=='Energy':
                if is_vanilla:
                    checkpoint = torch.load('checkpoints/COCO2014/vanilla_finetunedobjlocmaskpareto_attrGradCam_loclossEnergy_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.0001_layerFinal_feedbackTypemask/model_checkpoint_f1_best.pt')
                    model.load_state_dict(checkpoint["model"])
                elif is_bcos:
                    checkpoint = torch.load('checkpoints/COCO2014/bcos_finetunedobjlocmaskpareto_attrBCos_loclossEnergy_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.0001_layerFinal_feedbackTypemask/model_checkpoint_f1_best.pt')
                    model.load_state_dict(checkpoint["model"])
                elif is_xdnn:
                    checkpoint = torch.load('checkpoints/COCO2014/xdnn_finetunedobjlocmaskpareto_attrGradCam_loclossEnergy_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.0001_layerFinal_feedbackTypemask/model_checkpoint_f1_best.pt')
                    model.load_state_dict(checkpoint["model"])
            elif guidance=='Points':
                if is_vanilla:
                    checkpoint = torch.load('checkpoints/COCO2014/vanilla_finetunedobjlocpointspareto_attrGradCam_loclossEnergy_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.001_layerFinal_feedbackTypepoints/model_checkpoint_f1_best.pt')
                    model.load_state_dict(checkpoint["model"])
                elif is_bcos:
                    checkpoint = torch.load('checkpoints/COCO2014/bcos_finetunedobjlocpointspareto_attrBCos_loclossEnergy_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.001_layerFinal_feedbackTypepoints/model_checkpoint_f1_best.pt')
                    model.load_state_dict(checkpoint["model"])
                elif is_xdnn:
                    checkpoint = torch.load('checkpoints/COCO2014/xdnn_finetunedobjlocpointspareto_attrIxG_loclossEnergy_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.001_layerFinal_feedbackTypepoints/model_checkpoint_f1_best.pt')
                    model.load_state_dict(checkpoint["model"])

            model = model.cuda()
            model.eval()
            
            
            interpolate = True if layer_idx is not None else False
            attributor = attribution_methods.get_attributor(
                    model, "GradCam", True, False, interpolate, (224, 224), batch_mode=True)       
            model_activator = model_activators.ResNetModelActivator(
            model=model, layer=layer_idx, is_bcos=is_bcos)

            for batch_idx, (train_X, train_y, train_bbs, train_masks, guiding_points, indices) in enumerate(test_loader):
                train_X.requires_grad = True
                train_X = train_X.cuda()
                train_y = train_y.cuda()
                logits, features, acts = model_activator(train_X)
                try:
                    gt_classes = gts[batch_idx]
                    print('used target classes from last iteration!')
                except:
                    gt_classes = utils.get_random_optimization_targets(train_y)
                    gts[batch_idx] = gt_classes
                    print('updated target classes in the gts dict!')

                attributions = attributor(features, logits, classes=gt_classes).squeeze(1)
                print(f'explanations shape: {attributions.shape}')
                for id in range(len(train_X)):
                    plt.figure(figsize=(15,5))
                    plt.subplot(1,3,1)
                    # print(f'image: {train_X[id].shape}')
                    if is_bcos:
                        plt.imshow(train_X[id][:3,:,:].moveaxis(0,-1).detach().cpu())
                    else:
                        plt.imshow(torch.clamp(invTrans(train_X[id]), 0,1).moveaxis(0,-1).detach().cpu(), cmap='jet')
                    plt.subplot(1,3,2)
                    # print(f'target: {gt_classes}')
                    # print(f'mask: {train_masks[id]}')
                    plt.imshow(torch.where(train_masks[id]==gt_classes[id].cpu()+1, 1., 0.))
                    plt.subplot(1,3,3)
                    plt.imshow(attributions[id].detach().cpu())
                    plt.savefig(f'figures/{model_backbone}/{dataset}_{model_backbone}_{guidance}_batch_{batch_idx}_image_{id}_class_{gt_classes[id]}.png')
                    plt.close('all')
                break



