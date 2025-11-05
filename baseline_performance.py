import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import os
import argparse
import torchvision
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
import fixup_resnet

import matplotlib.pyplot as plt
import torchvision.transforms.functional as ff


class_names = {0: '__background__',
 1: 'person',
 2: 'bicycle',
 3: 'car',
 4: 'motorcycle',
 5: 'airplane',
 6: 'bus',
 7: 'train',
 8: 'truck',
 9: 'boat',
 10: 'traffic light',
 11: 'fire hydrant',
 12: 'stop sign',
 13: 'parking meter',
 14: 'bench',
 15: 'bird',
 16: 'cat',
 17: 'dog',
 18: 'horse',
 19: 'sheep',
 20: 'cow',
 21: 'elephant',
 22: 'bear',
 23: 'zebra',
 24: 'giraffe',
 25: 'backpack',
 26: 'umbrella',
 27: 'handbag',
 28: 'tie',
 29: 'suitcase',
 30: 'frisbee',
 31: 'skis',
 32: 'snowboard',
 33: 'sports ball',
 34: 'kite',
 35: 'baseball bat',
 36: 'baseball glove',
 37: 'skateboard',
 38: 'surfboard',
 39: 'tennis racket',
 40: 'bottle',
 41: 'wine glass',
 42: 'cup',
 43: 'fork',
 44: 'knife',
 45: 'spoon',
 46: 'bowl',
 47: 'banana',
 48: 'apple',
 49: 'sandwich',
 50: 'orange',
 51: 'broccoli',
 52: 'carrot',
 53: 'hot dog',
 54: 'pizza',
 55: 'donut',
 56: 'cake',
 57: 'chair',
 58: 'couch',
 59: 'potted plant',
 60: 'bed',
 61: 'dining table',
 62: 'toilet',
 63: 'tv',
 64: 'laptop',
 65: 'mouse',
 66: 'remote',
 67: 'keyboard',
 68: 'cell phone',
 69: 'microwave',
 70: 'oven',
 71: 'toaster',
 72: 'sink',
 73: 'refrigerator',
 74: 'book',
 75: 'clock',
 76: 'vase',
 77: 'scissors',
 78: 'teddy bear',
 79: 'hair drier',
 80: 'toothbrush'}

def eval_model_binary_mask(model, attributor, loader, num_batches, num_classes, loss_fn, writer=None, epoch=None):
    """
    This function evaluates the model on given data and computes classification metrics along with explanation agreement with class-specific segmentation masks used in lieu of "ground truth" expert feedback.
    """
    model.eval()
    f1_metric = metrics.MultiLabelMetrics(
        num_classes=num_classes, threshold=0.0)
    bb_metric = metrics.BinaryMaskEnergyMultiple()
    iou_metric = metrics.BinaryMaskIoUMultiple()

    total_loss = 0

    for batch_idx, (test_X, test_y, _, test_masks, _, _) in enumerate(loader):
        test_X.requires_grad = True
        test_X = test_X.cuda()
        test_y = test_y.cuda()
        logits, features, _ = model(test_X)
        loss = loss_fn(logits, test_y).detach()
        total_loss += loss
        f1_metric.update(logits, test_y)

        if attributor:
            for img_idx in range(len(test_X)):
                class_target = torch.where(test_y[img_idx] == 1)[0]
                for pred_idx, pred in enumerate(class_target):
                    attributions = attributor(
                        features, logits, pred, img_idx).detach().squeeze(0).squeeze(0)
                    # bb_list = utils.filter_bbs(test_bbs[img_idx], pred)
                    import matplotlib.pyplot as plt
                    bb_metric.update(attributions, test_masks[img_idx].cuda() == pred+1)
                    iou_metric.update(attributions, test_masks[img_idx].cuda() == pred+1)

    metric_vals = f1_metric.compute()
    if attributor:
        bb_metric_vals = bb_metric.compute()
        iou_metric_vals = iou_metric.compute()
        metric_vals["BB-Loc"] = bb_metric_vals
        metric_vals["BB-IoU"] = iou_metric_vals
    metric_vals["Average-Loss"] = total_loss.item()/num_batches        
    print(f"Validation Metrics: {metric_vals}")
    model.train()
    if writer is not None:
        writer.add_scalar("val_loss", total_loss.item()/num_batches, epoch)
        writer.add_scalar("accuracy", metric_vals["Accuracy"], epoch)
        writer.add_scalar("precision", metric_vals["Precision"], epoch)
        writer.add_scalar("recall", metric_vals["Recall"], epoch)
        writer.add_scalar("fscore", metric_vals["F-Score"], epoch)
        if attributor:
            writer.add_scalar("bbloc", metric_vals["BB-Loc"], epoch)
            writer.add_scalar("bbiou", metric_vals["BB-IoU"], epoch)
    return metric_vals


def eval_model(model, attributor, loader, num_batches, num_classes, loss_fn, writer=None, epoch=None):
    model.eval()
    f1_metric = metrics.MultiLabelMetrics(
        num_classes=num_classes, threshold=0.0)
    bb_metric = metrics.BoundingBoxEnergyMultiple()
    iou_metric = metrics.BoundingBoxIoUMultiple()

    total_loss = 0

    for batch_idx, (test_X, test_y, test_bbs, test_masks, _, _) in enumerate(loader):
        test_X.requires_grad = True
        test_X = test_X.cuda()
        test_y = test_y.cuda()
        logits, features, _ = model(test_X)
        loss = loss_fn(logits, test_y).detach()
        total_loss += loss
        f1_metric.update(logits, test_y)

        if attributor:
            for img_idx in range(len(test_X)):
                class_target = torch.where(test_y[img_idx] == 1)[0]
                for pred_idx, pred in enumerate(class_target):
                    attributions = attributor(
                        features, logits, pred, img_idx).detach().squeeze(0).squeeze(0)
                    bb_list = utils.filter_bbs(test_bbs[img_idx], pred)
                    bb_metric.update(attributions, bb_list)
                    iou_metric.update(attributions, bb_list)

    metric_vals = f1_metric.compute()
    if attributor:
        bb_metric_vals = bb_metric.compute()
        iou_metric_vals = iou_metric.compute()
        metric_vals["BB-Loc"] = bb_metric_vals
        metric_vals["BB-IoU"] = iou_metric_vals
    metric_vals["Average-Loss"] = total_loss.item()/num_batches        
    print(f"Validation Metrics: {metric_vals}")
    model.train()
    if writer is not None:
        writer.add_scalar("val_loss", total_loss.item()/num_batches, epoch)
        writer.add_scalar("accuracy", metric_vals["Accuracy"], epoch)
        writer.add_scalar("precision", metric_vals["Precision"], epoch)
        writer.add_scalar("recall", metric_vals["Recall"], epoch)
        writer.add_scalar("fscore", metric_vals["F-Score"], epoch)
        if attributor:
            writer.add_scalar("bbloc", metric_vals["BB-Loc"], epoch)
            writer.add_scalar("bbiou", metric_vals["BB-IoU"], epoch)
    return metric_vals


def main(args):
    print(f'Backbone: {args.model_backbone}, Attribution_method: {args.attribution_method}, Layer: {args.layer}, Lambda: {args.localization_loss_lambda}, localization loss: {args.localization_loss_fn}, feedback_type: {args.feedback_type}')
    if args.feedback_type == 'points':
        print('AdaptiveLambda: {args.adaptive_lambda}, , SimilarityThreshold: {args.similarity_threshold}, NumGuidingPoints: {args.num_guiding_points}.')
    
    utils.set_seed(args.seed)

    num_classes_dict = {"VOC2007": 20, "COCO2014":  80}
    num_classes = num_classes_dict[args.dataset]

    is_bcos = (args.model_backbone == "bcos")
    is_xdnn = (args.model_backbone == "xdnn")
    is_vanilla = (args.model_backbone == "vanilla")
        

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

    layer_idx = layer_dict[args.layer]

    if args.model_path is not None:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint["model"])

    model = model.cuda()
    # model.train()


    # orig_name = os.path.basename(
    #     args.model_path) if args.model_path else str(None)

    # model_prefix = args.model_backbone

    # optimize_explanation_str = "finetunedobjloc" if args.optimize_explanations else "standard"
    # optimize_explanation_str += args.feedback_type if args.feedback_type else ""
    # optimize_explanation_str += "pareto" if args.pareto else ""
    # optimize_explanation_str += "limited" if args.annotated_fraction < 1.0 else ""
    # optimize_explanation_str += "dilated" if args.box_dilation_percentage > 0 else ""

    # out_name = model_prefix + "_" + optimize_explanation_str + "_attr" + str(args.attribution_method) + "_locloss" + str(args.localization_loss_fn) + "_orig" + orig_name + "_resnet50" + "_lr" + str(
    #     args.learning_rate) + "_sll" + str(args.localization_loss_lambda) + "_layer" + str(args.layer) + "_feedbackType" + str(args.feedback_type)
    # if args.annotated_fraction < 1.0:
    #     out_name += f"limited{args.annotated_fraction}"
    # if args.box_dilation_percentage > 0:
    #     out_name += f"_dilation{args.box_dilation_percentage}"

    # if args.log_path is not None:
    #     writer = torch.utils.tensorboard.SummaryWriter(
    #         log_dir=os.path.join(args.log_path, args.dataset, out_name))
    # else:
    #     writer = None

    if is_bcos:
        transformer = bcos.data.transforms.AddInverse(dim=0)
    else:
        transformer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])

    root = os.path.join(args.data_path, args.dataset, "processed")
    
    

    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_loc = losses.get_localization_loss(
        args.localization_loss_fn) if args.localization_loss_fn else None

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    f1_tracker = utils.BestMetricTracker("F-Score")
    model_activator = model_activators.ResNetModelActivator(
        model=model, layer=layer_idx, is_bcos=is_bcos)
    
    
    if args.attribution_method:
        interpolate = True if layer_idx is not None else False
        attributor = attribution_methods.get_attributor(
                model, args.attribution_method, True, True, interpolate, (224, 224), batch_mode=True)       
        eval_attributor = attribution_methods.get_attributor(
            model, args.attribution_method, True, True, interpolate, (224, 224), batch_mode=False)
    else:
        attributor = None
        eval_attributor = None
    
    print(f"Loading test dataset from {root}")
    
    # if args.feedback_type == 'points':
    test_data = datasets.VOCDetectParsed(
        root=root, image_set="test_GuidingPoints", transform=transformer)
    # else: # for cases where args.feedback_type == 'bbox', mask', or None
    #     test_data = datasets.VOCDetectParsed(
    #         root=root, image_set="test", transform=transformer)
    num_test_batches = len(test_data) / args.eval_batch_size
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, collate_fn=datasets.VOCDetectParsed.collate_fn)
    final_metrics = eval_model_binary_mask(
        model_activator, eval_attributor, test_loader, num_test_batches, num_classes, loss_fn)
    final_state_dict = copy.deepcopy(model.state_dict())
    final_metrics.update(final_metric_vals)
    final_metrics.update(
        {"model": final_state_dict, "epochs": e+1} | vars(args))

    # f1_best_score, f1_best_model_dict, f1_best_epoch, f1_best_metric_vals = f1_tracker.get_best()
    # f1_best_metric_vals = utils.update_val_metrics(f1_best_metric_vals)
    # model.load_state_dict(f1_best_model_dict)
    # f1_best_metrics = eval_model_binary_mask(model_activator, eval_attributor, test_loader,
                                #  num_test_batches, num_classes, loss_fn)
    # f1_best_metrics.update(f1_best_metric_vals)
    # f1_best_metrics.update({"model": f1_best_model_dict, "epochs": f1_best_epoch+1} | vars(args))


parser = argparse.ArgumentParser()
parser.add_argument("--model_backbone", type=str, choices=["bcos", "xdnn", "vanilla"], required=True, help="Model backbone to train.")
parser.add_argument("--model_path", type=str, default=None, help="Path to checkpoint to fine tune from. When None, a model is trained starting from ImageNet pre-trained weights.")
parser.add_argument("--data_path", type=str, default="datasets/", help="Path to datasets.")
parser.add_argument("--total_epochs", type=int, default=100, help="Number of epochs to train for.")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate to use.")
parser.add_argument("--log_path", type=str, default=None, help="Path to save TensorBoard logs.")
parser.add_argument("--save_path", type=str, default="checkpoints/", help="Path to save trained models.")
parser.add_argument("--seed", type=int, default=0, help="Random seed to use.")
parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size to use for training.")
parser.add_argument("--dataset", type=str, required=True,
                    choices=["VOC2007", "COCO2014"], help="Dataset to train on.")
parser.add_argument("--localization_loss_lambda", type=float, default=1.0, help="Lambda to use to weight localization loss.")
parser.add_argument("--layer", type=str, default="Input",
                    choices=["Input", "Final", "Mid1", "Mid2", "Mid3"], help="Layer of the model to compute and optimize attributions on.")
parser.add_argument("--localization_loss_fn", type=str, default=None,
                    choices=["Energy", 'Energy_Points', "L1", "RRR", "PPCE"], help="Localization loss function to use.")
parser.add_argument("--attribution_method", type=str, default=None,
                    choices=["BCos", "GradCam", "IxG"], help="Attribution method to use for optimization.")
parser.add_argument("--optimize_explanations",
                    action="store_true", default=False, help="Flag for optimizing attributions. When False, a model is trained just using the classification loss.")
parser.add_argument("--min_fscore", type=float, default=-1, help="Minimum F-Score the best model so far must have to continue training. If the best F-Score drops below this threshold, stops training early.")
parser.add_argument("--pareto", action="store_true", default=False, help="Flag to save Pareto front of models based on F-Score, EPG Score, and IoU Score.")
parser.add_argument("--annotated_fraction", type=float, default=1.0, help="Fraction of training dataset from which bounding box annotations are to be used.")
parser.add_argument("--evaluation_frequency", type=int, default=1, help="Frequency (number of epochs) at which to evaluate the current model.")
parser.add_argument("--eval_batch_size", type=int, default=4, help="Batch size to use for evaluation.")
parser.add_argument("--box_dilation_percentage", type=float, default=0, help="Fraction of dilation to use for bounding boxes when training.")
parser.add_argument("--feedback_type", type=str, default=None, help="Type of feedback to be used for guiding explanations. Supported: mask, bbox, points.")
parser.add_argument("--num_guiding_points", type=int, default=10, help="Number of random points to sample within the object mask when using 'points' feedback.")    
parser.add_argument("--similarity_threshold", type=float, default=0.99, help="The threshold used for creating weakly supervised similarity masks from guiding points.")    
parser.add_argument("--adaptive_lambda", type=bool, default=False, help="Lambda to use to weight localization loss.")
parser.add_argument("--disable_verbose", type=bool, default=True, help="Whether to disable verbose printing of training progress.")
args = parser.parse_args()
main(args)

