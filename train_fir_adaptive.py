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
import fixup_resnet

import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import os

import matplotlib.pyplot as plt
import torchvision.transforms.functional as ff
import pickle

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
                if test_masks[img_idx].sum() == 0:
                    continue
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
        print(f'AdaptiveLambda: {args.adaptive_lambda}, , SimilarityThreshold: {args.similarity_threshold}, NumGuidingPoints: {args.num_guiding_points}.')
    
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
    model.train()


    orig_name = os.path.basename(
        args.model_path) if args.model_path else str(None)

    model_prefix = args.model_backbone

    optimize_explanation_str = "finetunedobjloc" if args.optimize_explanations else "standard"
    optimize_explanation_str += args.feedback_type if args.feedback_type else ""
    optimize_explanation_str += "pareto" if args.pareto else ""
    optimize_explanation_str += "limited" if args.annotated_fraction < 1.0 else ""
    optimize_explanation_str += "dilated" if args.box_dilation_percentage > 0 else ""

    out_name = model_prefix + "_" + optimize_explanation_str + "_attr" + str(args.attribution_method) + "_locloss" + str(args.localization_loss_fn) + "_orig" + orig_name + "_resnet50" + "_lr" + str(
        args.learning_rate) + "_sll" + str(args.localization_loss_lambda) + "_layer" + str(args.layer) + "_feedbackType" + str(args.feedback_type)
    if args.feedback_type == 'points_adaptive':
        out_name = out_name + f"_{args.adaptive_points_threshold}"
    if args.annotated_fraction < 1.0:
        out_name += f"limited{args.annotated_fraction}"
    if args.box_dilation_percentage > 0:
        out_name += f"_dilation{args.box_dilation_percentage}"

    save_path = os.path.join(args.save_path, args.dataset, out_name)
    os.makedirs(save_path, exist_ok=True)

    if args.log_path is not None:
        writer = torch.utils.tensorboard.SummaryWriter(
            log_dir=os.path.join(args.log_path, args.dataset, out_name))
    else:
        writer = None

    if is_bcos:
        transformer = bcos.data.transforms.AddInverse(dim=0)
    else:
        transformer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])

    root = os.path.join(args.data_path, args.dataset, "processed")
    
    
    # if args.feedback_type == 'points':
    print(f"Loading guiding points train dataset from {root}")
    train_data = datasets.VOCDetectParsed(
        root=root, image_set="train_GuidingPoints", transform=transformer, annotated_fraction=args.annotated_fraction)
    print(f"Loading guiding points validation dataset from {root}")
    val_data = datasets.VOCDetectParsed(
        root=root, image_set="val_GuidingPoints", transform=transformer)
    # else: # for cases where args.feedback_type == 'bbox', 'mask', or None
    #     print(f"Loading train dataset from {root}")
    #     train_data = datasets.VOCDetectParsed(
    #         root=root, image_set="train", transform=transformer, annotated_fraction=args.annotated_fraction)
    #     print(f"Loading validation dataset from {root}")
    #     val_data = datasets.VOCDetectParsed(
    #         root=root, image_set="val", transform=transformer)

    print(f"Train data size: {len(train_data)}")
    annotation_count = 0
    total_count = 0
    for idx in range(len(train_data)):
        if train_data[idx][2] is not None:
            annotation_count += 1
        total_count += 1
    print(f"Annotated: {annotation_count}, Total: {total_count}")

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.train_batch_size, shuffle=False, num_workers=4, collate_fn=datasets.VOCDetectParsed.collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.eval_batch_size, shuffle=False, num_workers=4, collate_fn=datasets.VOCDetectParsed.collate_fn)
    
    
    num_train_batches = len(train_data) / args.train_batch_size
    num_val_batches = len(val_data) / args.eval_batch_size

    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_loc = losses.get_localization_loss(
        args.localization_loss_fn) if args.localization_loss_fn else None

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    if args.attribution_method:
        interpolate = True if layer_idx is not None else False
        attributor = attribution_methods.get_attributor(
                model, args.attribution_method, loss_loc.only_positive, loss_loc.binarize, interpolate, (224, 224), batch_mode=True)       
        eval_attributor = attribution_methods.get_attributor(
            model, args.attribution_method, loss_loc.only_positive, loss_loc.binarize, interpolate, (224, 224), batch_mode=False)
    else:
        attributor = None
        eval_attributor = None
    f1_tracker = utils.BestMetricTracker("F-Score")
    if attributor:
        epg_tracker = utils.BestMetricTracker("BB-Loc")
    model_activator = model_activators.ResNetModelActivator(
        model=model, layer=layer_idx, is_bcos=is_bcos)
    if args.pareto:
        pareto_front_tracker = utils.ParetoFrontModels()
    flag = True
    gpoints_dict = {}
    for e in range(args.total_epochs):
        total_loss = 0
        total_class_loss = 0
        total_localization_loss = 0
        # print(f'epoch {e} with training set length {len(train_loader)}')
        # print(f'epoch {e} with guiding point set size {sys.getsizeof(train_loader.dataset.guiding_points)}')
        for batch_idx, (train_X, train_y, train_bbs, train_masks, guiding_points, indices) in enumerate(train_loader):
            # print(f'Processing batch {batch_idx} / {len(train_loader)}')
            batch_loss = 0
            localization_loss = 0
            optimizer.zero_grad()
            train_X.requires_grad = True
            train_X = train_X.cuda()
            train_y = train_y.cuda()
            logits, features, acts = model_activator(train_X)
            loss = loss_fn(logits, train_y)
            batch_loss += loss
            total_class_loss += loss.detach()
            if args.optimize_explanations:
                # print(f'targets: {train_y}')
                # print(f'sum of targets per image: {train_y.sum(dim=1)}')
                # for img_idx in range(len(train_X)):
                #     img_temp = (train_X[img_idx]-torch.min(train_X[img_idx]))
                #     img_temp = img_temp / torch.max(img_temp)
                    # plt.imshow(img_temp.detach().cpu().moveaxis(0, -1))
                    # plt.savefig(f'figures/input_image_{img_idx}.png')
                    # plt.close('all')
                gt_classes = utils.get_random_optimization_targets(train_y)
                attributions = attributor(features, logits, classes=gt_classes).squeeze(1)
                if args.feedback_type == "bbox":
                    for img_idx in range(len(train_X)):
                        if train_bbs[img_idx] is None:
                            continue
                        if train_masks[img_idx].sum() == 0:
                            continue
                        bb_list = utils.filter_bbs(
                            train_bbs[img_idx], gt_classes[img_idx])
                        if args.box_dilation_percentage > 0:
                            bb_list = utils.enlarge_bb(
                                bb_list, percentage=args.box_dilation_percentage)
                        item_locization_loss = loss_loc(attributions=attributions[img_idx], bb_coordinates=bb_list)
                        if item_locization_loss.isnan():
                            continue
                        else:
                            localization_loss += item_locization_loss
                    
                    batch_loss += args.localization_loss_lambda*localization_loss
                    if torch.is_tensor(localization_loss):
                        total_localization_loss += localization_loss.detach()
                    else:
                        total_localization_loss += localization_loss
                elif args.feedback_type == "points":
                    for img_idx in range(len(train_X)):
                        if train_masks[img_idx].sum() == 0:
                            continue
                        if guiding_points[img_idx][gt_classes[img_idx]] is None:
                            
                            # print(f'gt classes: {gt_classes[img_idx]}, and image classes: {torch.unique(train_masks[img_idx])}')
                            target_mask = torch.where(train_masks[img_idx].cuda()==gt_classes[img_idx]+1, 0., 1.).detach()
                            if (1. - target_mask.sum()) == 0:
                                guiding_points[img_idx][gt_classes[img_idx]] = []
                            else:
                                # print(f'target mask has shape: {target_mask.shape}!')
                                target_mask = ff.resize(target_mask.unsqueeze(0), size=(7,7), interpolation=torchvision.transforms.InterpolationMode.BICUBIC).squeeze().reshape(-1).clamp(0)

                                if  target_mask.sum() < 1.e-6 or torch.isnan(target_mask.sum() or torch.isnan(attributions[img_idx].sum())):
                                    train_loader.dataset.guiding_points[indices[img_idx]][gt_classes[img_idx]] = guiding_points[img_idx][gt_classes[img_idx]] = []
                                    # print(f'Warning: target mask for image {img_idx} in batch {batch_idx} (actual index in dataset: {indices[img_idx]}) and class {gt_classes[img_idx]} has sum {target_mask.sum()}. Skipping this image for this epoch.')
                                    break
                                else:
                                    if np.min([args.num_guiding_points, torch.sum(target_mask != 0).item()]) < 1:
                                        rand_indices = []
                                        points = []
                                    else:
                                        rand_indices = torch.multinomial(target_mask, np.min([args.num_guiding_points, torch.sum(target_mask != 0).item()]), replacement=False).cpu()
                                        # rand_indices = (target_mask).nonzero(as_tuple=False)
                                        x_index = torch.div(rand_indices, 7, rounding_mode='floor') # Row index
                                        y_index = rand_indices % 7   # Column index
                                        # points = list(set([(torch.floor(x_index[i]/32).int().item(),torch.floor(y_index[i]/32).int().item()) for i in range(len(rand_indices))]))
                                        points = list(set([(x_index[i].int().item(), y_index[i].int().item()) for i in range(len(rand_indices))]))

                                    guiding_points[img_idx][gt_classes[img_idx]] = train_loader.dataset.guiding_points[indices[img_idx]][gt_classes[img_idx]] = points

                        if args.similarity_threshold < 1.0:
                            weak_mask = torch.zeros(7,7).cuda()
                            # weak_mask[guiding_points[img_idx][gt_classes[img_idx]]] = 1.
                            for gpoint in guiding_points[img_idx][gt_classes[img_idx]]:
                                sim = torch.nn.functional.cosine_similarity(acts[img_idx, :, gpoint[0], gpoint[1]].unsqueeze(1).unsqueeze(1), acts[img_idx], dim=0)
                                weak_mask = torch.max(weak_mask, sim)
                                # weak_mask[gpoint] = 1.

                            weak_mask = ff.resize(weak_mask.unsqueeze(0).unsqueeze(0), size=(224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC).squeeze()
                            weak_mask = torch.where(weak_mask>args.similarity_threshold, 0, 1).detach()
                        else:
                            weak_mask = torch.ones(224,224).cuda()
                            for gpoint in guiding_points[img_idx][gt_classes[img_idx]]:
                                weak_mask[gpoint] = 0.
                            
                        
                        # Code used for debugging and visualization of guiding points, weak masks, attributions, and input images
                        # if True: # img_idx == 0 and batch_idx < 25:
                        #     # Plot the images, masks, and attributions for the first image in the batch for diagnostic purposes
                        #     plt.figure(figsize=(30,5))
                        #     plt.subplot(1,7,1)
                        #     plt.imshow(train_X[img_idx][:3].detach().cpu().moveaxis(0, -1))
                        #     plt.title('input')
                        #     plt.axis('off')
                        #     plt.subplot(1,7,2)
                        #     plt.imshow(train_X[img_idx][-3:].detach().cpu().moveaxis(0, -1))
                        #     plt.title('input - last 3 channels for BCos')
                        #     plt.axis('off')
                        #     plt.subplot(1,7,3)
                        #     plt.imshow(train_masks[img_idx])
                        #     plt.title('train masks')
                        #     plt.axis('off')
                        #     plt.subplot(1,7,4)
                        #     train_masks[img_idx] = torch.where(train_masks[img_idx] == gt_classes[img_idx].cpu()+1, 1, 0)
                        #     mx = attributions[img_idx].max().item()
                        #     # for point in guiding_points[img_idx][gt_classes[img_idx]]:
                        #         # attributions[img_idx][point] = mx*1.2
                        #         # train_masks[img_idx][point] = 10
                            
                        #     plt.imshow(train_masks[img_idx])
                        #     plt.title('filtered target masks')
                        #     plt.axis('off')
                        #     plt.subplot(1,7,5)
                        #     plt.imshow(attributions[img_idx].detach().cpu())
                        #     plt.suptitle(f'{gt_classes[img_idx].item()+1}:{class_names[gt_classes[img_idx].item()+1]} - max value {mx:1.4f}, of class {gt_classes[img_idx]+1}')
                        #     plt.title('attributions')
                        #     plt.axis('off')
                        #     plt.subplot(1,7,6)
                        #     plt.imshow(acts[img_idx].sum(dim=0).detach().cpu())
                        #     # plt.imshow(weak_mask[3:].detach().cpu(), alpha=0.5)
                        #     plt.title('features')
                        #     plt.axis('off')
                        #     plt.subplot(1,7,7)
                        #     plt.imshow(weak_mask[3:].detach().cpu())
                        #     plt.title(f'weak mask: {weak_mask.max().item()}')
                        #     plt.axis('off')
                        #     plt.savefig(f"figures/mask_{e}_{batch_idx}_{img_idx}_{gt_classes[img_idx]}.png")
                        #     plt.close('all')

                        # weak_mask = torch.where(train_masks[img_idx].cuda()==gt_classes[img_idx]+1, 1., 0.).detach()
                        if args.adaptive_lambda:
                            AdaptiveLambda = torch.numel(weak_mask)/torch.sum(1.-weak_mask)
                            if torch.isinf(AdaptiveLambda) or torch.isnan(AdaptiveLambda):
                                AdaptiveLambda = torch.tensor(1.).cuda()
                            localization_loss += AdaptiveLambda.detach()*args.localization_loss_lambda*loss_loc(attributions[img_idx], mask = weak_mask)
                        else:
                            localization_loss += args.localization_loss_lambda*loss_loc(attributions=attributions[img_idx], mask = weak_mask)
                            # print(f'localization loss for image {img_idx} in batch {batch_idx} is {loss_loc(attributions[img_idx], weak_mask).item()}')
                    batch_loss += localization_loss

                    if torch.is_tensor(localization_loss):
                        total_localization_loss += localization_loss.detach()
                    else:
                        total_localization_loss += localization_loss
                elif args.feedback_type == "points_adaptive":
                    for img_idx in range(len(train_X)):
                        if (torch.where(train_masks[img_idx].cuda().detach()==gt_classes[img_idx]+1, 1., 0.)).sum() == 0:
                            continue
                        if guiding_points[img_idx][gt_classes[img_idx]] is None:
                            target_mask = (torch.where(train_masks[img_idx].cuda().detach()==gt_classes[img_idx]+1, 0., 1.))
                            target_mask = ff.resize(target_mask.unsqueeze(0), size=(7,7), interpolation=torchvision.transforms.InterpolationMode.BICUBIC).squeeze().reshape(-1)
                            att = ff.resize(attributions[img_idx].unsqueeze(0), size=(7,7), interpolation=torchvision.transforms.InterpolationMode.BICUBIC).squeeze().reshape(-1)
                            # target_mask = target_mask * att / (att.sum().item()) # + 1.e-8)
                            target_mask = target_mask.clamp(min=0).detach()
                            # print(f'target mask size: {target_mask.shape}, sum: {target_mask.sum().item()}')
                            # print(f'attributions size: {attributions[img_idx].shape}, sum: {attributions[img_idx].sum().item()}')

                            if  target_mask.sum() < 1.e-6 or torch.isnan(target_mask.sum() or torch.isnan(attributions[img_idx].sum())):
                                # 
                                train_loader.dataset.guiding_points[indices[img_idx]][gt_classes[img_idx]] = guiding_points[img_idx][gt_classes[img_idx]] = []
                                # print(f'Warning: target mask for image {img_idx} in batch {batch_idx} (actual index in dataset: {indices[img_idx]}) and class {gt_classes[img_idx]} has sum {target_mask.sum()}. Skipping this image for this epoch.')
                                break
                            else:
                                if np.min([args.num_guiding_points, torch.sum(target_mask != 0).item()]) < 1:
                                    rand_indices = []
                                    points = []
                                else:
                                    weak_mask = torch.zeros(7,7).cuda()
                                    rand_indices = []
                                    x_index = []
                                    y_index = []
                                    while (target_mask.reshape(7,7)*torch.where(weak_mask>args.similarity_threshold, 0, 1)* att.reshape(7,7) / (att.sum().item())).sum() > args.adaptive_points_threshold and len(x_index) < args.num_guiding_points:
                                        # print(f'irrelevant area left for image {img_idx}: {(target_mask.reshape(7,7)*torch.where(weak_mask>args.similarity_threshold, 0, 1)* att.reshape(7,7) / (att.sum().item())).sum().item()}')
                                        prob = target_mask*(1-weak_mask.reshape(49))
                                        prob = prob.clamp(0)
                                        seed_choice = torch.multinomial(prob.cpu(), 1)
                                        x_index.append(torch.div(seed_choice, 7, rounding_mode='floor')) # Row index
                                        y_index.append(seed_choice % 7)   # Column index
                                        sim = torch.nn.functional.cosine_similarity(acts[img_idx, :, x_index[-1], y_index[-1]].unsqueeze(-1), acts[img_idx], dim=0)
                                        weak_mask = torch.max(weak_mask, sim)
                                    points = list(set([(x_index[i].int().item(), y_index[i].int().item()) for i in range(len(x_index))]))
                                guiding_points[img_idx][gt_classes[img_idx]] = train_loader.dataset.guiding_points[indices[img_idx]][gt_classes[img_idx]] = points
                        
                        
                        
                        if args.similarity_threshold < 1.0:
                            weak_mask = torch.zeros(7,7).cuda()
                            # weak_mask[guiding_points[img_idx][gt_classes[img_idx]]] = 1.
                            for gpoint in guiding_points[img_idx][gt_classes[img_idx]]:
                                sim = torch.nn.functional.cosine_similarity(acts[img_idx, :, gpoint[0], gpoint[1]].unsqueeze(1).unsqueeze(1), acts[img_idx], dim=0)
                                weak_mask = torch.max(weak_mask, sim)
                                # weak_mask[gpoint] = 1.

                            weak_mask = ff.resize(weak_mask.unsqueeze(0).unsqueeze(0), size=(224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC).squeeze()
                            weak_mask = torch.where(weak_mask>args.similarity_threshold, 0, 1).detach()
                        else:
                            weak_mask = torch.ones(224,224).cuda()
                            for gpoint in guiding_points[img_idx][gt_classes[img_idx]]:
                                weak_mask[gpoint] = 0.
                            
                        gt = (torch.where(train_masks[img_idx].cuda().detach()==gt_classes[img_idx]+1, 1., 0.))
                        # plt.imshow(gt.detach().cpu(), cmap='gray')
                        # plt.savefig(f'figures/mask_{img_idx}.png')
                        # plt.close('all')
                        # plt.imshow((1.-weak_mask).detach().cpu(), cmap='gray')
                        # plt.savefig(f'figures/weak_mask_{img_idx}.png')
                        # plt.close('all')
                        mask_bleed = ((1.-weak_mask)*gt)/(gt.sum().item())
                        gpoints_dict[indices[img_idx]] = {'seed_points': len(guiding_points[img_idx][gt_classes[img_idx]]), 'bleed': mask_bleed.sum().item()}
                        # print(f'mask bleed for image {img_idx} in batch {batch_idx} is {mask_bleed.sum().item()}')

                        # # Code used for debugging and visualization of guiding points, weak masks, attributions, and input images
                        # if True: # img_idx == 0 and batch_idx < 25:
                        #     # Plot the images, masks, and attributions for the first image in the batch for diagnostic purposes
                        #     plt.figure(figsize=(30,5))
                        #     plt.subplot(1,7,1)
                        #     plt.imshow(train_X[img_idx][:3].detach().cpu().moveaxis(0, -1))
                        #     plt.title('input')
                        #     plt.axis('off')
                        #     plt.subplot(1,7,2)
                        #     plt.imshow(train_X[img_idx][-3:].detach().cpu().moveaxis(0, -1))
                        #     plt.title('input - last 3 channels for BCos')
                        #     plt.axis('off')
                        #     plt.subplot(1,7,3)
                        #     plt.imshow(train_masks[img_idx])
                        #     plt.title('train masks')
                        #     plt.axis('off')
                        #     plt.subplot(1,7,4)
                        #     train_masks[img_idx] = torch.where(train_masks[img_idx] == gt_classes[img_idx].cpu()+1, 1, 0)
                        #     mx = attributions[img_idx].max().item()
                        #     # for point in guiding_points[img_idx][gt_classes[img_idx]]:
                        #         # attributions[img_idx][point] = mx*1.2
                        #         # train_masks[img_idx][point] = 10
                            
                        #     plt.imshow(train_masks[img_idx])
                        #     plt.title('filtered target masks')
                        #     plt.axis('off')
                        #     plt.subplot(1,7,5)
                        #     plt.imshow(attributions[img_idx].detach().cpu())
                        #     plt.suptitle(f'{gt_classes[img_idx].item()+1}:{class_names[gt_classes[img_idx].item()+1]} - max value {mx:1.4f}, of class {gt_classes[img_idx]+1}')
                        #     plt.title('attributions')
                        #     plt.axis('off')
                        #     plt.subplot(1,7,6)
                        #     plt.imshow(acts[img_idx].sum(dim=0).detach().cpu())
                        #     # plt.imshow(weak_mask[3:].detach().cpu(), alpha=0.5)
                        #     plt.title('features')
                        #     plt.axis('off')
                        #     plt.subplot(1,7,7)
                        #     plt.imshow(weak_mask[3:].detach().cpu())
                        #     plt.title(f'weak mask: {weak_mask.max().item()}')
                        #     plt.axis('off')
                        #     plt.savefig(f"figures/mask_{e}_{batch_idx}_{img_idx}_{gt_classes[img_idx]}.png")
                        #     plt.close('all')

                        # weak_mask = torch.where(train_masks[img_idx].cuda()==gt_classes[img_idx]+1, 1., 0.).detach()
                        if args.adaptive_lambda:
                            AdaptiveLambda = torch.numel(weak_mask)/torch.sum(1 - weak_mask)
                            # print(f'adaptive lambda before : {AdaptiveLambda}')

                            if torch.isinf(AdaptiveLambda) or torch.isnan(AdaptiveLambda):
                                AdaptiveLambda = torch.tensor(1.).cuda()
                            # print(f'Adaptive Lambda for image {img_idx} in batch {batch_idx} is {AdaptiveLambda.item()}')
                            # print(f'attributions size: {attributions[img_idx].shape}, weak mask size: {weak_mask.shape}')
                            localization_loss += AdaptiveLambda.detach()*args.localization_loss_lambda*loss_loc(attributions[img_idx], mask=weak_mask)
                        else:
                            localization_loss += args.localization_loss_lambda*loss_loc(attributions=attributions[img_idx], mask=weak_mask)
                            # print(f'localization loss for image {img_idx} in batch {batch_idx} is {loss_loc(attributions[img_idx], weak_mask).item()}')
                    batch_loss += localization_loss

                    if torch.is_tensor(localization_loss):
                        total_localization_loss += localization_loss.detach()
                    else:
                        total_localization_loss += localization_loss
                elif args.feedback_type == "mask":
                    for img_idx in range(len(train_X)):
                        if train_masks[img_idx].sum() == 0:
                            # print(f'Batch {batch_idx}, skipping image {img_idx} due to empty mask.')
                            continue
                        target_mask = torch.where(train_masks[img_idx].cuda()==gt_classes[img_idx]+1, 1., 0.).detach()
                        item_locization_loss = loss_loc(attributions=attributions[img_idx], mask=target_mask)
                        if item_locization_loss.isnan():
                            continue
                        else:
                            localization_loss += item_locization_loss
                        # localization_loss += loss_loc(attributions=attributions[img_idx], mask=target_mask)
                        # print(f'localization loss for image {img_idx} in batch {batch_idx} is {loss_loc(attributions=attributions[img_idx], mask=target_mask).item()}')
                    batch_loss += args.localization_loss_lambda*localization_loss
                    
                    if torch.is_tensor(localization_loss):
                        total_localization_loss += localization_loss.detach()
                    else:
                        total_localization_loss += localization_loss
                else:
                    raise NotImplementedError
               
            batch_loss.backward()
            total_loss += batch_loss.detach()
            optimizer.step()
        
        del train_X, train_y, logits, features

        print('')
        print(f"Epoch: {e}, Average Loss: {total_loss / num_train_batches}")

        if writer:
            writer.add_scalar("train_loss", total_loss, e+1)
            writer.add_scalar("class_loss", total_class_loss, e+1)
            writer.add_scalar("localization_loss", total_localization_loss, e+1)
        if (e+1) % args.evaluation_frequency == 0:
            metric_vals = eval_model_binary_mask(model_activator, eval_attributor, val_loader,
                                     num_val_batches, num_classes, loss_fn, writer, e)
            if args.pareto:
                pareto_front_tracker.update(model, metric_vals, e)
            best_fscore, _, _, _ = f1_tracker.get_best()
            if (best_fscore is not None) and (best_fscore < args.min_fscore):
                print(
                    f'F-Score below threshold, actual: {metric_vals["F-Score"]}, threshold: {args.min_fscore}')
                metric_vals.update(
                    {"model": None, "epochs": e+1} | vars(args))
                metric_vals.update({"BelowThresh": True})
                torch.save(metric_vals, os.path.join(
                    save_path, f"model_checkpoint_stopped_{e+1}.pt"))
                if args.pareto:
                    pareto_front_tracker.save_pareto_front(save_path)
                return
            f1_tracker.update(metric_vals, model, e)
            if attributor:
                epg_tracker.update(metric_vals, model, e)
        if args.feedback_type == 'points_adaptive':
            print(f'average guiding points per image: {np.mean(list([value["seed_points"] for value in gpoints_dict.values()]))}')
            print(f'average mask bleed per image: {np.mean(list([value["bleed"] for value in gpoints_dict.values()]))}')
    if args.feedback_type == 'points_adaptive':
        print(f'average guiding points per image: {np.mean(list([value["seed_points"] for value in gpoints_dict.values()]))}')
        with open(f'logs/COCO_Final/points_adaptive/bleed/{args.model_backbone}/stats/guiding_points_stats_{args.num_guiding_points}_{args.adaptive_points_threshold}_{args.similarity_threshold}_{args.localization_loss_lambda}_.pkl', 'wb') as f:
            pickle.dump(gpoints_dict, f)

    if args.pareto:
        pareto_front_tracker.save_pareto_front(save_path)

    final_metric_vals = metric_vals
    final_metric_vals = utils.update_val_metrics(final_metric_vals)

    del train_data, val_data, train_loader, val_loader
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

    f1_best_score, f1_best_model_dict, f1_best_epoch, f1_best_metric_vals = f1_tracker.get_best()
    f1_best_metric_vals = utils.update_val_metrics(f1_best_metric_vals)
    model.load_state_dict(f1_best_model_dict)
    f1_best_metrics = eval_model_binary_mask(model_activator, eval_attributor, test_loader,
                                 num_test_batches, num_classes, loss_fn)
    f1_best_metrics.update(f1_best_metric_vals)
    f1_best_metrics.update(
        {"model": f1_best_model_dict, "epochs": f1_best_epoch+1} | vars(args))

    if attributor:
        epg_best_score, epg_best_model_dict, epg_best_epoch, epg_best_metric_vals = epg_tracker.get_best()
        epg_best_metric_vals = utils.update_val_metrics(epg_best_metric_vals)
        model.load_state_dict(epg_best_model_dict)
        epg_best_metrics = eval_model_binary_mask(model_activator, eval_attributor, test_loader,
                                    num_test_batches, num_classes, loss_fn)
        epg_best_metrics.update(epg_best_metric_vals)
        epg_best_metrics.update(
            {"model": epg_best_model_dict, "epochs": epg_best_epoch+1} | vars(args))
        torch.save(epg_best_metrics, os.path.join(
            save_path, f"model_checkpoint_epg_best.pt"))

    torch.save(final_metrics, os.path.join(
        save_path, f"model_checkpoint_final_{e+1}.pt"))
    torch.save(f1_best_metrics, os.path.join(
        save_path, f"model_checkpoint_f1_best.pt"))


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
parser.add_argument("--adaptive_points_threshold", type=float, default=0.75, help="Used to determine the number of guiding points adaptively.")
args = parser.parse_args()
main(args)
