import os
import datasets 
import torch
import cv2
import numpy as np
import torchvision
import utils
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

dir = "VOC2007"
# dir = "COCO2014"

root = os.path.join(f"datasets/{dir}/processed")

print(f"Loading guiding points train dataset from {root}")
transformer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
train_data = datasets.VOCDetectParsed(
    root=root, image_set="train_GuidingPoints", transform=transformer, annotated_fraction=1.)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=32, shuffle=False, num_workers=0, collate_fn=datasets.VOCDetectParsed.collate_fn)

print(train_loader)

polygons = []        
for batch_idx, (train_X, train_y, train_bbs, train_masks, guiding_points, indices) in enumerate(tqdm(train_loader)):
    gt_classes = utils.get_random_optimization_targets(train_y)
    for img_idx in range(len(train_X)):
        target_mask = torch.where(train_masks[img_idx].cuda()==gt_classes[img_idx]+1, 1., 0.).detach()
        # binary_mask: 0/1 or 0/255, shape (H, W)
        binary_mask = (target_mask.detach().cpu().numpy() > 0).astype(np.uint8)

        # plt.subplot(3,1,1)
        # plt.imshow(target_mask.detach().cpu().numpy())

        # plt.subplot(3,1,2)
        # plt.imshow(binary_mask)

        # plt.subplot(3,1,3)
        # plt.imshow(train_X[img_idx].moveaxis(0,-1).detach().cpu().numpy())

        # plt.savefig(f'figures/mask_{batch_idx}.png')


        contours, hierarchy = cv2.findContours(
            binary_mask,
            mode=cv2.RETR_EXTERNAL,      # outer polygons only
            method=cv2.CHAIN_APPROX_SIMPLE
        )

        # Each contour is an Nx2 polygon
        polys = [c.squeeze(1) for c in contours]
        if len(polys) > 0:
            polygons.append(np.mean([len(polygon) for polygon in polys]))
    # break

print(np.mean(polygons))
with open('logs/VOC_stats.pkl', 'wb') as file:
    pickle.dump(polygons, file)


