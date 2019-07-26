'''
(c) Guoyao Shen
https://github.com/GuoyaoShen/HourGlass_torch
'''
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import json
import skimage
import time

from utils import imgutils
from models.hourglass import hg as hg_torch
from models.hourglass2 import hgnet_torch

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


# ====================== initialize the HOG descriptor/person detector ======================
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ================================== Load the checkpoint ==================================
device = torch.device('cuda:0')
# device = torch.device('cpu')

suffix = 'allEPOCH16'  # saved suffix to load
# path_ckpt_torch = 'models/ckpt_hg_torch_' + str(suffix) + '.tar'
path_ckpt_torch = 'models/hg_s8/ckpt_hg_torch_' + str(suffix) + '.tar'
checkpoint = torch.load(path_ckpt_torch)
print('===============CHECKPOINT LOADED===============')

net_hg_torch = hg_torch(num_stacks=8, num_blocks=1, num_classes=16)
# net_hg_torch = hgnet_torch(num_stacks=2, num_blocks=1, num_classes=16, num_features=64)
net_hg_torch.load_state_dict(checkpoint['model_state_dict'])
net_hg_torch = net_hg_torch.to(device)

print('Reconstruct Model DONE')

# net_hg_torch.eval()
# net_hg_torch.train()

print('model.training', net_hg_torch.training)

# ===============================================================================================

time_total_start = time.perf_counter()

num_img = 100  # change test image
path_testimg = 'test_imgs/'+str(num_img)+'.jpg'
image = cv2.imread(path_testimg)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# count time
time_read_start = time.perf_counter()

# prepare for model
shape = image.shape
im_copy = image.copy()
cv2.imshow("im_copy", im_copy)

im_copy = cv2.resize(im_copy, (int(shape[1]/10),int(shape[0]/10)))  # change resolution(affect speed)

(rects, weights) = hog.detectMultiScale(im_copy, winStride=(4, 4),
                                        padding=(4, 4), scale=1.05)

# delete overlap rects
rects = non_max_suppression_fast(rects, 0.3)

# crop image
for (x, y, w, h) in rects:
    im_center = [y+int(h/2), x+int(w/2)]
    square_len = (max(w, h))

    print(square_len)

    im_shape = im_copy.shape
    img_crop_new = np.zeros([square_len, square_len, 3], np.uint8)

    img_crop_new[int((square_len - h)/2):int((square_len - h)/2)+h,
                 int((square_len - w) / 2):int((square_len - w) / 2) + w] = im_copy[y:y+h, x:x+w]
    cv2.imshow("img_new", img_crop_new)

im_copy = cv2.resize(img_crop_new, (256, 256))
time_read_end = time.perf_counter()
print('read img use:', time_read_end - time_read_start, 's')


# ================================== Get heatmaps ==================================
# Reshape and change to Tensor
img_np = np.transpose(im_copy, (2, 0, 1))
img_np = img_np[np.newaxis, ...]  #(1,3,256,256)
print('img_np.SAPE', img_np.shape)

time_trans_start = time.perf_counter()
img = torch.from_numpy(img_np).float().to(device)
time_trans_end = time.perf_counter()
print('trans time use:', time_trans_end - time_trans_start, 's')

# Get predict heatmap
time_pred_start = time.perf_counter()
heatmaps_pred_eg = net_hg_torch(img)
time_pred_end = time.perf_counter()
print('prediction time use:', time_pred_end - time_pred_start, 's')
heatmaps_pred_eg = heatmaps_pred_eg[-1].double()  #(1,16,64,64)

time_total_end = time.perf_counter()
print('Total time use:', time_total_end - time_total_start, 's')

# Reshape pred heatmaps
time_retrans_start = time.perf_counter()
heatmaps_pred_eg_np = heatmaps_pred_eg.detach().cpu().numpy()
time_retrans_end = time.perf_counter()
print('retrans time use:', time_retrans_end - time_retrans_start, 's')

heatmaps_pred_eg_np = np.squeeze(heatmaps_pred_eg_np, axis=0)
heatmaps_pred_eg_np = np.swapaxes(np.swapaxes(heatmaps_pred_eg_np, 0, 2), 0, 1)  #(64,64,16)

# Show heatmaps
imgutils.show_heatmaps(im_copy, heatmaps_pred_eg_np)

# Stack points
coord_joints = imgutils.heatmaps_to_coords(heatmaps_pred_eg_np, resolu_out=[256, 256], prob_threshold=0.3)
imgutils.show_stack_joints(im_copy, coord_joints, draw_lines=True)
