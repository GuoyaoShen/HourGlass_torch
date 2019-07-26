'''
(c) Guoyao Shen
https://github.com/GuoyaoShen/HourGlass_torch
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as datatorch

import numpy as np
import matplotlib.pyplot as plt
import json
import skimage
import time

from utils import imgutils
from models.hourglass import hg as hg_torch
from models.hourglass2 import hgnet_torch

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#========================================================== TEST BY NEW IMGS ==========================================================
# Load test img
num_img = 11
path_testimg = 'test_imgs/'+str(num_img)+'.jpg'
img_np = imgutils.load_image(path_testimg)
print('img_np.SAPE', img_np.shape)

# Resize to (256,256,3)
img_np = skimage.transform.resize(img_np, [256,256])
img_np_copy = img_np
print('img_np.SAPE', img_np.shape)

# ================================== Load the Checkpoint ==================================
# device = torch.device('cuda:0')
device = torch.device('cpu')

suffix = 'allEPOCH16'  # saved suffix to load  # 4，900   5，300

# Get Model Dir
# path_ckpt_torch = 'models/ckpt_hg_torch_' + str(suffix) + '.tar'
path_ckpt_torch = 'models/hg_s8/ckpt_hg_torch_' + str(suffix) + '.tar'
# path_ckpt_torch = 'models/hg2_s4_f64/ckpt_hg_torch_' + str(suffix) + '.tar'

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

# ================================== Get Heatmaps ==================================
# Reshape and change to Tensor
img_np = np.transpose(img_np, (2, 0, 1))
img_np = img_np[np.newaxis, ...]  #(1,3,256,256)
print('img_np.HAPE', img_np.shape)
img = torch.from_numpy(img_np).float().to(device)

# Get predict heatmap
time_start = time.perf_counter()
heatmaps_pred_eg = net_hg_torch(img)
time_end = time.perf_counter()
print('prediction time use:', time_end - time_start, 's')
heatmaps_pred_eg = heatmaps_pred_eg[-1].double()  #(1,16,64,64)

# Reshape pred heatmaps
heatmaps_pred_eg_np = heatmaps_pred_eg.detach().cpu().numpy()
heatmaps_pred_eg_np = np.squeeze(heatmaps_pred_eg_np, axis=0)
heatmaps_pred_eg_np = np.swapaxes(np.swapaxes(heatmaps_pred_eg_np, 0, 2), 0, 1)  #(64,64,16)

# Show heatmaps
imgutils.show_heatmaps(img_np_copy, heatmaps_pred_eg_np)

# Stack points
coord_joints = imgutils.heatmaps_to_coords(heatmaps_pred_eg_np, resolu_out=[256, 256], prob_threshold=0.2)
imgutils.show_stack_joints(img_np_copy, coord_joints, draw_lines=True)
