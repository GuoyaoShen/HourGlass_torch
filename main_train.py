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
import os
import math

from utils import imgutils
from models.hourglass import hg as hg_torch
from models.hourglass2 import hgnet_torch as hgnet_torch
from losses.jointsmseloss import JointsMSELoss
from datasets.mpii import MPII

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ================================== Construct dataset ==================================
num_bcsz = 4

ds_torch = MPII(use_scale=True, use_flip=True, use_rand_color=True)
data_loader = datatorch.DataLoader(ds_torch, batch_size=num_bcsz, shuffle=True)

# ================================== Construct model ==================================
device = torch.device('cuda:0')
# device = torch.device('cpu')
learning_rate = 1e-3

net_hg_torch = hg_torch(num_stacks=2, num_blocks=1, num_classes=16).to(device)
# net_hg_torch = hgnet_torch(num_stacks=2, num_blocks=1, num_classes=16, num_features=64).to(device)

optimizer = torch.optim.RMSprop(net_hg_torch.parameters(), lr=learning_rate)
criteon = JointsMSELoss(use_target_weight=True).to(device)

# ================================== Train ==================================
num_epoch = 50
num_setsize = ds_torch.__len__()
print('num_setsize', num_setsize)
target_weight = np.array([[1.2, 1.1, 1, 1, 1.1, 1.2, 1, 1, 1, 1, 1.2, 1.1, 1, 1, 1.1, 1.2]])
target_weight = torch.from_numpy(target_weight).to(device).float()

plt.ion()
for i in range(num_epoch):
    for step, (img, heatmaps, pts) in enumerate(data_loader):
        # To GPU
        img, heatmaps = img.to(device), heatmaps.cuda()

        # All dtype change to float
        img, heatmaps = img.float(), heatmaps.float()

        print('')
        print('EPOCH', str(num_epoch), '/', i + 1, ' ||  STEP', math.ceil(num_setsize / num_bcsz), '/', step + 1)

        heatmaps_pred_list = net_hg_torch(img)

        # # Intermediate Supervision
        # loss = torch.zeros([]).to(device)
        # for ele_heatmaps_pred in heatmaps_pred_list:
        #     ele_loss = criteon(ele_heatmaps_pred, heatmaps, target_weight)
        #     loss = loss + ele_loss
        # loss = loss / len(heatmaps_pred_list)
        # heatmaps_pred_final = heatmaps_pred_list[-1]

        # Only final loss, NO intermediate supervision
        heatmaps_pred_final = heatmaps_pred_list[-1]
        loss = criteon(heatmaps_pred_final, heatmaps, target_weight)

        print('LOSS:', loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Show last layer output
        heatmaps_pred_copy = heatmaps_pred_final[1]
        heatmaps_copy = heatmaps[1]
        img_copy = img[1]

        # Show pred heatmaps
        heatmaps_pred_np = heatmaps_pred_copy.detach().cpu().numpy()
        heatmaps_pred_np = np.transpose(heatmaps_pred_np, (1, 2, 0))
        heatmaps_np = heatmaps_copy.detach().cpu().numpy()
        heatmaps_np = np.transpose(heatmaps_np, (1, 2, 0))
        img_np = img_copy.detach().cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))

        # In case too many figs to print
        if step % 5 == 0:
            imgutils.show_heatmaps(img_np, heatmaps_pred_np, num_fig=1)
            imgutils.show_heatmaps(img_np, heatmaps_np, num_fig=2)
            plt.pause(0.5)
        print('===================================================')

        if (step % 300 == 0) and (step > 0):
            suffix_epoch = i
            suffix_step = step
            path_ckpt_torch = 'models/ckpt_hg_torch_EPOCH' + str(suffix_epoch) + 'STEP' + str(
                suffix_step) + '.tar'
            torch.save({
                'epoch': i,
                'step': step,
                'model_state_dict': net_hg_torch.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, path_ckpt_torch)
            print('===============CHECKPOINT PARAMS SAVED===============')
        del img, heatmaps, pts, heatmaps_pred_copy, heatmaps_copy, img_copy, heatmaps_pred_np, heatmaps_np, img_np

    # ================================== Save checkpoint each epoch ==================================
    suffix_epoch = i
    suffix_step = step
    path_ckpt_torch = 'models/ckpt_hg_torch_allEPOCH' + str(suffix_epoch) + '.tar'
    torch.save({
        'epoch': i,
        'step': step,
        'model_state_dict': net_hg_torch.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path_ckpt_torch)
    print('===============CHECKPOINT PARAMS SAVED===============')
plt.ioff()

print('Training Complete')
