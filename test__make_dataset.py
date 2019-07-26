import numpy as np
import torch
import torch.utils.data as datatorch
import skimage
import matplotlib.pyplot as plt
import cv2
import json

from utils import imgutils
from datasets.mpii import MPII

import os


if __name__ == '__main__':
    # ========== Test dataset ==========
    ds_torch = MPII(use_scale=True, use_flip=True, use_rand_color=True)
    data_loader = datatorch.DataLoader(ds_torch, batch_size=1, shuffle=False)
    for step, (img, heatmaps, pts) in enumerate(data_loader):
        print('img', img.shape)
        print('heatmaps', heatmaps.shape)
        print('pts', pts.shape)

        img = np.transpose(img.squeeze().detach().numpy(), (1, 2, 0))
        # img = np.fliplr(img)
        heatmaps = np.transpose(heatmaps.squeeze().detach().numpy(), (1, 2, 0))
        pts = pts.squeeze().detach().numpy()
        # print('pts', pts)
        print('===========================================================')
        # imgutils2.show_heatmaps(img, heatmaps)
        img = skimage.transform.resize(img, (64, 64))
        imgutils.show_stack_joints(img, pts, c=[0, 0], draw_lines=True, num_fig=1)

