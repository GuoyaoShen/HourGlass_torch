'''
(c) Guoyao Shen
https://github.com/GuoyaoShen/HourGlass_torch
implement 'MULTI-SCALE SUPERVISED NETWORK FOR HUMAN POSE ESTIMATION'
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    '''
    C_in: planes_in
    C_out: 2 * planes_mid
    '''
    expansion = 2
    def __init__(self, planes_in, planes_mid, stride=1):
        super(Bottleneck, self).__init__()
        self.expansion = 2
        self.bn1 = nn.BatchNorm2d(planes_in)
        self.conv1 = nn.Conv2d(planes_in, planes_mid, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes_mid)
        self.conv2 = nn.Conv2d(planes_mid, planes_mid, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes_mid)
        self.conv3 = nn.Conv2d(planes_mid, planes_mid * self.expansion, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        self.extra = nn.Sequential()
        if planes_in != planes_mid * self.expansion:
            self.extra = nn.Sequential(
                nn.Conv2d(planes_in, planes_mid * self.expansion, kernel_size=1, stride=1, bias=True),
                nn.BatchNorm2d(planes_mid * self.expansion)
            )

    def forward(self, x):

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        residual = self.extra(x)

        out += residual

        return out

class Hourglass(nn.Module):
    '''
    No Change channels, always 2*planes
    '''
    def __init__(self, block, num_blocks, planes, depth):
        '''
        :param block:
        :param num_blocks:
        :param planes: indicate the planes in the middle, i.e. planes_mid
        :param depth:
        '''
        super(Hourglass, self).__init__()
        self.depth = depth
        self.hg = self._make_hourglass(block, num_blocks, planes, depth)

    def _stack_residual(self, block, num_blocks, planes):
        '''
        stack block for number of num_block
        No Change channels
        '''
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hourglass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._stack_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._stack_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hourglass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        # print('img.H&W', n, ':', x.shape)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hourglass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hourglass_forward(self.depth, x)

class HourglassNet(nn.Module):
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16, num_features=64):
        super(HourglassNet, self).__init__()
        self.num_stacks = num_stacks
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_expansion = block.expansion

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.hourglass = Hourglass(block, num_blocks, self.num_features, 4)

        self.bn_in = nn.BatchNorm2d(3)
        self.conv_in = nn.Conv2d(3, self.num_expansion*self.num_features, kernel_size=7, stride=2, padding=3, bias=True)

        self.bn_attach = nn.BatchNorm2d(self.num_expansion*self.num_features)
        # self.conv_attach = nn.Conv2d(self.num_expansion*self.num_features, self.num_classes,
        #                              kernel_size=1, bias=True)
        self.conv_attach = nn.Sequential(
            nn.Conv2d(self.num_expansion * self.num_features, self.num_expansion * self.num_features,
                      kernel_size=1, bias=True),
            nn.Conv2d(self.num_expansion * self.num_features, self.num_classes,
                      kernel_size=1, bias=True)
        )

        self.bn_fc = nn.BatchNorm2d(self.num_stacks*self.num_classes)
        # self.fc = nn.Conv2d(self.num_stacks*self.num_classes, self.num_classes,
        #                              kernel_size=1, bias=True)
        self.fc = nn.Sequential(
            nn.Conv2d(self.num_stacks * self.num_classes, self.num_stacks * self.num_classes,
                      kernel_size=1, bias=True),
            nn.Conv2d(self.num_stacks * self.num_classes, self.num_classes,
                      kernel_size=1, bias=True)
        )

    def forward(self, x):
        out_list = []

        # layer-in
        x = self.bn_in(x)
        x = self.relu(x)
        x = self.conv_in(x)
        x = self.maxpool(x)  # C: self.num_expansion*self.num_features
        # print('after pool:', x.shape)

        # layer stack hourglass
        for i in range(self.num_stacks):
            y = self.bn_attach(x)
            y = self.relu(y)
            y = self.hourglass(y)
            htmap = self.conv_attach(y)
            out_list.append(htmap)
            x = x + y

        # final out layer
        out = torch.cat(out_list, dim=1)
        out = self.bn_fc(out)
        out = self.relu(out)
        out = self.fc(out)
        out_list.append(out)
        return out_list

def hourglass(**kwargs):
    model = Hourglass(Bottleneck, num_blocks=kwargs['num_blocks'], planes=kwargs['planes'], depth=kwargs['depth'])
    return model

def hgnet_torch(**kwargs):
    model = HourglassNet(Bottleneck, num_stacks=kwargs['num_stacks'], num_blocks=kwargs['num_blocks'],
                      num_classes=kwargs['num_classes'], num_features=kwargs['num_features'])
    return model
