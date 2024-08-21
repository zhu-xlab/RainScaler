from torch import nn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, '../deep_learning')
from models.constraints import *
import cv2
from models.unet_layers import *
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SRCNN(nn.Module):
    def __init__(self, num_channels=1, constraints='none', upsampling_factor=3):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
        self.pool = self.pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
        self.sigmoid = nn.Sigmoid()
        self.softshrink = nn.Softshrink(0.001)
        self.is_constraints = False
        self.unet = UNet(1, 1)
        if constraints == 'mult':
            self.constraints = MultDownscaleConstraints(upsampling_factor = upsampling_factor)
            self.is_constraints = True
        if constraints == 'add':
            self.constraints = AddDownscaleConstraints(upsampling_factor = upsampling_factor)
            self.is_constraints = True
        if constraints == 'scadd':
            self.constraints = ScAddDownscaleConstraints(upsampling_factor = upsampling_factor)
            self.is_constraints = True
        if constraints == 'soft':
            self.constraints = SoftmaxConstraints(upsampling_factor = upsampling_factor)
            self.is_constraints = True
        if constraints == 'global':
            self.constraints = GlobalConstraints()
            self.is_constraints = True

    def forward(self, x):
        """
        :param x: B,C,W,H
        :return:
        """
        raw = x
        #bias = torch.FloatTensor(x.size()).fill_(0.1).to(device)
        difference_0 = self.unet(x)
        #difference = self.pool(hr) - x

        weights = self.sigmoid(1000*difference_0) #+ bias
        #weights = self.softshrink(difference)

        x = x * weights

        out = torch.nn.functional.interpolate(input=x, scale_factor=3, mode='bicubic')
        out = self.relu(self.conv1(out))
        out = self.relu(self.conv2(out))
        #final output layer
        out = self.conv3(out)
        #optional renormalization layer
        if self.is_constraints:
            out = self.constraints(out, raw)

        return out, difference_0



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleTonv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutTonv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)