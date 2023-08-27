import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from utils.data_load import visualize

cuda = True if torch.cuda.is_available() else False

# def weights_init_normal(m):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find("BatchNorm2d") != -1:
#         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#         torch.nn.init.constant_(m.bias.data, 0.0)

def weights_init_normal(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        colordim = 3

        self.conv1_1 = nn.Conv2d(colordim, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1_shortcut = nn.Conv2d(colordim, 64, 1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.conv2_shortcut = nn.Conv2d(64, 128, 1, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_shortcut = nn.Conv2d(128, 256, 1, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        # self.conv4_shortcut = nn.Conv2d(256, 512, 1, stride=2)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.bn5_1 = nn.BatchNorm2d(512+256)
        self.conv5_1 = nn.Conv2d(512+256, 256, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(256)
        self.conv5_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv5_shortcut = nn.Conv2d(512+256, 256, 1) 

        self.bn6_1 = nn.BatchNorm2d(256+128)
        self.conv6_1 = nn.Conv2d(256+128, 128, 3, padding=1)
        self.bn6_2 = nn.BatchNorm2d(128)
        self.conv6_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6_shortcut = nn.Conv2d(256+128, 128, 1) 

        self.bn7_1 = nn.BatchNorm2d(128+64)
        self.conv7_1 = nn.Conv2d(128+64, 64, 3, padding=1)
        self.bn7_2 = nn.BatchNorm2d(64)
        self.conv7_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv7_shortcut = nn.Conv2d(128+64, 64, 1) 

        self.conv8 = nn.Conv2d(64, 1, 1)

    def forward(self, x1):
        Z1 = self.conv1_2(F.relu(self.bn1(self.conv1_1(x1))))
        Z2 = self.conv2_2(F.relu(self.bn2_2(self.conv2_1(F.relu(self.bn2_1(Z1))))))
        Z3 = self.conv3_2(F.relu(self.bn3_2(self.conv3_1(F.relu(self.bn3_1(Z2))))))
        Z4 = self.conv4_2(F.relu(self.bn4_2(self.conv4_1(F.relu(self.bn4_1(Z3))))))
        Z4u = self.upsample(Z4)
        Z4c = torch.cat((Z3, Z4u), dim=1) # concat on the 2nd dimension, which is the num of channels
        Z5 = self.conv5_2(F.relu(self.bn5_2(self.conv5_1(F.relu(self.bn5_1(Z4c))))))
        Z5u = self.upsample(Z5)
        Z5c = torch.cat((Z2, Z5u), dim=1) # concat on the 2nd dimension, which is the num of channels
        Z6 = self.conv6_2(F.relu(self.bn6_2(self.conv6_1(F.relu(self.bn6_1(Z5c))))))
        Z6u = self.upsample(Z6)
        Z6c = torch.cat((Z1, Z6u), dim=1) # concat on the 2nd dimension, which is the num of channels
        Z7 = self.conv7_2(F.relu(self.bn7_2(self.conv7_1(F.relu(self.bn7_1(Z6c))))))
        Z8 = self.conv8(Z7)
        img = F.sigmoid(Z8)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 256 // 8
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        # self.adv_layer = nn.Sequential(nn.Conv2d(128, 1, 3, 1, 1), nn.Sigmoid())

    def forward(self, img):
        # input = torch.cat((satellite_img, img), dim = 1)
        out = self.model(img)
        out = out.reshape((out.shape[0], -1))
        validity = self.adv_layer(out)

        return validity
