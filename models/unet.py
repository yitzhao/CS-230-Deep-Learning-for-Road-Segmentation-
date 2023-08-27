import torch.nn as nn
import torch.nn.functional as F
import torch
from numpy.linalg import svd
from numpy.random import normal
from math import sqrt


class UNet(nn.Module):
    def __init__(self,colordim =3):
        super(UNet, self).__init__()

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
        
        self._initialize_weights()

    def forward(self, x1):
        Z1 = self.conv1_2(F.relu(self.bn1(self.conv1_1(x1))))
        # Z1 = torch.add(Z1, self.conv1_shortcut(x1))
        # print("Z1 size: ")
        # print(Z1.size())

        Z2 = self.conv2_2(F.relu(self.bn2_2(self.conv2_1(F.relu(self.bn2_1(Z1))))))
        # Z2 = torch.add(Z2, self.conv2_shortcut(Z1))
        # print("Z2 size: ")
        # print(Z2.size())

        Z3 = self.conv3_2(F.relu(self.bn3_2(self.conv3_1(F.relu(self.bn3_1(Z2))))))
        # Z3 = torch.add(Z3, self.conv3_shortcut(Z2))
        # print("Z3 size: ")
        # print(Z3.size())

        Z4 = self.conv4_2(F.relu(self.bn4_2(self.conv4_1(F.relu(self.bn4_1(Z3))))))
        Z4u = self.upsample(Z4)
        Z4c = torch.cat((Z3, Z4u), dim=1) # concat on the 2nd dimension, which is the num of channels
        # print("Z4 size before upsample and concatenation: ")
        # print(Z4.size())
        # print("Z4 size after upsample and concatenation: ")
        # print(Z4c.size())

        Z5 = self.conv5_2(F.relu(self.bn5_2(self.conv5_1(F.relu(self.bn5_1(Z4c))))))
        # Z5 = torch.add(Z5, self.conv5_shortcut(Z4c))
        Z5u = self.upsample(Z5)
        Z5c = torch.cat((Z2, Z5u), dim=1) # concat on the 2nd dimension, which is the num of channels
        # print("Z5 size before upsample and concatenation: ")
        # print(Z5.size())
        # print("Z5 size after upsample and concatenation: ")
        # print(Z5c.size())

        Z6 = self.conv6_2(F.relu(self.bn6_2(self.conv6_1(F.relu(self.bn6_1(Z5c))))))
        # Z6 = torch.add(Z6, self.conv6_shortcut(Z5c))
        Z6u = self.upsample(Z6)
        Z6c = torch.cat((Z1, Z6u), dim=1) # concat on the 2nd dimension, which is the num of channels
        # print("Z6 size before upsample and concatenation: ")
        # print(Z6.size())
        # print("Z6 size after upsample and concatenation: ")
        # print(Z6c.size())

        Z7 = self.conv7_2(F.relu(self.bn7_2(self.conv7_1(F.relu(self.bn7_1(Z6c))))))
        # Z7 = torch.add(Z7, self.conv7_shortcut(Z6c))
        # print("Z7 size: ")
        # print(Z7.size())

        Z8 = self.conv8(Z7) # final layer
        # print("Z8 size: ")
        # print(Z8.size())

        return F.sigmoid(Z8)





    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, sqrt(2. / n))
                # m.weight.data = m.weight.data

                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if torch.cuda.is_available():
    unet = UNet().cuda()