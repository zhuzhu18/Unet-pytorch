import torch
import torch.nn as nn
from parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.inconv = double_conv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

        self.up1 = up(512, 256, bilinear)
        self.up2 = up(256, 128, bilinear)
        self.up3 = up(128, 64, bilinear)
        self.up4 = up(64, 64, bilinear)

        self.outconv = outconv(64, n_classes)
        self.init()
    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x4, x5)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = self.outconv(x)

        return torch.sigmoid(x)

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, std=0.1)
                m.bias.data.zero_()
