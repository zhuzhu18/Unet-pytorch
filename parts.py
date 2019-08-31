import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    # size减4
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class down(nn.Module):
    # size先缩小一半, 再减4
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            double_conv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.mpconv(x)

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        self.bilinear = bilinear
        self.conv_trans = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.upconv = double_conv(2*in_ch, out_ch)

    def forward(self, front, later):
        if self.bilinear:
            later = F.interpolate(later, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            later = self.conv_trans(later)
        h_diff = front.size()[2] - later.size()[2]
        w_diff = front.size()[3] - later.size()[3]
        later = F.pad(later, pad=(w_diff//2,w_diff-w_diff//2,h_diff//2,h_diff-h_diff//2),
              mode='constant', value=0)
        x = torch.cat([front, later],dim=1)
        x = self.upconv(x)
        return x

class outconv(nn.Module):
    def __init__(self,in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x):
        return self.conv(x)