import torch.nn as nn


class ConvBN(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Mish(inplace=True)
        )

    def forward(self, x):
        return self.conv_bn(x)


class Conv3BN(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv3_bn = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.Mish(inplace=True)
        )

    def forward(self, x):
        return self.conv3_bn(x)

class UpConv(nn.Module):
    """(up convolution => [BN] => ReLU)"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                ConvBN(in_channels, out_channels)
            )
        else:
            self.up_conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.Mish(inplace=True)
            )

    def forward(self, x):
        return self.up_conv(x)

