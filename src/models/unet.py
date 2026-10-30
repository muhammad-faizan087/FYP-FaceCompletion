import torch
import torch.nn as nn

from src.models.unet_blocks import Double_Conv, DownBlock, UpBlock

class UNet(nn.Module):
  def __init__(self, in_channels=4, out_channels=3):
    super().__init__()

    self.down_conv1 = DownBlock(in_channels, 64)
    self.down_conv2 = DownBlock(64, 128)
    self.down_conv3 = DownBlock(128, 256)
    self.down_conv4 = DownBlock(256, 512)

    self.bottleneck = Double_Conv(512, 1024)

    self.up_conv1 = UpBlock(1024, 512, 512)
    self.up_conv2 = UpBlock(512, 256, 256)
    self.up_conv3 = UpBlock(256, 128, 128)
    self.up_conv4 = UpBlock(128, 64, 64)

    self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

  def forward(self, x):
    c1, p1 = self.down_conv1(x)
    c2, p2 = self.down_conv2(p1)
    c3, p3 = self.down_conv3(p2)
    c4, p4 = self.down_conv4(p3)

    b = self.bottleneck(p4)

    u1 = self.up_conv1(b, c4)
    u2 = self.up_conv2(u1, c3)
    u3 = self.up_conv3(u2, c2)
    u4 = self.up_conv4(u3, c1)

    out = self.out(u4)

    return out
