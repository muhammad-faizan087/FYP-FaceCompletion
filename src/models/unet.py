import torch
import torch.nn as nn

from src.models.unet_blocks import Double_Conv, DownBlock, UpBlock


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_channels=64):
        super().__init__()

        self.B = base_channels

        self.down_conv1 = DownBlock(in_channels, self.B)
        self.down_conv2 = DownBlock(self.B, self.B * 2)
        self.down_conv3 = DownBlock(self.B * 2, self.B * 4)
        self.down_conv4 = DownBlock(self.B * 4, self.B * 8)

        self.bottleneck = Double_Conv(self.B * 8, self.B * 16)

        self.up_conv1 = UpBlock(self.B * 16, self.B * 8, self.B * 8)
        self.up_conv2 = UpBlock(self.B * 8, self.B * 4, self.B * 4)
        self.up_conv3 = UpBlock(self.B * 4, self.B * 2, self.B * 2)
        self.up_conv4 = UpBlock(self.B * 2, self.B, self.B)

        self.out = nn.Conv2d(
            in_channels=self.B,
            out_channels=out_channels,
            kernel_size=1
        )

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
