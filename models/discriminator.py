import torch
import torch.nn as nn
from torchsummary import summary

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        self.ins = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.ins(x)
        x = self.relu(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            ConvBlock(out_channels, out_channels*2, kernel_size=4, stride=2, padding=1),
            ConvBlock(out_channels*2, out_channels*4, kernel_size=4, stride=2, padding=1),
            ConvBlock(out_channels*4, out_channels*8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(out_channels*8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

def disc_summary(shape):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    disc = Discriminator(4,64).to(device)
    print(summary(disc,shape))