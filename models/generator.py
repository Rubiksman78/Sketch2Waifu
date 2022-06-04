import torch
import torch.nn as nn
from torchsummary import summary

class Generator_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Generator_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.ins = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.ins(x)
        x = self.relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.ref = nn.ReflectionPad2d(3)
        self.conv1 = Generator_block(in_channels, out_channels, kernel_size=7, stride=1,padding=0)
        self.conv2 = Generator_block(out_channels, out_channels*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = Generator_block(out_channels*2, out_channels*4, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.decode = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=out_channels,out_channels=out_channels//2,kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(4),
            nn.Conv2d(in_channels=out_channels//2, out_channels=3, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        x = self.decode(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self,dim,dilation=1,use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=3,padding=0,dilation=dilation,bias=not use_spectral_norm),use_spectral_norm),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=3,padding=0,dilation=1,bias=not use_spectral_norm),use_spectral_norm),
            nn.InstanceNorm2d(dim),
        )

    def forward(self,x):
        output = x + self.convblock(x)
        return output

def spectral_norm(module,use_spectral_norm=True):
    if use_spectral_norm:
        return nn.utils.spectral_norm(module)
    return module

class Generator(nn.Module):
    def __init__(self, input_size,rs_blocks=8):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.encoder = Encoder(input_size,64)
        self.decoder = Decoder(256,128)
        blocks = []
        for _ in range(rs_blocks):
            block = ResnetBlock(256,2)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = (torch.tanh(x) +1)/2
        return x 

def gen_summary(shape):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gen = Generator(4).to(device)
    print(summary(gen,shape))