from settings import *


class DeconvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activate_last=True):
        super(DeconvResBlock, self).__init__()
        self.activate_last = activate_last
        self.in_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(in_channels, in_channels, 3, 1, 1, bias=False)
        )
        self.in_bnorm = nn.BatchNorm2d(in_channels)
        self.out_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.out_bnorm = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.in_layer(x) + x
        out = self.in_bnorm(out)
        out = self.elu(out)
        out = self.out_layer(out)
        out = self.out_bnorm(out)
        if self.activate_last:
            return self.elu(out)
        return out


class ConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activate_last=True):
        super(ConvResBlock, self).__init__()
        self.activate_last = activate_last
        self.in_layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False)
        )
        self.in_bnorm = nn.BatchNorm2d(in_channels)
        self.out_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.out_bnorm = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.in_layer(x) + x
        out = self.in_bnorm(out)
        out = self.elu(out)
        out = self.out_layer(out)
        out = self.out_bnorm(out)
        if self.activate_last:
            return self.elu(out)
        return out
