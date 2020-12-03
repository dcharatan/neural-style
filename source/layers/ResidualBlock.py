import torch
import torch.nn as nn
from .ConvolutionBlock import ConvolutionBlock

class ResidualBlock(torch.nn.Module):
    """This is a residual block as defined in "Perceptual Losses for Real-Time
    Style Transfer and Super-Resolution: Supplementary Material" by Johnson et
    al. See https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
    """

    conv1: ConvolutionBlock
    in1: nn.InstanceNorm2d
    relu: nn.ReLU
    conv2: ConvolutionBlock
    in2: nn.InstanceNorm2d


    def __init__(self, num_channels: int) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvolutionBlock(num_channels, num_channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(num_channels, affine=True)
        self.conv2 = ConvolutionBlock(num_channels, num_channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(num_channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, image: torch.Tensor):
        residual = image
        conv = self.relu(self.in1(self.conv1(image)))
        in2 = self.in2(self.conv2(conv))
        out = in2 + residual

        return out