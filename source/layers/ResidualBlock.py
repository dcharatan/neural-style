import torch
import torch.nn as nn
from ..layers.ConvolutionBlock import ConvolutionBlock


class ResidualBlock(nn.Module):
    """This is a residual block as defined in "Perceptual Losses for Real-Time
    Style Transfer and Super-Resolution: Supplementary Material" by Johnson et
    al. See https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
    """

    conv1: ConvolutionBlock
    relu: nn.ReLU
    conv2: ConvolutionBlock

    def __init__(self, num_channels: int) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvolutionBlock(
            num_channels, num_channels, kernel_size=3, stride=1
        )
        self.relu = nn.ReLU()
        self.conv2 = ConvolutionBlock(
            num_channels, num_channels, kernel_size=3, stride=1
        )

    def forward(self, image: torch.Tensor):
        return self.conv2(self.relu(self.conv1(image))) + image