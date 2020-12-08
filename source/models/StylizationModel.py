from source.layers.UpsampleBlock import UpsampleBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.ConvolutionBlock import ConvolutionBlock
from ..layers.ResidualBlock import ResidualBlock
from ..validation.validators import is_input_image
from typing import List


class StylizationModel(nn.Module):
    """This is the stylization network described here:
    https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
    """

    def __init__(self, block_dim=128) -> None:
        super(StylizationModel, self).__init__()

        self.down_convolution = nn.Sequential(
            ConvolutionBlock(3, 32, kernel_size=9, stride=1),
            nn.ReLU(),
            ConvolutionBlock(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            ConvolutionBlock(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.residual = nn.Sequential(*[ResidualBlock(128) for _ in range(5)])
        self.up_convolution = nn.Sequential(
            UpsampleBlock(128, 64, 3, 2),
            nn.ReLU(),
            UpsampleBlock(64, 32, 3, 2),
            nn.ReLU(),
            ConvolutionBlock(32, 3, 9, 1),
        )

    def forward(self, image: torch.Tensor):
        x = self.down_convolution(image)
        x = self.residual(x)
        x = self.up_convolution(x)
        return F.tanh(x) * 0.5 + 0.5