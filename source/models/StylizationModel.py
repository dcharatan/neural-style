import torch
import torch.nn as nn
from ..layers.ConvolutionBlock import ConvolutionBlock
from ..layers.TransposeConvolutionBlock import TransposeConvolutionBlock
from ..layers.ResidualBlock import ResidualBlock
from ..validation.validators import is_input_image
from typing import List


class StylizationModel(nn.Module):
    """This is the stylization network described here:
    https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
    """

    down1: ConvolutionBlock
    down2: ConvolutionBlock
    down3: ConvolutionBlock
    residual_blocks: List[ResidualBlock]
    up1: TransposeConvolutionBlock
    up2: TransposeConvolutionBlock
    up3: TransposeConvolutionBlock
    tanh: nn.Tanh

    def __init__(self) -> None:
        super(StylizationModel, self).__init__()
        self.down1 = ConvolutionBlock(3, 32, 9, 1)
        self.down2 = ConvolutionBlock(32, 64, 3, 2)
        self.down3 = ConvolutionBlock(64, 128, 3, 2)
        self.residual_blocks = [ResidualBlock(128) for _ in range(5)]
        self.up1 = TransposeConvolutionBlock(128, 64, 3, 2)
        self.up2 = TransposeConvolutionBlock(64, 32, 3, 2)
        self.up3 = TransposeConvolutionBlock(32, 3, 3, 1, include_relu=False)
        self.tanh = nn.Tanh()

    def forward(self, image: torch.Tensor):
        assert is_input_image(image)

        # Run through the down convolution layers.
        down1 = self.down1(image)
        down2 = self.down2(down1)
        down3 = self.down3(down2)

        # Run through the residual layers.
        residual = down3
        for i in range(5):
            residual = self.residual_blocks[i](residual)

        # Run through the up convolution layers.
        up1 = self.up1(residual)
        up2 = self.up2(up1)
        up3 = self.up3(up2)

        # Scale the results to [0, 255].
        return self.tanh(up3) * 255