import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class TransposeConvolutionBlock(nn.Module):
    """This is just a sequence of transpose convolution, batch normalization and
    ReLU.
    """

    stride: int
    conv: nn.ConvTranspose2d
    bn: nn.BatchNorm2d
    relu: Optional[nn.ReLU]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: int,
        stride: int,
        include_relu: bool = True,
    ) -> None:
        super(TransposeConvolutionBlock, self).__init__()
        self.stride = stride
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            filter_size,
            stride,
            filter_size // 2,  # Keep the image size the same.
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU() if include_relu else None

    def forward(self, image: torch.Tensor):
        output_size = np.array(image.shape)
        output_size[2:4] *= self.stride
        conv = self.conv(image, output_size=output_size)
        bn = self.bn(conv)
        return self.relu(bn) if self.relu is not None else bn
