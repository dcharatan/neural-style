import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):
    """This is just a sequence of convolution, batch normalization and ReLU."""

    conv: nn.Conv2d
    bn: nn.BatchNorm2d
    relu: nn.ReLU

    def __init__(
        self, in_channels: int, out_channels: int, filter_size: int, stride: int
    ) -> None:
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            filter_size,
            stride,
            filter_size // 2,  # Keep the image size the same.
            padding_mode="reflect",
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, image: torch.Tensor):
        conv = self.conv(image)
        bn = self.bn(conv)
        return self.relu(bn)
