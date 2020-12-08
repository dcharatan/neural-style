import torch
import torch.nn as nn
from .ConvolutionBlock import ConvolutionBlock


class UpsampleBlock(nn.Module):
    """This increases resolution by upsampling and then convolving."""

    reflection_pad: nn.ReflectionPad2d
    conv: ConvolutionBlock

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        scale_factor: float,
    ) -> None:

        super(UpsampleBlock, self).__init__()
        self.upsample_layer = nn.Upsample(scale_factor=scale_factor)
        self.conv = ConvolutionBlock(in_channels, out_channels, kernel_size, 1)

    def forward(self, image: torch.Tensor):
        upsampled = self.upsample_layer(image)
        return self.conv(upsampled)
