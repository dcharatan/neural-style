import torch
import torch.nn as nn


class ConvolutionBlock(torch.nn.Module):
    """This combines convolution with instance normalization."""

    reflection_pad: nn.ReflectionPad2d
    conv2d: nn.Conv2d

    def __init__(self, in_channels, out_channels, kernel_size, stride, no_norm=False):
        super(ConvolutionBlock, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.instance_norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.no_norm = no_norm

    def forward(self, image: torch.Tensor):
        pad = self.reflection_pad(image)
        conv = self.conv2d(pad)
        if self.no_norm:
            return conv
        return self.instance_norm(conv)
