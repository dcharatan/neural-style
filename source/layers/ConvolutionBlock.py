import torch
import torch.nn as nn


class ConvolutionBlock(torch.nn.Module):
    """This is just a sequence of convolution, batch normalization and ReLU."""

    reflection_pad: nn.ReflectionPad2d
    conv2d: nn.Conv2d

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvolutionBlock, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, image: torch.Tensor):
        pad = self.reflection_pad(image)
        conv = self.conv2d(pad)
        return conv
