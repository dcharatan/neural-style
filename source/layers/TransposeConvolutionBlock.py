import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class TransposeConvolutionBlock(nn.Module):
    """This is just a sequence of transpose convolution, batch normalization and
    ReLU.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    upsample: int
    reflection_pad: nn.ReflectionPad2d
    conv2d: nn.ConvTranspose2d

    def __init__(self, in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int, 
        upsample=None)-> None:

        super(TransposeConvolutionBlock, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.stride = stride

    def forward(self, image: torch.Tensor):
        if self.upsample:
            image = self.upsample_layer(image)
        
        pad = self.reflection_pad(image)
        conv = self.conv2d(pad)

        return conv
