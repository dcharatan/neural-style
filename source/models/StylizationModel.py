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

    dim: int
    conv1: ConvolutionBlock
    in1: nn.InstanceNorm2d
    conv2: ConvolutionBlock
    in2: nn.InstanceNorm2d
    conv3: ConvolutionBlock
    in3: nn.InstanceNorm2d
    residual_blocks: List[ResidualBlock]
    deconv1: TransposeConvolutionBlock
    in4: nn.InstanceNorm2d
    deconv2: TransposeConvolutionBlock
    in5: nn.InstanceNorm2d
    deconv3: TransposeConvolutionBlock
    relu: nn.ReLU

    def __init__(self, block_dim=128)-> None:
        super(StylizationModel, self).__init__()
        # Initial convolution layers
        self.dim = block_dim
        self.conv1 = ConvolutionBlock(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvolutionBlock(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvolutionBlock(64, self.dim, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.residual_blocks = [ResidualBlock(self.dim) for _ in range(5)]
        # Upsampling Layers
        self.deconv1 = TransposeConvolutionBlock(self.dim, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = TransposeConvolutionBlock(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvolutionBlock(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, image: torch.Tensor):
        conv1 = self.relu(self.in1(self.conv1(image)))
        conv2 = self.relu(self.in2(self.conv2(conv1)))
        conv3 = self.relu(self.in3(self.conv3(conv2)))
        
        residual = conv3
        for i in range(5):
            residual = self.residual_blocks[i](residual)

        deconv1 = self.relu(self.in4(self.deconv1(residual)))
        deconv2 = self.relu(self.in5(self.deconv2(deconv1)))
        deconv3 = self.deconv3(deconv2)

        return deconv3