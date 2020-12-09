from source.layers.UpsampleBlock import UpsampleBlock
import torch
import torch.nn as nn
from ..layers.ConvolutionBlock import ConvolutionBlock
from ..layers.ResidualBlock import ResidualBlock


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
        # Normalize the image on a per-channel, per-image basis.
        mu = torch.mean(image, dim=[2, 3])
        std = torch.std(image, dim=[2, 3])
        x = 0.5 + (image - mu[:, :, None, None]) / (2 * std[:, :, None, None])

        x = self.down_convolution(x)
        x = self.residual(x)
        x = self.up_convolution(x)
        return torch.tanh(x) * 0.5 + 0.5