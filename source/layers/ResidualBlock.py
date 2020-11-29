import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """This is a residual block as defined in "Perceptual Losses for Real-Time
    Style Transfer and Super-Resolution: Supplementary Material" by Johnson et
    al. See https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
    """

    conv1: nn.Conv2d
    bn1: nn.BatchNorm2d
    relu: nn.ReLU
    conv2: nn.Conv2d
    bn2: nn.BatchNorm2d

    def __init__(self, num_channels: int) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            num_channels, num_channels, 3, 1, 1, padding_mode="reflect"
        )
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            num_channels, num_channels, 3, 1, 1, padding_mode="reflect"
        )
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, image: torch.Tensor):
        # Sequentially run through the layers.
        conv1 = self.conv1(image)
        bn1 = self.bn1(conv1)
        relu = self.relu(bn1)
        conv2 = self.conv2(relu)
        bn2 = self.bn2(conv2)

        # Return the input plus the final layer's output.
        return image + bn2
