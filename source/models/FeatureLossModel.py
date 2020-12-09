import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as tf
from typing import List, Tuple, Dict

# Uncomment this if VGG16 download does not work
# import torchvision.models
# from torchvision.models.vgg import model_urls

VGG16_NORMALIZE = tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class FeatureLossModel(nn.Module):
    """This model is based on VGG16. However, instead of returning VGG16's
    output, it returns the"""

    vgg16: models.vgg.VGG
    desired_features: List[Tuple[int, int]]
    major: int
    minor: int
    features: Dict[str, torch.Tensor]

    def __init__(self, desired_features: List[Tuple[int, int]]) -> None:
        """desired_features should be a list of tuples of (major, minor), where
        major and minor are the one-indexed indices of the ReLU layers whose
        outputs should be outputted when the model is called. The major number
        indicates the index relative to the pooling operations, while the minor
        number indicates the index relative to convolution layers within a
        series of convolutions. The minor number resets each time a pooling
        layer is encountered.
        TL;DR: Populate desired_features with the same indices the authors use
        to refer to feature layers in the paper.
        """
        super(FeatureLossModel, self).__init__()

        # Uncomment this if VGG16 download does not work
        # model_urls['vgg16'] = model_urls['vgg16'].replace('https://', 'http://')

        # Load a pretrained VGG16 network with frozen parameters.
        self.vgg16 = models.vgg16(pretrained=True)
        for param in self.vgg16.parameters():
            param.requires_grad = False

        # Apply hooks to extract features.
        # desired_features is
        self.desired_features = desired_features
        self.minor = 1
        self.major = 1
        self.vgg16.apply(self.register_hook)
        self.features = {}

    def hook(self, name):
        def hook(module, input, output):
            self.features[name] = output

        return hook

    def register_hook(self, module):
        # Increment the minor number each time a Conv2d is encountered.
        if isinstance(module, nn.Conv2d):
            self.minor += 1

        # Reset the minor number and increment the major number each time a
        # MaxPool2d is encountered.
        if isinstance(module, nn.MaxPool2d):
            self.minor = 1
            self.major += 1

        # Register hooks for ReLU layers if they're in desired_features.
        if (
            isinstance(module, nn.ReLU)
            and (self.major, self.minor) in self.desired_features
        ):
            module.register_forward_hook(self.hook(f"relu{self.major}_{self.minor}"))

    def forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Run VGG16, but instead of returning the classification output, return
        the intermediate feature layer outputs.
        """
        self.features = {}

        # The input image is in the range [0, 1]. The pretrained VGG network
        # expects images to be normalized according to VGG16_NORMALIZE, however.
        self.vgg16(VGG16_NORMALIZE(image))
        return self.features