import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Tuple

# Uncomment this if VGG16 download does not work
# import torchvision.models
# from torchvision.models.vgg import model_urls


class FeatureLossModel(nn.Module):
    """This model is based on VGG16. However, instead of returning VGG16's
    output, it returns the"""

    vgg16: models.vgg.VGG
    features: List[torch.Tensor]

    def __init__(self) -> None:
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

        self.features = []
        self.register_hook()
        
    def hook(self, module, input, output):
        self.features.append(output)

    def register_hook(self):
        self.vgg16.features[3].register_forward_hook(self.hook)
        self.vgg16.features[8].register_forward_hook(self.hook)
        self.vgg16.features[15].register_forward_hook(self.hook)
        self.vgg16.features[22].register_forward_hook(self.hook)
        
    def forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Run VGG16, but instead of returning the classification output, return
        the intermediate feature layer outputs.
        """
        self.features = []
        self.vgg16(image)
        return self.features
