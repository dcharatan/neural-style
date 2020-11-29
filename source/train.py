import torch
import numpy as np
from .models.StylizationModel import StylizationModel
from .models.FeatureLossModel import FeatureLossModel

# Use CUDA if it's available.
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

# Try the model with random inputs to make sure the sizes work out.
input = torch.from_numpy(np.random.random((1, 3, 256, 256))).float()
model = StylizationModel()
output = model(input)

# Try the feature loss model to make sure the hooks work.
feature_loss_model = FeatureLossModel([(1, 2), (2, 2), (3, 3), (4, 3)])
vgg_output = feature_loss_model(input)
