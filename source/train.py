import torch
import torch.nn as nn
import numpy as np
from .models.StylizationModel import StylizationModel

# Use CUDA if it's available.
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

# Try the model with random inputs to make sure the sizes work out.
input = torch.from_numpy(np.random.random((1, 3, 256, 256))).float()
model = StylizationModel()
output = model(input)