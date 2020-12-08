from source.image import get_data_loader_transform, load_image, save_image
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import time
from ..models.StylizationModel import StylizationModel
from ..models.FeatureLossModel import FeatureLossModel
from .. import util
import os


# TODO: Constants -- tune later
IMAGE_SIZE = 256
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 1
FEATURE_WEIGHT = 1
STYLE_WEIGHT = 20000
MODEL_FOLDER = "saved_models"
STYLE_IMAGE = "style/starrynight.jpg"

# Use CUDA if it's available.
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
style_model = StylizationModel().to(device)
feature_model = FeatureLossModel([(1, 2), (2, 2), (3, 3), (4, 3)]).to(device)

# Set up the data loader.
trainset = torchvision.datasets.ImageFolder(
    "content", get_data_loader_transform(IMAGE_SIZE)
)
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE)
train_loader_len = len(train_loader)

# Load the style image.
style = load_image(STYLE_IMAGE).repeat(BATCH_SIZE, 1, 1, 1).to(device)
vgg_style = feature_model(style)
gram_style = {name: util.gram_matrix(tensor) for name, tensor in vgg_style.items()}

# This is equivalent to the squared normalized euclidean distance.
norm = torch.nn.MSELoss()
optimizer = Adam(style_model.parameters(), LEARNING_RATE)
style_model.train()
for e in range(EPOCHS):
    epoch_start_time = time.time()
    for batch_index, (x, _) in enumerate(train_loader):
        # The image x is normalized to [0, 1].
        batch_start_time = time.time()
        x = x.to(device)
        optimizer.zero_grad()

        # Get the stylized output.
        y_hat = style_model(x)
        mu = torch.mean(y_hat)
        std = torch.std(y_hat)
        mini = torch.min(y_hat)
        maxi = torch.max(y_hat)

        # Get the feature representations for the input and output.
        vgg_y_hat = feature_model(y_hat)
        vgg_x = feature_model(x)

        # Compute the feature reconstruction loss.
        feature_loss = FEATURE_WEIGHT * norm(vgg_y_hat["relu2_2"], vgg_x["relu2_2"])

        # Compute the style reconstruction loss.
        style_loss = 0.0
        gram_y_hat = {
            name: util.gram_matrix(tensor) for name, tensor in vgg_y_hat.items()
        }
        for name in vgg_y_hat.keys():
            style_loss += norm(gram_y_hat[name], gram_style[name])
        style_loss *= STYLE_WEIGHT

        total_loss = feature_loss + style_loss
        total_loss.backward()
        optimizer.step()

        if batch_index % 50 == 0:
            print(
                f"Batch {batch_index} of {train_loader_len} took {time.time() - batch_start_time} seconds."
            )
            print(f"Feature loss: {feature_loss.data.item()}")
            print(f"Style loss: {style_loss.data.item()}")
        if batch_index % 200 == 0:
            save_image(f"tmp_{e}_{batch_index}.png", y_hat[0].cpu().detach().numpy())
        if batch_index % 1000 == 0:
            torch.save(
                style_model.state_dict(),
                os.path.join(MODEL_FOLDER, f"tmp_model_{e}_{batch_index}.pth"),
            )

    print(f"Epoch: {e + 1} took {time.time() - epoch_start_time} seconds.")

# Save the trained style_model.
style_model.eval()
torch.save(style_model.state_dict(), os.path.join(MODEL_FOLDER, "final_model.pth"))
