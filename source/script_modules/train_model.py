import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from PIL import Image

from ..models.StylizationModel import StylizationModel
from ..models.FeatureLossModel import FeatureLossModel
from .. import util


# TODO: Constants -- tune later
IMAGE_SIZE = 256
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 1
FEATURE_WEIGHT = 17
STYLE_WEIGHT = 50
MODEL_PATH = "saved_models/model.pth"

# Use CUDA if it's available.
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


model = StylizationModel()

# Load the pre-trained feature loss model.
feature_loss_model = FeatureLossModel([(1, 2), (2, 2), (3, 3), (4, 3)])

# Load dummy training data set
train_transform = transforms.Compose(
    [
        # transforms.Scale(IMAGE_SIZE),
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        # normalized based on pretrained torchvision models
        # https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
# this training set below is filler -- the actual paper uses the COCO dataset
# TODO: replace trainset with appropriate training set
trainset = torchvision.datasets.ImageFolder("content", train_transform)
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE)
train_loader_len = len(train_loader)

# Style target image (using starry night for now)
style_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
style = Image.open("style/starrynight.jpg")
style = style_transform(style)
style = Variable(style.repeat(BATCH_SIZE, 1, 1, 1))
vgg_style = feature_loss_model(style)

# Compute Gram matrices for the style target image
gram_style = [util.gram_matrix(f).to(device) for f in vgg_style]

optimizer = Adam(model.parameters(), LEARNING_RATE)

# This is equivalent to the squared normalized euclidean distance.
euclidean_distance = torch.nn.MSELoss()

# Send everything to the device.
model = model.to(device)
feature_loss_model = feature_loss_model.to(device)

model.train()
for e in range(EPOCHS):
    epoch_start_time = time.time()
    for batch_index, (x, _) in enumerate(train_loader):
        x = x.to(device)
        batch_start_time = time.time()
        optimizer.zero_grad()

        # Get the stylized output.
        y_hat = model(x)

        # Get the feature representations for the input and output.
        vgg_y_hat = feature_loss_model(y_hat)
        vgg_x = feature_loss_model(x)

        # Compute the feature reconstruction loss.
        feature_loss = FEATURE_WEIGHT * euclidean_distance(vgg_y_hat[1], vgg_x[1])

        # Compute the style reconstruction loss.
        style_loss = 0.0
        gram_y_hat = [util.gram_matrix(f) for f in vgg_y_hat]
        len_x = len(x)
        for index in range(len(gram_y_hat)):
            # Compute the squared Frobenius norm, which is equivalent to the
            # squared Euclidean norm, so we can reuse the previous criterion.
            style_loss += euclidean_distance(
                gram_y_hat[index], gram_style[index][:len_x]
            )
        style_loss *= STYLE_WEIGHT

        total_loss = feature_loss + style_loss
        total_loss.backward()
        optimizer.step()

        print(
            f"Batch {batch_index} of {train_loader_len} took {time.time() - batch_start_time} seconds."
        )
        print(f"Feature loss: {feature_loss.data.item()}")
        print(f"Style loss: {style_loss.data.item()}")

    print(f"Epoch: {e + 1} took {time.time() - epoch_start_time} seconds.")

# Save the trained model.
model.eval()
torch.save(model.state_dict(), MODEL_PATH)
