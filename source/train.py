import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from .models.StylizationModel import StylizationModel
from .models.FeatureLossModel import FeatureLossModel
from . import util


# TODO: Constants -- tune later
IMAGE_SIZE = 256
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 10
FEATURE_WEIGHT = 1
STYLE_WEIGHT = 1

# Use CUDA if it's available.
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

# Try the model with random inputs to make sure the sizes work out.
# input = torch.from_numpy(np.random.random((1, 3, 256, 256))).float()
model = StylizationModel()
# output = model(input)

# Try the feature loss model to make sure the hooks work.
feature_loss_model = FeatureLossModel([(1, 2), (2, 2), (3, 3), (4, 3)])
# vgg_output = feature_loss_model(input)

# Load dummy training data set
train_transform = transforms.Compose([
    transforms.Scale(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    # normalized based on pretrained torchvision models
    # https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])])
# this training set below is filler -- the actual paper uses the COCO dataset 
# TODO: replace trainset with appropriate training set 
trainset = torchvision.datasets.ImageFolder('content', train_transform) 
train_loader = DataLoader(trainset, batch_size = BATCH_SIZE)

# Style target image (using starry night for now)
style_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])
style = Image.open("./style/starrynight.jpg")
style = style_transform(style)
style = Variable(style.repeat(BATCH_SIZE, 1, 1, 1))
vgg_style = feature_loss_model(style)

# Compute Gram matrices for the style target image
gram_style = [util.gram_matrix(f) for f in vgg_style]

optimizer = Adam(model.parameters(), LEARNING_RATE)
euclidean_distance = torch.nn.MSELoss() # equivalent to the squared normalized euclidean dist

for e in range(EPOCHS):
    model.train()
    for i, (x, labels) in enumerate(train_loader):
        len_x = len(x)
        # Zero gradients
        optimizer.zero_grad()

        # Get potential stylized output
        y_hat = model(x)

        # Get feature representations
        vgg_y_hat = feature_loss_model(y_hat)
        vgg_x = feature_loss_model(x)

        # Feature reconstruction loss
        feature_loss = FEATURE_WEIGHT * euclidean_distance(vgg_y_hat[1], vgg_x[1])

        # Style reconstruction loss
        style_loss = 0.0
        gram_y_hat = [util.gram_matrix(f) for f in vgg_y_hat]
        for index in range(len(gram_y_hat)):
            # Compute the squared Frobenius norm, which is equivalent to the 
            # squared Euclidean norm, so we can reuse the previous criterion
            style_loss += euclidean_distance(gram_y_hat[index], gram_style[index][:len_x])
        style_loss *= STYLE_WEIGHT

        total_loss = feature_loss + style_loss
        print("Feature loss: %s" % feature_loss.data.item())
        print("Style loss: %s" % style_loss.data.item())
        total_loss.backward()
        optimizer.step()
    print("Epoch: %s" % str(e+1))

model.eval()
test = Image.open("./test/test.jpg")
test = train_transform(test)
test = Variable(test.repeat(BATCH_SIZE, 1, 1, 1))
output_test = model(test)
util.show_img(output_test.data[0])