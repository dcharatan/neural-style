from source.image import load_image, save_image
import torch
from ..models.StylizationModel import StylizationModel
from PIL import Image
import torchvision.transforms as tf
import numpy as np

MODEL_PATH = "saved_models/starrynight_norm.pth"
INPUT_PATH = "test/stadt.jpg"
OUTPUT_PATH = "result/test1.jpg"
IMAGE_SIZE = 256
BATCH_SIZE = 4

# Load the pre-trained model.
model = StylizationModel()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Run the image through the model.
test = load_image(INPUT_PATH)
test = tf.Compose(
    [
        tf.Resize(IMAGE_SIZE),
        tf.CenterCrop(IMAGE_SIZE),
    ]
)(test)
output_test = model(test).detach().numpy()
save_image(OUTPUT_PATH, output_test[0, :, :, :])