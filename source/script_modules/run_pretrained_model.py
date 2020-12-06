import torch
from ..models.StylizationModel import StylizationModel
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

MODEL_PATH = "saved_models/model.pth"
INPUT_PATH = "test/test.jpg"
OUTPUT_PATH = "result/test.jpg"
IMAGE_SIZE = 256
BATCH_SIZE = 4

# Load the pre-trained model.
model = StylizationModel()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

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

# Run the image through the model.
test = Image.open(INPUT_PATH)
test = train_transform(test)
test = torch.Tensor(test.repeat(BATCH_SIZE, 1, 1, 1))
output_test = model(test).detach().numpy()
image = output_test[0, :, :, :]
image = np.transpose(image, (1, 2, 0))

Image.fromarray(np.uint8(image * 255)).save(OUTPUT_PATH)