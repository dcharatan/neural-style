import torch
from ..models.StylizationModel import StylizationModel
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

MODEL_PATH = "saved_models/model.pth"
INPUT_PATH = "test/scene.jpg"
OUTPUT_PATH = "result/test1.jpg"
IMAGE_SIZE = 256
BATCH_SIZE = 4

# Load the pre-trained model.
model = StylizationModel()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

train_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))]
)

# Run the image through the model.
test = Image.open(INPUT_PATH).convert("RGB")
test = train_transform(test)
test = torch.Tensor(test.repeat(BATCH_SIZE, 1, 1, 1))
output_test = model(test).detach().numpy()
image = output_test[0, :, :, :]
image = np.transpose(image, (1, 2, 0))

Image.fromarray(np.uint8(image * 255)).save(OUTPUT_PATH)