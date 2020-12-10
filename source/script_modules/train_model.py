import os
import torch
import torchvision
import time
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader
from ..image import get_data_loader_transform, load_image, save_image
from ..models.StylizationModel import StylizationModel
from ..models.FeatureLossModel import FeatureLossModel
from .. import util
from ..SettingsLoader import SettingsLoader

# Load the settings and make folders for the results.
settings = SettingsLoader.load_settings_from_argv()


def with_folder(file_name: str):
    return os.path.join(settings["results_folder"], file_name)


def make_folder(folder_name: str):
    Path(folder_name).mkdir(parents=True, exist_ok=True)


make_folder(with_folder("intermediate_images"))
make_folder(with_folder("intermediate_models"))

# Use CUDA if it's available.
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
style_model = StylizationModel().to(device)
feature_model = FeatureLossModel([(1, 2), (2, 2), (3, 3), (4, 3)]).to(device)

# Set up the data loader.
trainset = torchvision.datasets.ImageFolder(
    "content", get_data_loader_transform(settings["image_size"])
)
batch_size = settings["batch_size"]
train_loader = DataLoader(trainset, batch_size=batch_size)
train_loader_len = len(train_loader)

# Load the style image.
style = load_image(settings["style_image"]).repeat(batch_size, 1, 1, 1).to(device)
vgg_style = feature_model(style)
gram_style = {name: util.gram_matrix(tensor) for name, tensor in vgg_style.items()}

# This is equivalent to the squared normalized euclidean distance.
norm = torch.nn.MSELoss()
optimizer = Adam(style_model.parameters(), settings["learning_rate"])
style_model.train()
for e in range(settings["num_epochs"]):
    epoch_start_time = time.time()
    for batch_index, (images, _) in enumerate(train_loader):
        # The image x is normalized to [0, 1].
        batch_start_time = time.time()
        x = images.to(device)
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
        feature_loss = settings["feature_weight"] * norm(
            vgg_y_hat["relu2_2"], vgg_x["relu2_2"]
        )

        # Compute the style reconstruction loss.
        style_loss = 0.0
        gram_y_hat = {
            name: util.gram_matrix(tensor) for name, tensor in vgg_y_hat.items()
        }
        for name in vgg_y_hat.keys():
            style_loss += norm(gram_y_hat[name], gram_style[name])
        style_loss *= settings["style_weight"]

        total_loss = feature_loss + style_loss
        total_loss.backward()
        optimizer.step()

        # Left-pad the epoch and batch indices for printing.
        e_str = str(e).zfill(len(str(settings["num_epochs"])))
        b_str = str(batch_index).zfill(len(str(len(train_loader))))

        # Print a status message.
        if batch_index % settings["print_status_every"] == 0:
            print("===============================")
            print(
                f"Batch {b_str} of {train_loader_len} took {time.time() - batch_start_time} seconds."
            )
            print(f"Feature loss: {feature_loss.data.item()}")
            print(f"Style loss: {style_loss.data.item()}")

        # Save intermediate images.
        if batch_index % settings["save_image_every"] == 0:
            input = images[0]
            output = y_hat[0].cpu().detach().numpy()
            save_image(
                with_folder(f"intermediate_images/image_{e_str}_{b_str}_input.png"),
                input,
            )
            save_image(
                with_folder(f"intermediate_images/image_{e_str}_{b_str}_output.png"),
                output,
            )

        # Save the model.
        if batch_index % settings["save_model_every"] == 0 and batch_index > 0:
            torch.save(
                style_model.state_dict(),
                with_folder(f"intermediate_models/model_{e_str}_{b_str}.pth"),
            )

# Save the trained style_model.
style_model.eval()
torch.save(style_model.state_dict(), with_folder("final_model.pth"))
