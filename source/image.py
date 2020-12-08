import torchvision.transforms as tf
import torch
import numpy as np
from PIL import Image
from typing import Optional


def get_data_loader_transform(image_size: int):
    transforms = [
        tf.Resize(image_size),
        tf.CenterCrop(image_size),
        tf.ToTensor(),
    ]
    return tf.Compose(transforms)


def get_pillow_transform(image_size: Optional[int]):
    if image_size is None:
        transforms = []
    else:
        transforms = [
            tf.Resize(image_size),
            tf.CenterCrop(image_size),
        ]
    transforms.append(tf.ToTensor())
    return tf.Compose(transforms)


def load_image(file_name: str) -> torch.Tensor:
    """Load an image so that its shape is (B, C, H, W) and it's normalized to
    the range [0, 1].
    """
    transform = get_pillow_transform(None)
    image = Image.open(file_name)
    return transform(image).unsqueeze(0)


def save_image(file_name: str, image: np.ndarray):
    # Go from (C, H, W) to (H, W, C).
    image = np.transpose(image, (1, 2, 0))

    # Clip the image from 0 to 1.
    image = np.clip(image, 0, 1)

    # Go from [0, 1] to [0, 255].
    Image.fromarray(np.uint8(image * 255)).save(file_name)


def get_cv2_transform(image_size: int):
    transforms = [
        tf.Resize(image_size),
        tf.CenterCrop(image_size),
        tf.ToTensor(),
    ]
    return tf.Compose(transforms)