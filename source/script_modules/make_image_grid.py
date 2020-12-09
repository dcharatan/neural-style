import os
import cv2
import numpy as np
import re

# This script makes an image grid of training images for the presentation.

IMAGE_FOLDER = "."

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
TILE_SIZE = 120

assert TARGET_WIDTH % TILE_SIZE == 0
assert TARGET_HEIGHT % TILE_SIZE == 0

num_rows = TARGET_HEIGHT // TILE_SIZE
num_cols = TARGET_WIDTH // TILE_SIZE
num_images = num_rows * num_cols

# Find all PNG files.
files = []
for file in os.listdir(IMAGE_FOLDER):
    if file.endswith(".png"):
        files.append(file)

# Extract the valid files and sort them by index.
valid_files = [f for f in files if re.match("tmp_[0-9]+_[0-9]+\.png", f)]


def get_numbers(x):
    first = x.index("_")
    x = x[first + 1 :]
    second = x.index("_")
    num1 = int(x[:second])
    x = x[second + 1 :]
    third = x.index(".")
    num2 = int(x[:third])
    return num1, num2


valid_files = sorted(valid_files, key=get_numbers)
valid_files = valid_files[:num_images]

assert len(valid_files) == num_images and "If this fails you don't have enough images."

images = [cv2.imread(os.path.join(IMAGE_FOLDER, f)) for f in valid_files]
images = [cv2.resize(image, (TILE_SIZE, TILE_SIZE)) for image in images]

grid = np.empty((TARGET_HEIGHT, TARGET_WIDTH, 3))

for row in range(num_rows):
    for col in range(num_cols):
        row_start = row * TILE_SIZE
        col_start = col * TILE_SIZE
        grid[
            row_start : row_start + TILE_SIZE, col_start : col_start + TILE_SIZE, :
        ] = images[row * num_cols + col]

cv2.imwrite("tmp_image_grid.png", grid)
