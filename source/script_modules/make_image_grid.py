import os
import cv2
import numpy as np
import re
from ..SettingsLoader import SettingsLoader

# This script makes an image grid of training images for the presentation.
# Run this after having run the training script to produce an image grid with
# the first n saved training outputs.

# Load the settings and make folders for the results.
settings = SettingsLoader.load_settings_from_argv()

TARGET_WIDTH = 1920 + 120 * 4
TARGET_HEIGHT = 1080
TILE_SIZE = 120

assert TARGET_WIDTH % TILE_SIZE == 0
assert TARGET_HEIGHT % TILE_SIZE == 0

num_rows = TARGET_HEIGHT // TILE_SIZE
num_cols = TARGET_WIDTH // TILE_SIZE
num_images = num_rows * num_cols

# Get the first num_images PNG files.
folder = os.path.join(settings["results_folder"], "intermediate_images")
files = os.listdir(folder)
files = sorted(files)[:num_images]

assert len(files) == num_images and "If this fails you don't have enough images."

images = [cv2.imread(os.path.join(folder, f)) for f in files]
images = [cv2.resize(image, (TILE_SIZE, TILE_SIZE)) for image in images]

grid = np.empty((TARGET_HEIGHT, TARGET_WIDTH, 3))

for row in range(num_rows):
    for col in range(num_cols):
        row_start = row * TILE_SIZE
        col_start = col * TILE_SIZE
        grid[
            row_start : row_start + TILE_SIZE, col_start : col_start + TILE_SIZE, :
        ] = images[row * num_cols + col]

cv2.imwrite(os.path.join(settings["results_folder"], "image_grid.png"), grid)
