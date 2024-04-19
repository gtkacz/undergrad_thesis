import os

import numpy as np
from skimage import io


def read_images(directory: str) -> list[np.ndarray]:
    """
    Read all images in a directory.

    Args:
        directory (str): The directory to read the images from.

    Returns:
        list[np.ndarray]: A list of images as numpy arrays.
    """
    images = []
    for filename in os.listdir(directory):

        filepath = os.path.join(directory, filename)

        if os.path.isfile(filepath):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image = io.imread(filepath)
                images.append(image)

    return images
