import os

import numpy as np
import torch.nn as nn
import torch.optim as optim
from skimage import io
from torch.utils.data import DataLoader


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


def flatten_prediction(prediction: list[dict[str, str | float]]) -> dict[str, str | float]:
    """
    Flatten a list of predictions into a dictionary.

    Args:
        prediction (list[dict[str, str | float]]): The list of predictions to flatten.

    Returns:
        dict[str, float]: The flattened predictions.
    """
    return {pred['label']: pred['score'] for pred in prediction}


def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.modules.loss._Loss, optimizer: optim.Optimizer, num_epochs: int = 10) -> None:
    """
    Train a PyTorch model.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): The DataLoader for the training data.
        criterion (nn.modules.loss._Loss): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        num_epochs (int, optional): The number of epochs to train for. Defaults to 10.
    """
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            labels = labels.float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')
