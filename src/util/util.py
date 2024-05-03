import os

import numpy as np
import torch.nn as nn
import torch.optim as optim
from skimage import io
from torch import device as torchdevice
from torch.utils.data import DataLoader, Dataset, Subset

from .types import *


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


def train_model(model: nn.Module, train_loader: DataLoader, criterion: LossFunction, optimizer: optim.Optimizer, num_epochs: int = 10, device: torchdevice = torchdevice('cpu')) -> None:
    """
    Train a PyTorch model.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): The DataLoader for the training data.
        criterion (LossFunction): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        num_epochs (int, optional): The number of epochs to train for. Defaults to 10.
        torchdevice (torch.device, optional): The device to train on. Defaults to torch.device('cpu').
    """
    for epoch in range(num_epochs):
        model.to(device).train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            labels = labels.float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')


def get_datasets(dataset_class: type[Dataset[Tensor]], training_ratio: float, validation_ratio: float, testing_ratio: float, seed: int = 42, **kwargs) -> tuple[TrainingDataset, ValidationDataset, TestDataset]:
    """
    Split a dataset into training, validation, and testing sets.

    Args:
        dataset_class (type[Dataset]): The class of the dataset to split.
        training_ratio (float): The ratio of the dataset to use for training.
        validation_ratio (float): The ratio of the dataset to use for validation.
        testing_ratio (float): The ratio of the dataset to use for testing.
        seed (int, optional): The random seed. Defaults to 42.

    Returns:
        tuple[Dataset, Dataset, Dataset]: The training, validation, and testing datasets.
    """
    dataset = dataset_class(**kwargs)
    dataset_size = len(dataset)

    indices = list(range(dataset_size))

    np.random.seed(seed)
    np.random.shuffle(indices)

    training_split = int(np.floor(training_ratio * dataset_size))
    validation_split = int(np.floor(validation_ratio * dataset_size))
    testing_split = int(np.floor(testing_ratio * dataset_size))

    training_indices = indices[:training_split]
    validation_indices = indices[training_split:training_split+validation_split]
    testing_indices = indices[training_split +
                              validation_split:training_split+validation_split+testing_split]

    training_set = Subset(dataset, training_indices)
    validation_set = Subset(dataset, validation_indices)
    testing_set = Subset(dataset, testing_indices)

    return training_set, validation_set, testing_set
