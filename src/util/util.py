import copy
import os
from functools import wraps

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skimage import io
from torch import Tensor
from torch import device as torchdevice
from torch.utils.data import DataLoader, Dataset, Subset

from .types import LossFunction, TestingDataset, TrainingDataset


def __use_parameters_by_value(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        args_copy = [copy.deepcopy(arg) for arg in args]
        kwargs_copy = {key: copy.deepcopy(value) for key, value in kwargs.items()}

        return func(*args_copy, **kwargs_copy)

    return wrapper


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

        if os.path.isfile(filepath) and filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp")
        ):
            image = io.imread(filepath)
            images.append(image)

    return images


def flatten_prediction(
    prediction: list[dict[str, str | float]],
) -> dict[str, str | float]:
    """
    Flatten a list of predictions into a dictionary.

    Args:
        prediction (list[dict[str, str | float]]): The list of predictions to flatten.

    Returns:
        dict[str, float]: The flattened predictions.
    """
    return {pred["label"]: pred["score"] for pred in prediction} # type: ignore


@__use_parameters_by_value
def train_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: LossFunction,
    optimizer: optim.Optimizer,
    num_epochs: int = 10,
    device: torchdevice = torchdevice("cpu"),
    verbose: bool = True,
) -> None:
    """
    Train a PyTorch model.

    Args:
        model (nn.Module): The model to train.
        data_loader (DataLoader): The DataLoader for the training data.
        criterion (LossFunction): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        num_epochs (int, optional): The number of epochs to train for. Defaults to 10.
        torchdevice (torch.device, optional): The device to train on. Defaults to torch.device('cpu').
        verbose (bool, optional): Whether to print the loss. Defaults to True.
    """
    model.to(device).train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            labels = labels.float()
            optimizer.zero_grad()
            outputs = model(images)
            # loss = criterion(outputs.squeeze(), labels)
            # loss = criterion(outputs, labels.unsqueeze(1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader):.4f}")


@__use_parameters_by_value
def split_datasets(
    dataset: Dataset[Tensor],
    training_ratio: float,
    testing_ratio: float,
    seed: int = 42,
) -> tuple[TrainingDataset, TestingDataset]:
    """
    Split a dataset into training, validation, and testing sets.

    Args:
        dataset_class (type[Dataset]): The class of the dataset to split.
        training_ratio (float): The ratio of the dataset to use for training.
        testing_ratio (float): The ratio of the dataset to use for testing.
        seed (int, optional): The random seed. Defaults to 42.

    Returns:
        tuple[Dataset, Dataset, Dataset]: The training, validation, and testing datasets.
    """
    dataset_size = len(dataset) # type: ignore

    indices = list(range(dataset_size))

    np.random.seed(seed)
    np.random.shuffle(indices)

    training_split = int(np.floor(training_ratio * dataset_size))
    testing_split = int(np.floor(testing_ratio * dataset_size))

    training_indices = indices[:training_split]
    testing_indices = indices[training_split : training_split + testing_split]

    training_set = Subset(dataset, training_indices)
    testing_set = Subset(dataset, testing_indices)

    return training_set, testing_set


@__use_parameters_by_value
def test_model(model: nn.Module, test_loader: DataLoader, device: torchdevice) -> float:
    """
    Test a PyTorch model.

    Args:
        model (nn.Module): The model to test.
        test_loader (DataLoader): The DataLoader for the testing data.
        device (torchdevice): The device to test on.

    Returns:
        float: The accuracy of the model.
    """
    model.to(device).eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()

    return correct / total
