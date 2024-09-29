import copy
import os
import sys
from functools import wraps
from time import time as timer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skimage import io
from torch import Tensor
from torch import device as torchdevice
from torch.utils.data import DataLoader, Dataset, Subset

from .types import LossFunction, TestingDataset, TrainingDataset, ValidationDataset


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
    return {pred["label"]: pred["score"] for pred in prediction}  # type: ignore


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
    model = model.to(device)
    print(f"Training on {device}...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            labels = labels.float()
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if verbose:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader):.4f}"
            )


@__use_parameters_by_value
def split_datasets(
    dataset: Dataset[Tensor],
    training_ratio: float,
    testing_ratio: float,
    validation_ratio: float,
    seed: int = 42,
) -> tuple[TrainingDataset, TestingDataset, ValidationDataset]:
    """
    Split a dataset into training, validation, and testing sets.

    Args:
        dataset_class (type[Dataset]): The class of the dataset to split.
        training_ratio (float): The ratio of the dataset to use for training.
        testing_ratio (float): The ratio of the dataset to use for testing.
        validation_ratio (float): The ratio of the dataset to use for validation.
        seed (int, optional): The random seed. Defaults to 42.

    Returns:
        tuple[Dataset, Dataset, Dataset]: The training, validation, and testing datasets.
    """
    dataset_size = len(dataset)  # type: ignore

    indices = list(range(dataset_size))

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.shuffle(indices)

    training_split = int(np.floor(training_ratio * dataset_size))
    testing_split = int(np.floor(testing_ratio * dataset_size))

    training_indices = indices[:training_split]
    testing_indices = indices[training_split : training_split + testing_split]
    validation_indices = indices[training_split + testing_split :]

    training_set = Subset(dataset, training_indices)
    testing_set = Subset(dataset, testing_indices)
    validation_set = Subset(dataset, validation_indices)

    return training_set, testing_set, validation_set


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
    print(f"Testing on {device}...")
    model.to(device).eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float()

            outputs = model(images).squeeze()

            total += labels.size(0)
            correct += torch.sum((outputs > 0.5).float() == labels).item()

    return correct / total


@__use_parameters_by_value
def train_and_evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module = nn.MSELoss(),
    num_epochs: int = 10,
    device: torchdevice = torchdevice("cpu"),
    verbose: bool = True,
) -> float:
    """
    Train and evaluate a PyTorch model.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        test_loader (DataLoader): DataLoader for the testing data.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        num_epochs (int, optional): Number of epochs to train. Defaults to 10.
        device (torch.device, optional): Device to train on. Defaults to CPU.
        verbose (bool, optional): Whether to print loss and accuracy. Defaults to True.
    """
    model.to(device)
    print(f"\nTraining on {device}...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Evaluation phase
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images).squeeze()
                outputs = torch.sigmoid(outputs)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total

        if verbose:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}"
            )

    return accuracy


def evaluate(
    model: nn.Module,
    optimizer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module = nn.MSELoss(),
    device: torchdevice = torchdevice("cpu"),
    num_epochs: int = 10,
    verbose: bool = True,
) -> float:
    if not verbose:
        sys.stdout = open(os.devnull, "w")

    # Variables to track best model and plot metrics
    best_val_accuracy = 0.0  # To track the best validation accuracy
    train_losses, val_losses = [], []  # Lists to store training and validation losses
    train_accuracies, val_accuracies = (
        [],
        [],
    )  # Lists to store training and validation accuracies

    # Record the start time of training
    start_time = timer()

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        training_running_corrects = 0
        total_train_samples = 0

        for images, labels in train_loader:
            # Move data to GPU
            images, labels = (
                images.to(device),
                labels.to(device).float(),
            )  # Ensure labels are floats for MSELoss

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)  # Forward pass
            outputs = outputs.squeeze()  # Remove extra dimensions from the output
            loss = criterion(outputs, labels)  # Compute MSE loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()  # Accumulate loss
            preds = (
                outputs > 0.5
            ).float()  # Predictions: apply threshold to get binary output
            training_running_corrects += torch.sum(
                preds == labels
            ).item()  # Count correct predictions
            total_train_samples += labels.size(0)  # Count total samples

        # Compute metrics for the epoch
        train_loss = running_loss / len(train_loader)  # Average training loss
        train_accuracy = (
            training_running_corrects / total_train_samples
        )  # Training accuracy

        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_running_corrects = 0
        total_val_samples = 0

        with torch.no_grad():  # Disable gradient calculation
            for images, labels in validation_loader:
                # Move data to GPU
                images, labels = (
                    images.to(device),
                    labels.to(device).float(),
                )  # Ensure labels are floats for MSELoss

                outputs = model(images)  # Forward pass
                outputs = outputs.squeeze()  # Remove extra dimensions from the output
                loss = criterion(outputs, labels)  # Compute MSE loss
                val_loss += loss.item()  # Accumulate validation loss
                preds = (
                    outputs > 0.5
                ).float()  # Predictions: apply threshold to get binary output
                val_running_corrects += torch.sum(
                    preds == labels
                ).item()  # Count correct predictions
                total_val_samples += labels.size(0)  # Count total samples

        # Compute metrics for validation
        val_loss = val_loss / len(validation_loader)  # Average validation loss
        val_accuracy = val_running_corrects / total_val_samples  # Validation accuracy

        # Print metrics for this epoch
        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy*100:.2f}%, "
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy*100:.2f}%"
        )

        # Save losses and accuracies for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # Save the model with the best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(
                f"Best model saved at epoch {epoch+1} with Validation Accuracy: {val_accuracy*100:.2f}%"
            )

    # Calculate and print training duration
    training_duration = timer() - start_time
    training_duration_minutes = training_duration / 60  # Convert duration to minutes
    print(f"Total training duration: {training_duration_minutes:.2f} minutes")

    model.eval()  # Set the model to evaluation mode (disables dropout, batch norm updates, etc.)

    # Initialize variables to track performance metrics during testing
    test_running_corrects = 0  # Counter for correct predictions
    total_test_samples = 0  # Counter for total number of test samples

    # Lists to store predictions and true labels
    all_preds = []
    all_labels = []

    # Perform testing without gradient calculations
    with torch.no_grad():  # Disable gradient tracking to save memory and computation
        for images, labels in test_loader:
            # Move data to the appropriate device (GPU or CPU)
            images, labels = (
                images.to(device),
                labels.to(device).float(),
            )  # Ensure labels are floats for MSELoss

            outputs = model(images)  # Get model predictions
            outputs = outputs.squeeze()  # Remove extra dimensions from the output
            preds = (
                outputs > 0.5
            ).float()  # Apply threshold to get binary predictions (0 or 1)

            # Update counters with the number of correct predictions
            test_running_corrects += torch.sum(
                preds == labels
            ).item()  # Compare predicted and true labels
            total_test_samples += labels.size(0)

            # Collect predictions and true labels for metrics computation
            all_preds.extend(
                preds.cpu().numpy()
            )  # Move predictions to CPU and convert to numpy array
            all_labels.extend(
                labels.cpu().numpy()
            )  # Move true labels to CPU and convert to numpy array

    # Calculate test accuracy
    test_accuracy = test_running_corrects / total_test_samples

    # Print the accuracy of the binary classification model on the test set
    print(f"Test Accuracy of the Binary Classification Model: {test_accuracy*100:.2f}%")

    return test_accuracy
