import os

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SkinDiseaseDataset(Dataset):
    """
    A custom Dataset class for skin images.
    Assumes images are stored in two folders: '/dataset/healthy' for healthy skin images
    and '/dataset/diseased' for diseased skin images.
    """
    def __init__(self, root_dir: str, transform: transforms.Compose | None = None):
        """
        Args:
            root_dir (str): _description_
            transform (_type_, optional): _description_. Defaults to None.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.labels = []
        self.image_paths = []

        for label, condition in enumerate(['healthy', 'diseased']):
            condition_path = os.path.join(self.root_dir, condition)

            for filename in os.listdir(condition_path):
                self.image_paths.append(os.path.join(condition_path, filename))
                self.labels.append(label)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Generates one sample of data.
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class BinaryCNN(nn.Module):
    """
    A simple convolutional neural network with layers suited for binary classification.
    """
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)

        return x
