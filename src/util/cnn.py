import torch
import torch.nn as nn

torch.backends.cudnn.benchmark = True


class BinaryCNN(nn.Module):
    """
    BinaryCNN is a simple Convolutional Neural Network (CNN) for binary classification. Based on: https://github.com/Harikrishnan6336/Mask_Classifier/blob/master/src/Mask_Classifier_CNN_.ipynb.
    """

    def __init__(self, device: torch.device | None = None):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

        # print(f"Using device: {self.device}")

        # Convolutional layer block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                3, 32, kernel_size=3, padding=1
            ),  # Convolutional layer with 32 filters
            nn.BatchNorm2d(32),  # Batch normalization
            nn.ReLU(),  # ReLU activation function
            nn.MaxPool2d(2),  # Max pooling with 2x2 window
        )

        # Convolutional layer block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                32, 64, kernel_size=3, padding=1
            ),  # Convolutional layer with 64 filters
            nn.BatchNorm2d(64),  # Batch normalization
            nn.ReLU(),  # ReLU activation function
            nn.MaxPool2d(2),  # Max pooling with 2x2 window
        )

        # Convolutional layer block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                64, 128, kernel_size=3, padding=1
            ),  # Convolutional layer with 128 filters
            nn.BatchNorm2d(128),  # Batch normalization
            nn.ReLU(),  # ReLU activation function
            nn.MaxPool2d(2),  # Max pooling with 2x2 window
        )

        # Convolutional layer block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                128, 256, kernel_size=3, padding=1
            ),  # Convolutional layer with 256 filters
            nn.BatchNorm2d(256),  # Batch normalization
            nn.ReLU(),  # ReLU activation function
            nn.MaxPool2d(2),  # Max pooling with 2x2 window
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(
                256 * 8 * 8, 512
            ),  # Adjusted input size after conv layers for 512x512 input
            nn.ReLU(),  # ReLU activation function
            nn.Linear(512, 256),  # Fully connected layer with 256 units
            nn.ReLU(),  # ReLU activation function
            nn.Linear(256, 1),  # Binary output (1 unit)
            nn.Sigmoid(),  # Sigmoid activation for binary classification
        )

    def forward(self, x):
        x = self.conv1(x)  # Apply first convolutional block
        x = self.conv2(x)  # Apply second convolutional block
        x = self.conv3(x)  # Apply third convolutional block
        x = self.conv4(x)  # Apply fourth convolutional block
        x = x.view(
            -1, 256 * 8 * 8
        )  # Flatten the tensor to prepare for fully connected layers
        x = self.fc_layers(x)  # Apply fully connected layers
        return x
