import torch
import torch.nn as nn
from torchvision.models import vgg16


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

        self.vgg16 = vgg16(pretrained=True)

        # Modify the classifier part of VGG16
        # VGG16 has a classifier with 4096 units in the first layer, 4096 in the second, and 1000 output units (for ImageNet classes)
        # We will replace the last fully connected layer to output 1 unit (for binary classification)
        self.vgg16.classifier[6] = nn.Linear(4096, 1)

    def forward(self, x):
        x = self.vgg16(x)
        x = torch.sigmoid(x)  # Apply sigmoid activation for binary classification
        return x
