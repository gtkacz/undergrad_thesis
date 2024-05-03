import os

from PIL import Image
from torch import Tensor
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
            root_dir (str): The root directory of the dataset.
            transform (transforms.Compose, optional): Optional transform to be applied on a sample.
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

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        """
        Generates one sample of data.
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
