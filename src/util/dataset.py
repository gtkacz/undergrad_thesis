import os
from typing import Callable, Optional, Sequence

from skimage import io
from torch.utils.data import Dataset


class PsoriasisDataset(Dataset):
    def __init__(self, dir_path: str, transform: Optional[Sequence[Callable]] = None):
        self.dir_path = dir_path
        self.transform = transform
        self.images = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_name = os.path.join(self.dir_path, self.images[idx])
        image = io.imread(img_name)

        if self.transform:
            for transform in self.transform:
                image = transform(image)

        label = 1 if 'diseased' in img_name else 0

        return image, label
