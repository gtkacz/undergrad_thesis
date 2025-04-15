import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SkinDiseaseDataset(Dataset):
	"""
	A custom Dataset class for skin images.
	Assumes images are stored in two folders: '/dataset/healthy' for healthy skin images
	and '/dataset/diseased' for diseased skin images.
	"""

	def __init__(
		self,
		root_dir: str,
		transform: transforms.Compose | None = None,
		max_samples: int = 10_000,
	):
		"""
		Args:
		    root_dir (str): The root directory of the dataset.
		    transform (transforms.Compose, optional): Optional transform to be applied on a sample.
		    max_samples (int): The maximum number of samples to load from the dataset.
		"""
		self.root_dir = root_dir
		self.transform = transform
		self.labels = []
		self.image_paths = []

		for label, condition in enumerate(["healthy", "diseased"]):
			condition_path = os.path.join(self.root_dir, condition)

			for filename in os.listdir(condition_path):
				if filename.split(".")[-1] not in ["jpg", "jpeg", "png"]:
					continue

				self.image_paths.append(os.path.join(condition_path, filename))
				self.labels.append(label)

				if len(self.image_paths) % max_samples == 0 and self.image_paths:
					break

	def __len__(self) -> int:
		"""
		Returns the total number of samples in the dataset.
		"""
		return len(self.image_paths)

	def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
		"""
		Generates one sample of data.
		"""
		img_path = self.image_paths[idx]

		label = self.labels[idx]

		image = Image.open(img_path).convert("RGB")

		if self.transform:
			image = self.transform(image)

		return image, label
