"""Dataset loading, splitting, and DataLoader construction."""

from collections.abc import Sequence
from math import isclose
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from .config import TrainingConfig
from .types import TestingDataset, TrainingDataset, ValidationDataset


class SkinDiseaseDataset(Dataset):
	"""Custom Dataset for binary skin disease classification.

	Loads images from two subdirectories: 'healthy' (label=0) and 'diseased' (label=1).
	"""

	def __init__(
		self,
		root_dir: str | Path,
		transform: transforms.Compose | None = None,
		max_samples: int = 10_000,
	) -> None:
		"""Initialize the dataset.

		Args:
			root_dir: Root directory containing 'healthy' and 'diseased' subdirectories.
			transform: Optional torchvision transforms to apply to each image.
			max_samples: Maximum number of samples to load per class.

		Raises:
			FileNotFoundError: If either class directory is absent.
		"""
		self.root_dir = Path(root_dir)
		self.transform = transform
		self.labels: list[int] = []
		self.image_paths: list[Path] = []

		valid_extensions = {".jpg", ".jpeg", ".png"}

		for label, condition in enumerate(["healthy", "diseased"]):
			condition_path = self.root_dir / condition
			if not condition_path.is_dir():
				raise FileNotFoundError(
					f"Expected dataset class directory at {condition_path}. "
					"See README.md for the required dataset layout.",
				)
			count = 0

			for filepath in sorted(condition_path.iterdir()):
				if filepath.suffix.lower() not in valid_extensions:
					continue

				self.image_paths.append(filepath)
				self.labels.append(label)
				count += 1

				if count >= max_samples:
					break

	def __len__(self) -> int:
		"""Return the total number of samples in the dataset.

		Returns:
			Number of samples.
		"""
		return len(self.image_paths)

	def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
		"""Load and return one sample.

		Args:
			idx: Sample index.

		Returns:
			Tuple of (image, label).
		"""
		with Image.open(self.image_paths[idx]) as source:
			image = source.convert("RGB")

		if self.transform:
			image = self.transform(image)

		return image, self.labels[idx]


def split_datasets(
	dataset: Dataset[Tensor],
	training_ratio: float,
	testing_ratio: float,
	validation_ratio: float,
	seed: int = 42,
) -> tuple[TrainingDataset, TestingDataset, ValidationDataset]:
	"""Split a dataset into training, testing, and validation subsets.

	Args:
		dataset: The full dataset to split.
		training_ratio: Fraction of data for training.
		testing_ratio: Fraction of data for testing.
		validation_ratio: Fraction of data for validation.
		seed: Random seed for reproducibility.

	Returns:
		Tuple of (training_subset, testing_subset, validation_subset).

	Raises:
		ValueError: If ratios are negative or do not sum to one.
	"""
	ratios = (training_ratio, testing_ratio, validation_ratio)
	if any(ratio < 0 for ratio in ratios) or not isclose(sum(ratios), 1.0):
		raise ValueError("training, testing, and validation ratios must be nonnegative and sum to 1")

	dataset_size = len(dataset)  # type: ignore[arg-type]
	indices = list(range(dataset_size))

	torch.manual_seed(seed)
	np.random.seed(seed)  # ruff: ignore[numpy-legacy-random]

	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	np.random.shuffle(indices)  # ruff: ignore[numpy-legacy-random]

	training_split = int(np.floor(training_ratio * dataset_size))
	testing_split = int(np.floor(testing_ratio * dataset_size))

	training_indices = indices[:training_split]
	testing_indices = indices[training_split : training_split + testing_split]
	validation_indices = indices[training_split + testing_split :]

	return (
		Subset(dataset, training_indices),
		Subset(dataset, testing_indices),
		Subset(dataset, validation_indices),
	)


def get_data_loaders(
	training_config: TrainingConfig,
	cpu_transforms: Sequence[nn.Module | object] | None = None,
	training_ratio: float = 0.8,
	testing_ratio: float = 0.1,
	validation_ratio: float = 0.1,
	seed: int = 47,
) -> tuple[DataLoader, DataLoader, DataLoader]:
	"""Create train/test/validation DataLoaders for the skin disease dataset.

	The DataLoader applies only CPU-side transforms (Resize + ToTensor + optional extras).
	GPU-side transforms (kornia) are applied separately after .to(device).

	Args:
		training_config: Training hyperparameters (batch_size, num_workers, etc.).
		cpu_transforms: Additional CPU-side transforms applied after Resize + ToTensor.
		training_ratio: Fraction of data for training.
		testing_ratio: Fraction of data for testing.
		validation_ratio: Fraction of data for validation.
		seed: Random seed for dataset splitting.

	Returns:
		Tuple of (train_loader, test_loader, validation_loader).
	"""
	base_transforms: list[object] = [
		transforms.Resize((training_config.resize_dim, training_config.resize_dim)),
		transforms.ToTensor(),
	]
	if cpu_transforms:
		base_transforms.extend(cpu_transforms)

	transform = transforms.Compose(base_transforms)

	common_loader_kwargs: dict = {
		"batch_size": training_config.batch_size,
		"num_workers": training_config.num_workers,
		"pin_memory": training_config.pin_memory,
	}

	if training_config.num_workers > 0:
		common_loader_kwargs["persistent_workers"] = True
		common_loader_kwargs["prefetch_factor"] = 4

	base_dataset = SkinDiseaseDataset(root_dir="dataset", transform=transform)
	train_dataset, test_dataset, validation_dataset = split_datasets(
		base_dataset,
		training_ratio,
		testing_ratio,
		validation_ratio,
		seed,
	)

	return (
		DataLoader(train_dataset, shuffle=training_config.shuffle, **common_loader_kwargs),
		DataLoader(test_dataset, shuffle=False, **common_loader_kwargs),
		DataLoader(validation_dataset, shuffle=False, **common_loader_kwargs),
	)
