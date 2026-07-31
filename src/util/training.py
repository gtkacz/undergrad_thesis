"""Training loop primitives: single-epoch train and validation passes."""

import logging

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .types import LossFunction

logger = logging.getLogger(__name__)


def train_epoch(
	model: nn.Module,
	train_loader: DataLoader,
	criterion: LossFunction,
	optimizer: optim.Optimizer,
	device: torch.device,
	use_amp: bool,
	scaler: torch.amp.GradScaler | None,
	gpu_transforms: list[nn.Module] | None = None,
) -> tuple[float, float]:
	"""Run one training epoch.

	Args:
		model: The neural network model.
		train_loader: DataLoader for training data.
		criterion: Loss function.
		optimizer: Optimizer instance.
		device: Target device (CPU or CUDA).
		use_amp: Whether to use automatic mixed precision.
		scaler: GradScaler for AMP (None if CPU).
		gpu_transforms: Optional GPU-side preprocessing transforms.

	Returns:
		Tuple of (average_loss, accuracy) for this epoch.
	"""
	model.train()
	running_loss = 0.0
	running_corrects = 0
	total_samples = 0

	for images, labels in train_loader:
		images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
		labels = labels.to(device, non_blocking=True).float()

		if gpu_transforms:
			for t in gpu_transforms:
				images = t(images)

		optimizer.zero_grad(set_to_none=True)

		with torch.amp.autocast("cuda", enabled=use_amp):
			outputs = model(images).squeeze(-1)
			loss = criterion(outputs, labels)

		if scaler is not None:
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			loss.backward()
			optimizer.step()

		running_loss += loss.item() * labels.size(0)
		preds = (outputs > 0.0).float()
		running_corrects += torch.sum(preds == labels).item()
		total_samples += labels.size(0)

	avg_loss = running_loss / total_samples
	accuracy = running_corrects / total_samples
	return avg_loss, accuracy


def validate_epoch(
	model: nn.Module,
	validation_loader: DataLoader,
	criterion: LossFunction,
	device: torch.device,
	use_amp: bool,
	gpu_transforms: list[nn.Module] | None = None,
) -> tuple[float, float]:
	"""Run one validation epoch.

	Args:
		model: The neural network model.
		validation_loader: DataLoader for validation data.
		criterion: Loss function.
		device: Target device (CPU or CUDA).
		use_amp: Whether to use automatic mixed precision.
		gpu_transforms: Optional GPU-side preprocessing transforms.

	Returns:
		Tuple of (average_loss, accuracy) for this epoch.
	"""
	model.eval()
	val_loss = 0.0
	running_corrects = 0
	total_samples = 0

	with torch.no_grad():
		for images, labels in validation_loader:
			images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
			labels = labels.to(device, non_blocking=True).float()

			if gpu_transforms:
				for t in gpu_transforms:
					images = t(images)

			with torch.amp.autocast("cuda", enabled=use_amp):
				outputs = model(images).squeeze(-1)
				loss = criterion(outputs, labels)

			val_loss += loss.item() * labels.size(0)
			preds = (outputs > 0.0).float()
			running_corrects += torch.sum(preds == labels).item()
			total_samples += labels.size(0)

	avg_loss = val_loss / total_samples
	accuracy = running_corrects / total_samples
	return avg_loss, accuracy
