from collections.abc import Callable, Sequence
from typing import ClassVar

import kornia
import torch
import torchvision.transforms.functional as tv_functional
from torch import Tensor, nn


def apply_gpu_transforms(batch: Tensor, transforms: Sequence[nn.Module]) -> Tensor:
	"""Apply a sequence of GPU-side transforms to a batch already on device.

	Args:
		batch: Image batch of shape (B, C, H, W) already on the target device.
		transforms: Sequence of nn.Module transforms to apply in order.

	Returns:
		Transformed batch tensor.
	"""
	for t in transforms:
		batch = t(batch)
	return batch


class NormalizeTransform(nn.Module):
	"""Apply a fixed per-channel affine transform.

	The configured values are constants. They are not estimated from each image
	or from the training set.
	"""

	def __init__(
		self,
		mean: list[float] | None = None,
		std: list[float] | None = None,
	) -> None:
		"""Args:
		mean: Per-channel means; defaults to [0.5].
		std: Per-channel standard deviations; defaults to [0.5].
		"""
		super().__init__()
		mean = mean or [0.5]
		std = std or [0.5]
		self.register_buffer("mean", torch.tensor(mean))
		self.register_buffer("std", torch.tensor(std))

	def forward(self, tensor: Tensor) -> Tensor:
		"""Args:
		tensor: Image tensor of shape (C, H, W) or (B, C, H, W).

		Returns:
			Normalized tensor.
		"""
		return tv_functional.normalize(tensor, self.mean.tolist(), self.std.tolist())


class DenoiseTransform(nn.Module):
	"""Apply Kornia bilateral filtering.

	Input tensors are not rescaled before filtering.
	"""

	def __init__(
		self,
		kernel_size: int = 7,
		sigma_space: float = 21,
		sigma_color: float = 10,
	) -> None:
		"""Initialize the bilateral-filter parameters."""
		super().__init__()
		kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
		self.kernel_size = (kernel_size, kernel_size)
		self.sigma_color = float(sigma_color)
		self.sigma_space = (float(sigma_space), float(sigma_space))

	def forward(self, img: Tensor) -> Tensor:
		"""Filter one image or a batch of images.

		Returns:
			The bilateral-filtered tensor.
		"""
		if not isinstance(img, torch.Tensor):
			img = tv_functional.to_tensor(img)

		needs_batch = img.dim() == 3
		if needs_batch:
			img = img.unsqueeze(0)

		# bilateral_blur unfolds input to B x C x H x W x K^2, so large batches
		# with large kernels can exhaust GPU memory.
		max_chunk = 32
		if img.size(0) > max_chunk:
			denoised = torch.cat([
				kornia.filters.bilateral_blur(chunk, self.kernel_size, self.sigma_color, self.sigma_space)
				for chunk in img.split(max_chunk)
			])
		else:
			denoised = kornia.filters.bilateral_blur(img, self.kernel_size, self.sigma_color, self.sigma_space)

		if needs_batch:
			denoised = denoised.squeeze(0)

		return denoised


class ColorSpaceTransform(nn.Module):
	"""Change the tensor's color representation with Kornia.

	Supported color spaces are RGB, BGR, HSV, LAB, and YUV. Kornia represents
	HSV hue in radians, so RGB-to-HSV output is not confined to the unit cube.
	The transform clamps its input to [0, 1] before conversion, matching the
	released experiment.
	"""

	_CONVERSIONS: ClassVar[dict[tuple[str, str], Callable[[Tensor], Tensor]]] = {
		("RGB", "HSV"): kornia.color.rgb_to_hsv,
		("RGB", "LAB"): kornia.color.rgb_to_lab,
		("HSV", "RGB"): kornia.color.hsv_to_rgb,
		("LAB", "RGB"): kornia.color.lab_to_rgb,
		("RGB", "YUV"): kornia.color.rgb_to_yuv,
		("YUV", "RGB"): kornia.color.yuv_to_rgb,
	}

	def __init__(self, source_space: str = "RGB", target_space: str = "HSV") -> None:
		"""Select the declared source and target color spaces.

		Raises:
			ValueError: If the conversion pair is unsupported.
		"""
		super().__init__()
		self.source_space = source_space
		self.target_space = target_space

		if (source_space == "RGB" and target_space == "BGR") or (source_space == "BGR" and target_space == "RGB"):
			self._convert = lambda x: x.flip(-3)
		else:
			key = (source_space, target_space)
			if key not in self._CONVERSIONS:
				raise ValueError(f"Unsupported color space conversion: {source_space} to {target_space}")
			self._convert = self._CONVERSIONS[key]

	def forward(self, img: Tensor) -> Tensor:
		"""Convert one image or a batch after clamping its values to [0, 1].

		Returns:
			The converted tensor.
		"""
		if not isinstance(img, torch.Tensor):
			img = tv_functional.to_tensor(img)

		needs_batch = img.dim() == 3
		if needs_batch:
			img = img.unsqueeze(0)

		# Kornia color conversions expect [0, 1];
		# upstream transforms (e.g. normalization) may shift values outside it.
		img = img.clamp(0.0, 1.0)

		converted = self._convert(img)

		if needs_batch:
			converted = converted.squeeze(0)

		return converted


class EqualizationTransform(nn.Module):
	"""Equalize the Y channel after treating a three-channel input as RGB.

	The input is clamped to [0, 1]. The pipeline runner does not track color
	space, so an equalization step placed after RGB-to-HSV conversion still
	interprets the three channels as RGB. This behavior is retained to reproduce
	the released experiment and is a stated limitation of the paper.
	"""

	def forward(self, img: Tensor) -> Tensor:
		"""Equalize one image or a batch of images.

		Returns:
			The equalized tensor.
		"""
		if not isinstance(img, torch.Tensor):
			img = tv_functional.to_tensor(img)

		needs_batch = img.dim() == 3
		if needs_batch:
			img = img.unsqueeze(0)

		# Normalization may shift values outside the range required by
		# equalization and YUV conversion.
		img = img.clamp(0.0, 1.0)

		if img.shape[-3] == 3:
			yuv = kornia.color.rgb_to_yuv(img)
			y_eq = kornia.enhance.equalize(yuv[:, 0:1, :, :])
			yuv = torch.cat([y_eq, yuv[:, 1:, :, :]], dim=1)
			equalized = kornia.color.yuv_to_rgb(yuv)
		else:
			equalized = kornia.enhance.equalize(img)

		if needs_batch:
			equalized = equalized.squeeze(0)

		return equalized
