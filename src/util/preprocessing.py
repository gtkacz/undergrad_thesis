import cv2
import numpy as np
import torchvision.transforms.functional as TF
from torch import nn


class NormalizeTransform:
	def __init__(self, mean=list((0.5,)), std=list((0.5,))):
		"""
		Args:
		    mean (list or tuple): Sequence of means for each channel.
		    std (list or tuple): Sequence of standard deviations for each channel.
		"""
		self.mean = mean
		self.std = std

	def __call__(self, tensor):
		"""
		Args:
		    tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

		Returns:
		    Tensor: Normalized image.
		"""
		return TF.normalize(tensor, self.mean, self.std)


class DenoiseTransform(nn.Module):
	"""
	Apply denoising to the input image using Non-Local Means Denoising algorithm.
	"""

	def __init__(self, h=10, template_window_size=7, search_window_size=21):
		super().__init__()
		self.h = h
		self.template_window_size = template_window_size
		self.search_window_size = search_window_size

	def forward(self, img):
		if not isinstance(img, np.ndarray):
			img = TF.to_pil_image(img)
			img = np.array(img)

		if len(img.shape) == 3 and img.shape[2] == 3:
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			denoised = cv2.fastNlMeansDenoisingColored(
				img,
				None,
				self.h,
				self.h,
				self.template_window_size,
				self.search_window_size,
			)
			denoised = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
		else:
			denoised = cv2.fastNlMeansDenoising(img, None, self.h, self.template_window_size, self.search_window_size)

		return TF.to_tensor(denoised)


class ColorSpaceTransform(nn.Module):
	"""
	Change the color space of the input image.
	Supported color spaces: 'RGB', 'BGR', 'HSV', 'LAB', 'YUV'
	"""

	def __init__(self, source_space="RGB", target_space="HSV"):
		super().__init__()
		self.source_space = source_space
		self.target_space = target_space

	def forward(self, img):
		if not isinstance(img, np.ndarray):
			img = TF.to_pil_image(img)
			img = np.array(img)

		if self.source_space == "RGB" and self.target_space == "HSV":
			converted = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		elif self.source_space == "RGB" and self.target_space == "LAB":
			converted = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
		elif self.source_space == "HSV" and self.target_space == "RGB":
			converted = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
		elif self.source_space == "LAB" and self.target_space == "RGB":
			converted = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
		elif self.source_space == "RGB" and self.target_space == "BGR":
			converted = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		elif self.source_space == "BGR" and self.target_space == "RGB":
			converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		elif self.source_space == "RGB" and self.target_space == "YUV":
			converted = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
		else:
			raise ValueError(f"Unsupported color space conversion: {self.source_space} to {self.target_space}")

		return TF.to_tensor(converted)


class EqualizationTransform(nn.Module):
	"""
	Apply histogram equalization to the input image.
	"""

	def forward(self, img):
		if not isinstance(img, np.ndarray):
			img = TF.to_pil_image(img)
			img = np.array(img)

		if len(img.shape) == 3 and img.shape[2] == 3:
			img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
			img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
			equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
		else:
			equalized = cv2.equalizeHist(img)

		return TF.to_tensor(equalized)
