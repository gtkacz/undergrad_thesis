import cv2
import numpy as np
import PIL.Image
import skimage
import torch.nn as nn
from PIL import ImageOps
from PIL.Image import Image
from torch import Tensor
from torchvision import transforms

from .enums import *


class _Preprocess(nn.Module):
    @staticmethod
    def _parse_forward_input(image: np.ndarray | Tensor | Image) -> tuple[np.ndarray, type[Tensor | Image]]:
        return_type = Tensor if isinstance(image, Tensor) else Image
        parsed_image: np.ndarray = np.ndarray((0, 0, 0))

        if isinstance(image, Tensor):
            parsed_image = np.array(transforms.ToPILImage()(image))

        elif isinstance(image, Image):
            parsed_image = np.array(image)

        elif isinstance(image, np.ndarray):
            assert image.ndim == 3, 'Image must be in RGB format.'
            parsed_image = image

        else:
            raise ValueError(f'Unknown image type: {type(image)}.')

        return parsed_image, return_type


class ChangeColorSpaceTransform(_Preprocess):
    def __init__(self, domain: ColorDomain):
        super().__init__()
        self.domain = domain

    def forward(self, image: np.ndarray | Tensor | Image) -> Tensor | Image:
        """
        Change the color space of an image.

        Args:
            image (np.ndarray): The image to change the color space of as a numpy array.

        Raises:
            ValueError: If the image is not in RGB format or if an unknown color domain is provided.

        Returns:
            np.ndarray: The image in the new color space as a numpy array.
        """
        image, return_type = self._parse_forward_input(image)

        return_value = None

        match self.domain:
            case ColorDomain.GRAYSCALE:
                return_value = skimage.color.rgb2gray(image)

            case ColorDomain.HSV:
                return_value = skimage.color.rgb2hsv(image)[:, :, 2]

            case ColorDomain.LAB:
                return_value = skimage.color.rgb2lab(image)[:, :, 0]

            case ColorDomain.YUV:
                return_value = skimage.color.rgb2yuv(image)[:, :, 0]

            case ColorDomain.YCBCR:
                return_value = cv2.cvtColor(
                    image, cv2.COLOR_RGB2YCR_CB)[:, :, 0]

            case _:
                raise ValueError(f'Unknown color domain: {self.domain}.')

        return PIL.Image.fromarray(return_value) if return_type == Image else transforms.ToTensor()(PIL.Image.fromarray(return_value))


class DenoiseTransform(_Preprocess):
    def __init__(self, method: DenoisingMethod):
        super().__init__()
        self.method = method

    def forward(self, image: np.ndarray | Tensor | Image) -> Tensor | Image:
        """
        Denoise an image.

        Args:
            image (np.ndarray): The image to denoise as a numpy array.

        Returns:
            np.ndarray: The denoised image as a numpy array.
        """
        image, return_type = self._parse_forward_input(image)

        return_value = None

        match self.method:
            case DenoisingMethod.CHAMBOLLE:
                return_value = skimage.restoration.denoise_tv_chambolle(
                    image, weight=0.1)

            case DenoisingMethod.BILATERAL:
                return_value = skimage.restoration.denoise_bilateral(
                    image[:, :, :3], sigma_color=0.05, sigma_spatial=15)

            case DenoisingMethod.WAVELET:
                return_value = skimage.restoration.denoise_wavelet(image)

            case DenoisingMethod.NL_MEANS:
                return_value = skimage.restoration.denoise_nl_means(
                    image, h=0.05)

            case DenoisingMethod.MEDIAN:
                return_value = skimage.filters.median(image)

            case DenoisingMethod.MEAN:
                return_value = skimage.filters.rank.mean(
                    image, skimage.morphology.disk(1))

            case DenoisingMethod.GAUSSIAN:
                return_value = skimage.filters.gaussian(image)

            case _:
                raise ValueError(f'Unknown denoize method: {self.method}')

        return PIL.Image.fromarray(return_value) if return_type == Image else transforms.ToTensor()(PIL.Image.fromarray(return_value))


class NormalizeTransform(_Preprocess):
    def __init__(self):
        super().__init__()

    def forward(self, image: np.ndarray | Tensor | Image) -> Tensor | Image:
        """
        Normalize an image.

        Args:
            image (np.ndarray): The image to normalize as a numpy array.

        Returns:
            np.ndarray: The normalized image as a numpy array.
        """
        image, return_type = self._parse_forward_input(image)

        return_value = (image - np.min(image)) / \
            (np.max(image) - np.min(image))

        return PIL.Image.fromarray(return_value) if return_type == Image else transforms.ToTensor()(PIL.Image.fromarray(return_value))


# class EqualizeTransform(_Preprocess):
#     def __init__(self, method: EqualizationMethod, nbins: int = 256):
#         super().__init__()
#         self.method = method
#         self.nbins = nbins

#     def forward(self, image: np.ndarray | Tensor | Image) -> Tensor | Image:
#         """
#         Equalize an image using histogram equalization.

#         Args:
#             image (np.ndarray): The image to equalize as a numpy array.

#         Returns:
#             np.ndarray: The equalized image as a numpy array.
#         """
#         image, return_type = self._parse_forward_input(image)

#         return_value = None

#         match self.method:
#             case EqualizationMethod.THRESHOLD:
#                 return_value = image > skimage.filters.threshold_otsu(
#                     image)

#             case EqualizationMethod.WATERSHED:
#                 return_value = skimage.segmentation.watershed(
#                     image)

#             case EqualizationMethod.FELZENSZWALB:
#                 return_value = skimage.segmentation.felzenszwalb(
#                     image)

#             case EqualizationMethod.QUICKSHIFT:
#                 return_value = skimage.segmentation.quickshift(
#                     image)

#             case EqualizationMethod.SLIC:
#                 return_value = skimage.segmentation.slic(image)

#             case _:
#                 raise ValueError(
#                     f'Unknown equalization method: {self.method}')

#         return PIL.Image.fromarray(return_value) if return_type == Image else transforms.ToTensor()(PIL.Image.fromarray(return_value))

class EqualizeTransform():
    """
    Equalizes an image using various methods.

    Args:
        method (EqualizeMethod): The equalization method to use.
        **kwargs (dict): Additional arguments specific to the method.

    Supported methods and kwargs:
        - histogram (no additional arguments)
        - clahe (clip_limit: float): Sets the contrast limiting parameter (default: 4.0)
    """

    def __init__(self, method = 'HISTOGRAM_EQUALIZATION', **kwargs):
        self.method = method
        self.kwargs = kwargs

    def __call__(self, img):
        """
        Equalizes the given PIL image.

        Args:
            img (PIL.Image): The image to equalize.

        Returns:
            PIL.Image: The equalized image.
        """
        if self.method == 'HISTOGRAM_EQUALIZATION':
            return ImageOps.equalize(img)
        elif self.method == 'CLAHE':
            return ImageOps.equalize(img, mask=ImageOps.grayscale(img))
        else:
            raise ValueError(f"Unsupported equalization method: {self.method}")