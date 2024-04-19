import random
from typing import Sequence

import cv2
import numpy as np
import skimage

from .enums import *


def change_color_space(image: np.ndarray, domain: ColorDomain) -> np.ndarray:
    """
    Change the color space of an image.

    Args:
        image (np.ndarray): The image to change the color space of as a numpy array.
        domain (ColorDomain): The color domain to change the image to.

    Raises:
        ValueError: If the image is not in RGB format or if an unknown color domain is provided.

    Returns:
        np.ndarray: The image in the new color space as a numpy array.
    """
    assert image.ndim == 3, 'Image must be in RGB format.'

    match domain:
        case ColorDomain.GRAYSCALE:
            return skimage.color.rgb2gray(image)

        case ColorDomain.HSV:
            return skimage.color.rgb2hsv(image)[:, :, 2]

        case ColorDomain.LAB:
            return skimage.color.rgb2lab(image)[:, :, 0]

        case ColorDomain.YUV:
            return skimage.color.rgb2yuv(image)[:, :, 0]

        case ColorDomain.YCBCR:
            return cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)[:, :, 0]

        case _:
            raise ValueError(f'Unknown color domain: {domain}.')


def resize(image: np.ndarray, size: int = 512) -> np.ndarray:
    """
    Change the size of an image to a specified square size.

    Args:
        image (np.ndarray): The image to resize as a numpy array.
        size (int): The size to resize the image to. Defaults to 512.

    Returns:
        np.ndarray: The resized image as a numpy array.
    """
    return skimage.transform.resize(image, (size, size))


def denoise(image: np.ndarray, denoize_methods: Sequence[DenoisingMethod] = [DenoisingMethod.MEAN]) -> np.ndarray:
    """
    Denoise an image using a specified method.

    Args:
        image (np.ndarray): The image to denoise as a numpy array.
        denoize_methods (Sequence[DenoisingMethod]): The denoize method(s) to use for denoizing the image. Defaults to [DenoisingMethod.MEAN].

    Raises:
        ValueError: If an unknown denoize method is provided.

    Returns:
        np.ndarray: The denoized image as a numpy array.
    """
    processed_image = image.copy()

    for method in denoize_methods:
        match method:
            case 'chambolle':
                processed_image = skimage.restoration.denoise_tv_chambolle(
                    image, weight=0.1)

            case 'bilateral':
                processed_image = skimage.restoration.denoise_bilateral(
                    image, sigma_color=0.05, sigma_spatial=15)

            case 'wavelet':
                processed_image = skimage.restoration.denoise_wavelet(image)

            case 'nl_means':
                processed_image = skimage.restoration.denoise_nl_means(
                    image, h=0.05)

            case 'median':
                processed_image = skimage.filters.median(image)

            case 'mean':
                processed_image = skimage.filters.rank.mean(
                    image, skimage.morphology.disk(1))

            case 'gaussian':
                processed_image = skimage.filters.gaussian(image)

            case _:
                raise ValueError(f'Unknown denoize method: {method}')

    return processed_image


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize an image to the range [0, 1].

    Args:
        image (np.ndarray): The image to normalize as a numpy array.

    Returns:
        np.ndarray: The normalized image as a numpy array.
    """
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def equalize(image: np.ndarray, nbins: int = 256) -> np.ndarray:
    """
    Equalize an image using histogram equalization.

    Args:
        image (np.ndarray): The image to equalize as a numpy array.
        nbins (int): The number of bins to use for the histogram. Defaults to 256.

    Returns:
        np.ndarray: The equalized image as a numpy array.
    """
    return skimage.exposure.equalize_hist(image, nbins=nbins)


def segment(image: np.ndarray, segmentation_methods: Sequence[SegmentationMethod] = [SegmentationMethod.THRESHOLD]) -> np.ndarray:
    """
    Segment an image using a specified method.

    Args:
        image (np.ndarray): The image to segment as a numpy array.
        segmentation_methods (Sequence[SegmentationMethod]): The segmentation method(s) to use for segmenting the image. Defaults to [SegmentationMethod.THRESHOLD].

    Raises:
        ValueError: If an unknown segmentation method is provided.

    Returns:
        np.ndarray: The segmented image as a numpy array.
    """
    processed_image = image.copy()

    for method in segmentation_methods:
        match method:
            case 'threshold':
                processed_image = processed_image > skimage.filters.threshold_otsu(
                    processed_image)

            case 'watershed':
                processed_image = skimage.segmentation.watershed(
                    processed_image)

            case 'felzenszwalb':
                processed_image = skimage.segmentation.felzenszwalb(
                    processed_image)

            case 'quickshift':
                processed_image = skimage.segmentation.quickshift(
                    processed_image)

            case 'slic':
                processed_image = skimage.segmentation.slic(processed_image)

            case _:
                raise ValueError(
                    f'Unknown segmentation method: {method}')

    return processed_image


def augment(image: np.ndarray, augmentations: Sequence[Augmentation] = [Augmentation.FLIP]) -> np.ndarray:
    """
    Augment an image.

    Args:
        image (np.ndarray): The image to augment as a numpy array.
        augmentations (Sequence[Augmentation]): The augmentation(s) to apply to the image. Defaults to [Augmentation.FLIP].

    Raises:
        ValueError: If an unknown augmentation is provided.

    Returns:
        np.ndarray: The augmented image as a numpy array.
    """
    processed_image = image.copy()

    for augmentation in augmentations:
        match augmentation:
            case 'flip':
                processed_image = np.fliplr(processed_image)

            case 'rotate':
                processed_image = skimage.transform.rotate(
                    processed_image, random.randint(0, 360))

            case 'translate':
                processed_image = skimage.transform.warp(processed_image, skimage.transform.AffineTransform(
                    translation=(random.randint(-10, 10), random.randint(-10, 10))))

            case 'scale':
                processed_image = skimage.transform.rescale(
                    processed_image, random.uniform(0.9, 1.1))

            case 'shear':
                processed_image = skimage.transform.warp(
                    processed_image, skimage.transform.AffineTransform(shear=random.uniform(-0.1, 0.1)))

            case 'perspective':
                processed_image = skimage.transform.warp(
                    processed_image, skimage.transform.ProjectiveTransform())

            case _:
                raise ValueError(f'Unknown augmentation: {augmentation}')

    return processed_image
