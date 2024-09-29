import cv2
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps


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


class DenoiseTransform:
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image to be denoised.
        Returns:
            PIL Image: Denoised image.
        """
        # Convert PIL Image to NumPy array
        img_array = np.array(img)
        # Convert RGB to BGR for OpenCV
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        # Apply denoising
        denoised_array = cv2.fastNlMeansDenoisingColored(
            img_array, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
        )
        # Convert back to RGB
        denoised_array = cv2.cvtColor(denoised_array, cv2.COLOR_BGR2RGB)
        # Convert back to PIL Image
        return Image.fromarray(denoised_array)


class ColorSpaceTransform:
    def __init__(self, color_space):
        """
        Args:
            color_space (str): Target color space ('GRAY', 'RGB', 'HSV').
        """
        self.color_space = color_space

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.
        Returns:
            PIL Image: Image converted to the target color space.
        """
        if self.color_space == "GRAY":
            return img.convert("L")
        elif self.color_space == "RGB":
            return img.convert("RGB")
        elif self.color_space == "HSV":
            return img.convert("HSV")
        else:
            raise ValueError(f"Color space '{self.color_space}' not supported.")


class EqualizationTransform:
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.
        Returns:
            PIL Image: Image after histogram equalization.
        """
        if img.mode != "RGB":
            # For grayscale images
            return ImageOps.equalize(img)

        # Split into individual channels
        r, g, b = img.split()

        # Equalize each channel
        r_eq = ImageOps.equalize(r)
        g_eq = ImageOps.equalize(g)
        b_eq = ImageOps.equalize(b)

        return Image.merge("RGB", (r_eq, g_eq, b_eq))
