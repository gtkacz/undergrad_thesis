from enum import Enum


class DenoisingMethod(Enum):
    CHAMBOLLE = 'chambolle'
    BILATERAL = 'bilateral'
    WAVELET = 'wavelet'
    NL_MEANS = 'nl_means'
    MEDIAN = 'median'
    MEAN = 'mean'
    GAUSSIAN = 'gaussian'


class SegmentationMethod(Enum):
    THRESHOLD = 'threshold'
    WATERSHED = 'watershed'
    FELZENSZWALB = 'felzenszwalb'
    QUICKSHIFT = 'quickshift'
    SLIC = 'slic'


class Augmentation(Enum):
    FLIP = 'flip'
    ROTATE = 'rotate'
    TRANSLATE = 'translate'
    SCALE = 'scale'
    SHEAR = 'shear'
    PERSPECTIVE = 'perspective'


class ColorDomain(Enum):
    RGB = 'rgb'
    GRAYSCALE = 'grayscale'
    HSV = 'hsv'
    LAB = 'lab'
    YUV = 'yuv'
    YCBCR = 'ycbcr'


class EqualizationMethod(Enum):
    THRESHOLD = 'threshold'
    WATERSHED = 'watershed'
    FELZENSZWALB = 'felzenszwalb'
    QUICKSHIFT = 'quickshift'
    SLIC = 'slic'
