from enum import Enum as E
from enum import EnumMeta

from torch import nn


class MetaEnum(EnumMeta):
	@property
	def names(cls) -> list[str]:
		return sorted(list(cls.__members__.keys()))


class Enum(E, metaclass=MetaEnum):
	pass


class DenoisingMethod(Enum):
	CHAMBOLLE = "chambolle"
	BILATERAL = "bilateral"
	WAVELET = "wavelet"
	NL_MEANS = "nl_means"
	MEDIAN = "median"
	MEAN = "mean"
	GAUSSIAN = "gaussian"


class SegmentationMethod(Enum):
	THRESHOLD = "threshold"
	WATERSHED = "watershed"
	FELZENSZWALB = "felzenszwalb"
	QUICKSHIFT = "quickshift"
	SLIC = "slic"


class Augmentation(Enum):
	FLIP = "flip"
	ROTATE = "rotate"
	TRANSLATE = "translate"
	SCALE = "scale"
	SHEAR = "shear"
	PERSPECTIVE = "perspective"


class ColorDomain(Enum):
	RGB = "rgb"
	GRAYSCALE = "grayscale"
	HSV = "hsv"
	LAB = "lab"
	YUV = "yuv"
	YCBCR = "ycbcr"


class EqualizationMethod(Enum):
	THRESHOLD = "threshold"
	WATERSHED = "watershed"
	FELZENSZWALB = "felzenszwalb"
	QUICKSHIFT = "quickshift"
	SLIC = "slic"


class LossFunction(Enum):
	L1 = nn.L1Loss
	NLL = nn.NLLLoss
	PoissonNLL = nn.PoissonNLLLoss
	GaussianNLL = nn.GaussianNLLLoss
	KLDiv = nn.KLDivLoss
	MSE = nn.MSELoss
	BCE = nn.BCELoss
	BCEWithLogits = nn.BCEWithLogitsLoss
	HingeEmbedding = nn.HingeEmbeddingLoss
	MultiLabelMargin = nn.MultiLabelMarginLoss
	SmoothL1 = nn.SmoothL1Loss
	Huber = nn.HuberLoss
	SoftMargin = nn.SoftMarginLoss
	CrossEntropy = nn.CrossEntropyLoss
	MultiLabelSoftMargin = nn.MultiLabelSoftMarginLoss
	CosineEmbedding = nn.CosineEmbeddingLoss
	MarginRanking = nn.MarginRankingLoss
	MultiMargin = nn.MultiMarginLoss
	TripletMargin = nn.TripletMarginLoss
	TripletMarginWithDistance = nn.TripletMarginWithDistanceLoss
	CTC = nn.CTCLoss


class Optimizer(Enum):
	pass
