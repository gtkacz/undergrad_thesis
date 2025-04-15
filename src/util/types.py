from collections.abc import Callable
from typing import TypeAlias, TypedDict

from torch import Tensor
from torch.utils.data import Dataset

LossFunction: TypeAlias = Callable[[Tensor, Tensor], Tensor]
TrainingDataset: TypeAlias = Dataset[Tensor]
ValidationDataset: TypeAlias = Dataset[Tensor]
TestingDataset: TypeAlias = Dataset[Tensor]


class ConfusionMatrix(TypedDict):
	true_positives: int
	true_negatives: int
	false_positives: int
	false_negatives: int
