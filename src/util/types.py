from typing import Callable, TypeAlias

from torch import Tensor
from torch.utils.data import Dataset

LossFunction: TypeAlias = Callable[[Tensor, Tensor], Tensor]

TrainingDataset: TypeAlias = Dataset[Tensor]
ValidationDataset: TypeAlias = Dataset[Tensor]
TestDataset: TypeAlias = Dataset[Tensor]
