from typing import Callable, TypeAlias

from torch import Tensor

LossFunction: TypeAlias = Callable[[Tensor, Tensor], Tensor]
