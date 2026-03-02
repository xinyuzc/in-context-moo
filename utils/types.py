"""Type hints and type annotations: NestedFloatList, FloatListOrNested, FloatListOrNestedOrTensor"""

from typing import List, Union
from torch import Tensor

NestedFloatList = List[List[float]]
FloatListOrNested = Union[List[float], NestedFloatList]
FloatListOrNestedOrTensor = Union[FloatListOrNested, Tensor]
