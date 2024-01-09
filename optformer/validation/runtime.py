# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common validator logic for runtime / output checking."""

from typing import Any, Sequence, Tuple, Union
import numpy as np

FloatLike = Union[float, np.ndarray]
IntLike = Union[int, np.ndarray]
IntervalType = Tuple[float, float]


def assert_in_interval(interval: IntervalType, x: FloatLike) -> None:
  """Checks if all values in x are within the interval."""
  low, high = interval
  if not np.logical_and(x >= low, x <= high).all():
    raise ValueError(f"Input {x} out of bounds from [{low}, {high}].")


def assert_is_int_like(x: IntLike) -> None:
  """Checks if array is type int."""
  if isinstance(x, np.ndarray) and x.dtype not in [np.int32, np.int64]:
    raise ValueError(f"Input {x} has non integer type {x.dtype}.")


def assert_length(x: Sequence[Any], length: int) -> None:
  if len(x) != length:
    raise ValueError(f"Sequence length {len(x)} != expected {length}.")


def assert_all_elements_same(x: Sequence[Any]) -> None:
  """Checks if all elements in x are the same.

  Args:
    x: a sequence of elements

  NOTE: Be careful about checking a sequence of mutable objects.
  """
  if not x:
    return

  if not all(y == x[0] for y in x):
    raise ValueError(f"Not all elements in {x} are the same")
