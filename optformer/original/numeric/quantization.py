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

"""Quantizers (e.g. for converting floats to normalized ints)."""

from typing import Union

import attrs
import numpy as np
from optformer.original.numeric import base
from optformer.validation import runtime

FloatLike = Union[float, np.ndarray]
IntLike = Union[int, np.ndarray]


@attrs.define
class NormalizedQuantizer(base.NumericMapper[FloatLike, IntLike]):
  """Quantizes floating point values, for e.g. eventual integer tokenization."""

  num_bins: int = attrs.field(
      kw_only=True,
      default=1000,
      validator=[attrs.validators.instance_of(int), attrs.validators.ge(1)],
  )
  dequantization_shift: float = attrs.field(
      kw_only=True,
      default=0.5,
      validator=[attrs.validators.lt(1.0), attrs.validators.ge(0.0)],
  )

  def map(self, x: FloatLike) -> IntLike:
    """Convert a fraction number to an integer.

    If we have 1000 bins, then e.g. 0.12345 --> 123, and our range [0.0, 1.0]
    --> [0, 999].

    Args:
      x: Number or array with values in [0.0, 1.0].

    Returns:
      Integer bin within [0, num_bins - 1].
    """
    runtime.assert_in_interval((0.0, 1.0), x)

    if isinstance(x, np.ndarray):
      return np.minimum((x * self.num_bins).astype(np.int32), self.num_bins - 1)
    else:
      return min(int(x * self.num_bins), self.num_bins - 1)

  def unmap(self, y: IntLike) -> FloatLike:
    """Reverse the quantization method.

    Given an integer in e.g. [0, num_bins - 1], maps it back to [0.0, 1.0].

    Args:
      y: Integer or np.int array with values in [0, num_bins - 1].

    Returns:
      Normalized value in [0.0, 1.0].
    """

    runtime.assert_is_int_like(y)
    runtime.assert_in_interval((0, self.num_bins - 1), y)

    return (y + self.dequantization_shift) / self.num_bins
