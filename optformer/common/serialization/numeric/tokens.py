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

"""General float serializers using dedicated tokens."""

import re
from typing import Sequence, Union

import attrs
import gin
import numpy as np
from optformer.common.serialization import tokens as tokens_lib
import ordered_set


@gin.configurable
@attrs.define
class DigitByDigitFloatTokenSerializer(
    tokens_lib.CartesianProductTokenSerializer[float]
):
  """Serializes floats digit-by-digit using dedicated tokens.

  NOTE: It was experimentally verified this was the best serialization method.

  A float f can be represented as:

  `s * m * 10^e`

  where:
    s: Positive/Negative sign (+, -)
    m: Mantissa representing leading digits.
    e: Exponent.

  Attributes:
    num_digits: Number of digits in `m`. Each digit (even the leading) is
      between <0> and <9>.
    exponent_range: Controls number of exponent tokens, e.g. if 10, the exponent
      token range will be [<E-10>, <E10>], affecting the range of representable
      floats.
  """

  num_digits: int = attrs.field(default=4)
  exponent_range: int = attrs.field(default=10)

  tokens_serializer: tokens_lib.TokenSerializer[Sequence[Union[str, int]]] = (
      attrs.field(
          kw_only=True,
          factory=tokens_lib.UnitSequenceTokenSerializer,
      )
  )

  @property
  def num_tokens_per_obj(self) -> int:
    return 2 + self.num_digits

  def tokens_used(self, index: int) -> ordered_set.OrderedSet[str]:
    if index < 0 or index >= self.num_tokens_per_obj:
      raise ValueError(f'Index {index} out of bounds.')

    if index == 0:  # beginning
      return ordered_set.OrderedSet(self.tokens_serializer.to_str(['+', '-']))

    elif index == self.num_tokens_per_obj - 1:  # end
      exps = [
          f'E{i}' for i in range(-self.exponent_range, self.exponent_range + 1)
      ]
      return ordered_set.OrderedSet(self.tokens_serializer.to_str(exps))
    else:  # middle (digit)
      digits = list(range(0, 10))
      return ordered_set.OrderedSet(self.tokens_serializer.to_str(digits))

  @property
  def _max_abs_val(self) -> float:
    """Largest representable positive number."""
    return float(self.num_digits * '9') * (10.0**self.exponent_range)

  @property
  def _min_abs_val(self) -> float:
    """Smallest representable positive number."""
    min_mantissa = float('1' + (self.num_digits - 1) * '0')
    return min_mantissa * (10 ** (-self.exponent_range))

  def _round_float(self, f: float) -> float:
    """Rounds float to the closest in-range value."""
    abs_f = abs(f)
    abs_f = min(abs_f, self._max_abs_val)
    if abs_f < self._min_abs_val:
      # Decides whether to move to 0.0 or `min_abs_val`.
      zero_or_min = round(abs_f / self._min_abs_val)
      abs_f = self._min_abs_val * zero_or_min
    return abs_f if f >= 0 else -abs_f

  def to_str(self, f: float, /) -> str:
    f = self._round_float(f)
    s = np.format_float_scientific(
        f,
        precision=self.num_digits - 1,
        min_digits=self.num_digits - 1,
        sign=True,
    )
    # We expect numpy to produce scientific notation of the form `+2.123e+4`.
    # It will round for us and ensure leading digit isn't zero, unless the
    # number is zero.
    m = re.fullmatch('([+-])([0-9.]*)e(.*)', s)
    if not m:
      raise RuntimeError(f'Unexpected numpy notation: {s}')
    sign: str = m.group(1)
    digits = list(m.group(2).replace('.', ''))
    exp = int(m.group(3)) - len(digits) + 1 if f else 0
    return self.tokens_serializer.to_str([sign] + digits + [f'E{exp}'])

  def from_str(self, s: str, /) -> float:
    tokens = self.tokens_serializer.from_str(s)

    sign = -1 if tokens[0] == '-' else 1
    mantissa = int(''.join(map(str, tokens[1:-1])))
    exp = int(''.join(tokens[-1]).lstrip('E'))

    return float(sign * mantissa * 10**exp)
