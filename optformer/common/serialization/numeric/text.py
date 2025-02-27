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

"""Text-based serializers of numbers."""

import attrs
import numpy as np
from optformer.common.serialization import base


class FloatTextSerializer(base.Serializer[float], base.Deserializer[float]):
  """Convenient type to denote reversibility."""


@attrs.define
class SimpleScientificFloatTextSerializer(FloatTextSerializer):
  """Represent floats in a simple scientific notation.

  Mantissa is always same length. Exponent is 2 digits unless needed to be 3.
  """

  precision: int = attrs.field(default=2, kw_only=True)

  def to_str(self, obj: float, /) -> str:
    return format(obj, f'.{self.precision}e')

  def from_str(self, s: str, /) -> float:
    return float(s)


@attrs.define
class ScientificFloatTextSerializer(FloatTextSerializer):
  """Represent floats in raw scientific notation.

  Meant to be used to fine-tune most pretrained LLMs.

  NOTE: Because the serialized string output may be of variable length, consider
  turning on EOS usage for training and inference.
  """

  exp_digits: int = attrs.field(default=2, kw_only=True)
  precision: int = attrs.field(default=2, kw_only=True)

  def to_str(self, obj: float, /) -> str:
    return np.format_float_scientific(
        obj,
        unique=True,
        trim='0',
        precision=self.precision,
        exp_digits=self.exp_digits,
    )

  def from_str(self, s: str, /) -> float:
    return float(s)

  @property
  def max_num_chars(self) -> int:
    # leading_digit + `.` + precision_digits + 'e' + sign + exp_digits
    return 1 + 1 + self.precision + 1 + 1 + self.exp_digits


@attrs.define
class ExpandedScientificFloatSerializer(FloatTextSerializer):
  """Represent floats in a special expanded scientific notation.

  Meant to be used to fine-tune most pretrained LLMs. From
  https://arxiv.org/abs/2102.13019.

  Examples:
    123.0  = [+ 1 10e2 2 10e1 3 10e0]
    -12.3 = [- 1 10e1 2 10e0 3 10e-1]
    1.0 = [+ 1 10e0]
    0 = [+ 0 10e0]
  """

  precision: int = attrs.field(default=5, kw_only=True)

  def to_str(self, f: float, /) -> str:
    if f == 0.0:
      return '[+ 0 10e0]'
    # the str output can look like: 1.e+00, -1.2345e+02, -1.23e-04, etc.
    scientific_str = np.format_float_scientific(
        f,
        unique=True,
        precision=self.precision,
        exp_digits=2,
    )
    num_str, exp_str = scientific_str.split('e')
    exp = int(exp_str)
    segs = []
    for s in num_str:
      # Ignore decimal and sign
      if s in ('.', '-'):
        continue
      segs.append(f'{s} 10e{exp}')
      exp -= 1
    output = ' '.join(segs)
    if f < 0:
      output = '- ' + output
    else:
      output = '+ ' + output
    return f'[{output}]'

  def from_str(self, s: str, /) -> float:
    # Remove brackets and parse by spaces.
    toks = s[1:-1].split()

    sign = -1 if toks.pop(0) == '-' else 1
    # TODO: float(10e2) in Python is actually 1000.0. Consider
    # being consistent w/ Python conventions.
    f = 0.0
    for digit_index in range(0, len(toks), 2):
      digit = int(toks[digit_index])
      power = float(toks[digit_index + 1]) / 10.0
      f += digit * power

    return sign * f


@attrs.define
class SimpleFloatTextSerializer(FloatTextSerializer):
  """Represent floats as simple text."""

  precision: int = attrs.field(default=5, kw_only=True)

  def to_str(self, f: float, /) -> str:
    return str(round(f, self.precision))

  def from_str(self, s: str, /) -> float:
    return float(s)


@attrs.define(kw_only=True)
class NormalizedFloatSerializer(FloatTextSerializer):
  """Represent floats in base B, for values within [0, 1]."""

  base: int = attrs.field(default=10)
  precision: int = attrs.field(default=5)
  digit_map: str = attrs.field(default='0123456789abcdefghijklmnopqrstuvwxyz')

  def __attrs_post_init__(self):
    if self.base > len(self.digit_map):
      raise ValueError(
          f'Base {self.base} is too large for the provided digit map.'
      )
    if self.base < 2:
      raise ValueError(f'Base {self.base} must be at least 2.')

  def to_str(self, f: float, /) -> str:
    if f == 1.0:
      return '0.' + self.digit_map[self.base - 1] * self.precision
    if not 0 <= f <= 1:
      raise ValueError(f'Input float {f} is not within the range [0, 1].')
    if f == 0.0:
      return '0.' + '0' * self.precision

    fractional_part = f

    base_b_fraction = ''
    for _ in range(self.precision):
      fractional_part *= self.base
      digit_value = int(fractional_part)
      base_b_fraction += self.digit_map[digit_value]  # use digit map
      fractional_part -= digit_value

    return f'0.{base_b_fraction}'

  def from_str(self, s: str, /) -> float:
    if not s.startswith('0.'):
      raise ValueError(
          f"Input string '{s}' is not a valid base {self.base} fractional"
          " representation starting with '0.'"
      )

    fraction_str = s[2:]  # remove "0." prefix
    f = 0.0
    for i, digit_char in enumerate(fraction_str):
      digit_value = self.digit_map.index(digit_char)  # reverse digit map
      f += digit_value * (self.base ** -(i + 1))
    return f
