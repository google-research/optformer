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

"""Classes for serializing primitives (values and containers).

Values can be `int`, `float`, `str`, etc.
Containers can be `dict, `list, `np.array`, etc. containing values.
"""

from typing import Any, Dict, Optional, Sequence, Union

import attrs
import numpy as np
from optformer.common.serialization import base

PrimitiveType = Any


@attrs.define(kw_only=True)
class PrimitiveSerializer(base.Serializer[PrimitiveType]):
  """Serializes (possibly-nested) primitive objects.

  This is slightly different from simply using JSON or `str(object)` as we need
  more precise control over e.g. floating point values, spacing, and brackets.
  """

  # Whether to add brackets to sequences, ex: '[a,b,c]' vs 'a,b,c'.
  include_sequence_brackets: bool = attrs.field(default=True)
  # Whether to add brackets to dicts, ex: '{a:b}' vs 'a:b'.
  include_dict_brackets: bool = attrs.field(default=True)

  # Indicator for separating elements in sequences, ex: '[a*b*c]' vs '[a,b,c]'.
  sequence_separator: str = attrs.field(default=',')
  # Indicator for separating elements in dicts, ex: '{a:x,b:y}' vs '{a:x*b:y}'.
  dict_item_separator: str = attrs.field(default=',')

  # Whether to add explicit quotes on dict keys, ex: '{"hi":5}' vs '{hi:5}'.
  # Str dict keys are **not** considered values.
  dict_key_use_quotes: bool = attrs.field(default=False)
  # Whether to apply quotes on string values, ex: '"hello"' vs 'hello'.
  str_use_quotes: bool = attrs.field(default=True)

  integer_serializer: Optional[base.Serializer[int]] = attrs.field(default=None)
  float_serializer: Optional[base.Serializer[float]] = attrs.field(default=None)

  def to_str(self, obj: PrimitiveType, /) -> str:
    """Performs string conversion on a variety of Python primitives."""
    if isinstance(obj, (str, float, int)):
      return self._value_to_str(obj)
    elif obj is None:
      return self._none_to_str(obj)
    elif isinstance(obj, dict):
      return self._dict_to_str(obj)
    elif isinstance(obj, (list, tuple)):
      return self._sequence_to_str(obj)
    elif isinstance(obj, np.ndarray):
      return self._ndarray_to_str(obj)
    else:
      raise ValueError(f'Type {type(obj)} is not supported.')

  def _str_to_str(self, s: str) -> str:
    return '"' + s + '"' if self.str_use_quotes else s

  def _int_to_str(self, i: int) -> str:
    if self.integer_serializer:
      return self.integer_serializer.to_str(i)
    return str(i)

  def _float_to_str(self, f: float) -> str:
    if self.float_serializer:
      return self.float_serializer.to_str(f)
    return '{{:.{}g}}'.format(3).format(f)

  def _bool_to_str(self, b: bool) -> str:
    return str(b)

  def _none_to_str(self, n: type(None)) -> str:
    return 'None'

  def _value_to_str(self, v: Union[bool, str, int, float]) -> str:
    if isinstance(v, str):
      return self._str_to_str(v)
    elif isinstance(v, float):
      return self._float_to_str(v)
    elif isinstance(v, int):
      return self._int_to_str(v)
    elif isinstance(v, bool):
      return self._bool_to_str(v)
    else:
      raise ValueError(f'Type {type(v)} is not a supported value.')

  def _dict_to_str(self, d: Dict[str, Any]) -> str:
    dict_item_strs = []

    for k, v in d.items():
      if self.dict_key_use_quotes:
        key_str = '"' + k + '"'
      else:
        key_str = k
      item = f'{key_str}:{self.to_str(v)}'
      dict_item_strs.append(item)

    dict_interior = self.dict_item_separator.join(dict_item_strs)
    if self.include_dict_brackets:
      return '{' + dict_interior + '}'
    return dict_interior

  def _ndarray_to_str(self, arr: np.ndarray) -> str:
    if arr.dtype in [np.int32, np.int64]:
      interior = self.sequence_separator.join(
          [self._int_to_str(int(v_i)) for v_i in arr]
      )
    elif arr.dtype in [np.float32, np.float64]:
      interior = self.sequence_separator.join(
          [self._float_to_str(float(v_i)) for v_i in arr]
      )
    else:
      raise ValueError(f'np.ndarray type {arr.dtype} is not supported.')

    if self.include_sequence_brackets:
      return '[' + interior + ']'
    return interior

  def _sequence_to_str(self, s: Sequence[Any]) -> str:
    interior = self.sequence_separator.join(map(self.to_str, s))
    if self.include_sequence_brackets:
      return '[' + interior + ']'
    return interior
