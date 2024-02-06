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

"""Base class for mappings between tokens and objects."""

import abc
import re
from typing import Any, Generic, Sequence, Tuple, Type, TypeVar

import attrs
from optformer.common.serialization import base
from optformer.validation import runtime
import ordered_set

_V = TypeVar('_V')


# TODO: Allow for different forward/backward types.
class TokenSerializer(base.Serializer[_V], base.Deserializer[_V]):
  """Base class for mapping an object to custom tokens."""

  DELIMITERS: Tuple[str, str] = ('<', '>')


class UnitTokenSerializer(TokenSerializer[_V]):
  """Bijective mapping between single object and single token."""

  def to_str(self, obj: _V) -> str:
    left_d, right_d = self.DELIMITERS
    return f'{left_d}{obj}{right_d}'

  def from_str(self, s: str) -> _V:
    left_d, right_d = self.DELIMITERS
    pattern = f'{left_d}{self.regex_type}{right_d}'
    m = re.fullmatch(pattern, s)
    if not m:
      raise ValueError(f'Input string {s} is not a valid token.')
    return self.type(m.group(1))

  @property
  @abc.abstractmethod
  def regex_type(self) -> str:
    """Regex type used for deserialization."""

  @property
  @abc.abstractmethod
  def type(self) -> Type[_V]:
    """Type of the token value, used for deserialization."""


class IntegerTokenSerializer(UnitTokenSerializer[int]):

  @property
  def regex_type(self) -> str:
    return '([-+]?\\d+)'

  @property
  def type(self) -> Type[int]:
    return int


class StringTokenSerializer(UnitTokenSerializer[str]):

  @property
  def regex_type(self) -> str:
    return '(.*?)'

  @property
  def type(self) -> Type[str]:
    return str


@attrs.define
class UnitSequenceTokenSerializer(Generic[_V], TokenSerializer[Sequence[_V]]):
  """Bijective mapping between sequence of objects to sequence of tokens.

  Uses type-specific tokenizers with ordered priority to handle every sequence
  object.

  By default, handles integers and strings, e.g. [42, 'x', -1] -> '<42><x><-1>'.
  """

  token_serializers: Sequence[UnitTokenSerializer[_V]] = attrs.field(
      factory=lambda: [IntegerTokenSerializer(), StringTokenSerializer()]
  )

  def to_str(self, obj: Sequence[Any], /) -> str:
    """Performs string conversion on decoder-type inputs."""
    out = []
    for o in obj:
      for token_serializer in self.token_serializers:
        if isinstance(o, token_serializer.type):
          out.append(token_serializer.to_str(o))
          break
      else:
        raise ValueError(f'Type {type(o)} is not supported.')

    return ''.join(out)

  def from_str(self, s: str, /) -> Sequence[Any]:
    left_d, right_d = self.DELIMITERS
    pattern = re.compile(f'{left_d}(.*?){right_d}')
    matches = pattern.finditer(s)

    # Makes best effort to use single tokenizers to deserialize match.
    single_values = []
    for match in matches:
      for token_serializer in self.token_serializers:
        s = f'{left_d}{match.group(1)}{right_d}'
        try:
          v = token_serializer.from_str(s)
          single_values.append(v)
          break
        except ValueError:
          # TODO: Make dedicated `SerializationError`.
          pass
      else:
        raise ValueError(f'Could not deserialize `{s}`.')

    return single_values


class OneToManyTokenSerializer(TokenSerializer[_V]):
  """Maps one object to many (fixed count) tokens."""

  @property
  @abc.abstractmethod
  def num_tokens_per_obj(self) -> int:
    """Number of tokens used to represent each object."""


@attrs.define
class RepeatedUnitTokenSerializer(OneToManyTokenSerializer[_V]):
  """Simply outputs repeats of a unit token."""

  unit_token_serializer: UnitTokenSerializer[_V] = attrs.field()
  num_tokens_per_obj = attrs.field()

  def to_str(self, obj: _V) -> str:
    return self.num_tokens_per_obj * self.unit_token_serializer.to_str(obj)

  def from_str(self, s: str) -> _V:
    left_d, right_d = self.DELIMITERS
    pattern = re.compile(f'{left_d}(.*?){right_d}')
    matches = pattern.finditer(s)
    inner_strs = [match.group(1) for match in matches]

    runtime.assert_all_elements_same(inner_strs)

    s = f'{left_d}{inner_strs[0]}{right_d}'
    return self.num_tokens_per_obj * self.unit_token_serializer.from_str(s)


# TODO: Use this to refactor `ScientificFloatTokenSerializer`.
class CartesianProductTokenSerializer(OneToManyTokenSerializer[_V]):
  """Maps an object to a fixed number of tokens based on cartesian product.

  Output will be of form e.g. <a><b><c>... where <a> is from set A, <b> is from
  set <B>, <c> is from set <C>, etc.
  """

  def all_tokens_used(self) -> ordered_set.OrderedSet[str]:
    """Returns ordered set of all tokens used."""
    out = []
    for i in range(self.num_tokens_per_obj):
      out.extend(self.tokens_used(i))
    return ordered_set.OrderedSet(out)

  @abc.abstractmethod
  def tokens_used(self, index: int) -> ordered_set.OrderedSet[str]:
    """Returns ordered set of tokens used at position `index`."""
