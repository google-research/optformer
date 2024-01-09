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

"""Base classes for serializers."""
import abc
from typing import Generic, Optional, Protocol, TypeVar

_T = TypeVar('_T')


class Serializer(abc.ABC, Generic[_T]):
  """Base class for stringifying objects.

  Should always have deterministic behavior (i.e. the same input value should
  always map to the same output value).
  """

  @abc.abstractmethod
  def to_str(self, obj: _T, /) -> str:
    """Turns an object to text."""


class SerializerFactory(Protocol[_T]):
  """Factory for creating serializers.

  Useful abstraction for simulating random serialization behavior.
  """

  @abc.abstractmethod
  def __call__(self, *, seed: Optional[int] = None) -> Serializer[_T]:
    """Creates the Serializer from seed."""


class Deserializer(abc.ABC, Generic[_T]):
  """Base class for deserializing strings.

  Should always have deterministic behavior (i.e. the same input value should
  always map to the same output value).
  """

  @abc.abstractmethod
  def from_str(self, s: str, /) -> _T:
    """Turns the string back into the object."""
