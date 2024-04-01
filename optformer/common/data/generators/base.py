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

"""Base classes for data generation."""
import abc
from typing import Generic, Optional, TypeVar

_F = TypeVar('_F')


# TODO: Either make Protocol picklable, or change `__call__` to
# a named method. Check whether beam pickling requires __hash__ or __eq__ and
# whether Protocol + Attrs changes them.
class SeededFactory(abc.ABC, Generic[_F]):
  """Base class for creating objects from a seed.

  Note: This isn't a Protocol due to incompatibility with beam's pickling of
  attrs-wrapped classes that try to implement a Protocol.
  """

  @abc.abstractmethod
  def __call__(self, *, seed: Optional[int] = None) -> _F:
    """Returns an object based on seed."""
