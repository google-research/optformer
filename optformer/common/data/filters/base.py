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

"""Filters which entirely reject the object."""

import abc
from typing import Generic, TypeVar

_T = TypeVar('_T')


class Filter(Generic[_T], abc.ABC):
  """Filter abstraction."""

  def __call__(self, obj: _T, /) -> bool:
    """Filters an object.

    Args:
      obj: Object to be filtered.

    Returns:
      True if the object is useful.

    Raises:
      ValueError: Instead of returning False, optionally
        raise an Error to improve logging at the cost of
        performance.
    """
