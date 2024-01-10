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

"""Base class for augmenting objects during training."""

import abc
from typing import Generic, TypeVar

T = TypeVar('T')


class Augmenter(Generic[T], abc.ABC):
  """Base data augmenter class."""

  @abc.abstractmethod
  def augment(self, obj: T, /) -> T:
    """Augments the object.

    For efficiency, ideally the object should be augmented in-place. Copying
    should be used as a last resort.

    Args:
      obj: Object to be augmented.

    Returns:
      Augmented object. Could be a reference to the original input object if
        modifying in-place.
    """
