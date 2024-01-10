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

"""Base class for processing features."""

from typing import Protocol, TypeVar
import tensorflow as tf


_T = TypeVar('_T')


class Processor(Protocol[_T]):
  """Atomic method for processing tensors.

  The output type should be compatible with `tf.data.Dataset.map()`, e.g.
   1. tf.Tensor
   2. Mapping[str, tf.Tensor]
   3. Tuple[tf.Tensor, ...]

   There are utility functions such as @seqio.map_over_dataset to automatically
   convert Processors to dataset mappers.
  """

  def __call__(self, features: tf.Tensor) -> _T:
    """Processes the features."""
