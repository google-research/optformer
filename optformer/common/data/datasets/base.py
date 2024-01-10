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

"""Base abstractions for dataset functions."""
from typing import Protocol, TypeVar

import tensorflow as tf

_S = TypeVar('_S')


class DatasetFn(Protocol[_S]):
  """Base Dataset Function class."""

  def __call__(self, source: _S) -> tf.data.Dataset:
    """Transforms a source to a TF Dataset."""
