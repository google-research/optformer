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

"""Shuffling-related dataset functions."""

from typing import Optional

import attrs
from optformer.common.data.datasets import base
import tensorflow as tf


@attrs.define
class ShuffleDatasetFn(base.DatasetFn[tf.data.Dataset]):
  """Customized shuffling API. Might get modified over time."""

  # NOTE: Choose to be high enough to simulate IID sampling (when stuck with
  # streaming data) but low enough it doesn't blow up RAM.
  buffer_size: int = attrs.field(default=1000, kw_only=True)
  seed: Optional[int] = attrs.field(default=None, kw_only=True)

  def __call__(self, source: tf.data.Dataset) -> tf.data.Dataset:
    return source.shuffle(self.buffer_size, seed=self.seed)
