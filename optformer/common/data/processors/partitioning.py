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

"""Processors which partition batches of data."""

from typing import Dict, Tuple

import attrs
import numpy as np
from optformer.common.data.processors import base
import tensorflow as tf


@attrs.define
class Partitioner(base.Processor[Dict[str, tf.Tensor]]):
  """Partitions a batch of features according to split ratios."""

  split_ratios: Dict[str, float] = attrs.field(kw_only=True)

  def __attrs_post_init__(self):
    if not np.isclose(sum(self.split_ratios.values()), 1.0):
      raise ValueError(f'Ratios {self.split_ratios} do not sum to 1.0.')

  def __call__(self, features: tf.Tensor) -> Dict[str, tf.Tensor]:
    slices = tf.numpy_function(
        self.numpy_fn, [features], Tout=len(self.split_ratios) * [tf.string]
    )
    return dict(zip(self.split_ratios.keys(), slices, strict=True))

  def numpy_fn(self, s: np.ndarray) -> Tuple[np.ndarray, ...]:
    batch_size = s.shape[0]

    ratios = list(self.split_ratios.values())
    split_indices = (batch_size * np.cumsum(ratios)[:-1]).astype(np.int_)
    return np.split(s, split_indices, axis=0)
