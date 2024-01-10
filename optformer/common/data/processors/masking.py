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

"""Masking-based preprocessors."""

from typing import Sequence, Union

import attrs
from optformer.common.data.processors import base
import tensorflow as tf


ValueType = Union[tf.Tensor, int, float, str, bool]


@attrs.define
class ValueMask(base.Processor[tf.Tensor]):
  """Computes value-matched mask in the matrix.

  If our masked_values are * and |, then:
  [*, |, x, y]

  will have mask:
  [0, 0, 1, 1].
  """

  masked_values: Sequence[ValueType]

  def __call__(self, features: tf.Tensor) -> tf.Tensor:
    tensor = features
    mask = tf.fill(tf.shape(tensor), True)
    for v in self.masked_values:
      mask = tf.logical_and(mask, tf.not_equal(tensor, v))

    return mask


@attrs.define
class BetweenDelimitersMask(base.Processor[tf.Tensor]):
  """Computes the mask for a tensor given two ordered delimiters.

  For example, if our left/right delimiters are '*' and '|', then the following
  input:
  [*, x, y, z, |, *, |]

  will have mask:
  [0, 1, 1, 1, 0, 0, 0].
  """

  left: ValueType
  right: ValueType

  def __call__(self, features: tf.Tensor) -> tf.Tensor:
    tensor = features
    left_match = tf.cast(tensor == self.left, tf.int32)
    right_match = tf.cast(tensor == self.right, tf.int32)

    # Check if count(left) == count(right)
    left_count = tf.reduce_sum(left_match, axis=-1)
    right_count = tf.reduce_sum(right_match, axis=-1)
    tf.debugging.assert_equal(left_count, right_count)

    # If our example tensor is [x, *, y, |], then example outputs are commented:
    left_cs = tf.math.cumsum(left_match, axis=-1)  # [0, 1, 1, 1]
    right_cs = tf.math.cumsum(right_match, axis=-1)  # [0, 0, 0, 1]
    left_cs_slice = left_cs[..., :-1]  # [0, 1, 1]
    zeros = tf.zeros(shape=left_cs_slice.shape[:-1] + [1], dtype=tf.int32)
    shifted_left_cs = tf.concat([zeros, left_cs_slice], axis=-1)  # [0, 0, 1, 1]
    mask = shifted_left_cs - right_cs  # [0, 0, 1, 0]

    # Check if there are no -1's (from wrong right -> left orderings).
    all_ones_and_zeros = tf.reduce_all((mask == 0) | (mask == 1))
    tf.debugging.assert_equal(True, all_ones_and_zeros)

    mask = tf.cast(mask, tf.bool)
    return mask
