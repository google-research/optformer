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

"""Useful featurizers for testing."""

import functools
from optformer.common.data.featurizers import base
import tensorflow as tf


class IdentityFeaturizer(base.Featurizer[str]):
  """Simply returns identity of string input."""

  def to_features(self, obj: str, /) -> dict[str, tf.Tensor]:
    return {
        'inputs': tf.constant(obj, dtype=tf.string),
        'targets': tf.constant(obj, dtype=tf.string),
    }

  @functools.cached_property
  def element_spec(self) -> dict[str, tf.TensorSpec]:
    return {
        'inputs': tf.TensorSpec(shape=(), dtype=tf.string),
        'targets': tf.TensorSpec(shape=(), dtype=tf.string),
    }

  @functools.cached_property
  def empty_output(self) -> dict[str, tf.Tensor]:
    return {
        'inputs': tf.constant('', dtype=tf.string),
        'targets': tf.constant('', dtype=tf.string),
    }
