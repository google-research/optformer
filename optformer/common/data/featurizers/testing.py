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
import attrs
from optformer.common.data.featurizers import base
import tensorflow as tf


@attrs.define
class IdentityFeaturizer(base.Featurizer[str]):
  """Simply returns identity of string input."""

  rank: int = attrs.field(default=0)

  def to_features(self, obj: str, /) -> dict[str, tf.Tensor]:
    ranked_obj = tf.constant(obj, dtype=tf.string)
    for _ in range(self.rank):
      ranked_obj = tf.expand_dims(ranked_obj, 0)

    return {'inputs': ranked_obj, 'targets': ranked_obj}

  @functools.cached_property
  def element_spec(self) -> dict[str, tf.TensorSpec]:
    shape = [None for _ in range(self.rank)]
    return {
        'inputs': tf.TensorSpec(shape, dtype=tf.string),
        'targets': tf.TensorSpec(shape, dtype=tf.string),
    }

  @functools.cached_property
  def empty_output(self) -> dict[str, tf.Tensor]:
    ranked_obj = tf.constant('', dtype=tf.string)
    for _ in range(self.rank):
      ranked_obj = tf.expand_dims(ranked_obj, 0)
    return {'inputs': ranked_obj, 'targets': ranked_obj}


@attrs.define
class PassThroughFeaturizer(base.Featurizer):
  """Input is already featurized, just pass it through."""

  ref_featurizer: base.Featurizer = attrs.field(init=True)

  def to_features(self, obj: dict[str, tf.Tensor], /) -> dict[str, tf.Tensor]:
    return obj

  @property
  def element_spec(self) -> dict[str, tf.TensorSpec]:
    return self.ref_featurizer.element_spec

  @property
  def empty_output(self) -> dict[str, tf.Tensor]:
    return self.ref_featurizer.empty_output
