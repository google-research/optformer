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

"""PyGlove Featurizers."""

import functools
from typing import Dict

import attrs
from optformer.common import serialization as s_lib
from optformer.common.data import featurizers
from optformer.pyglove import serialization as pg_serializers
from optformer.pyglove import types
import tensorflow as tf


@attrs.define(kw_only=True)
class PyGloveStudyFeaturizer(featurizers.Featurizer[types.PyGloveStudy]):
  """Directly featurizes a PyGlove Study."""

  _encoder_serializer: s_lib.Serializer[types.PyGloveStudy] = attrs.field(
      factory=pg_serializers.SuggestionsKeyValueSerializer
  )
  _decoder_serializer: s_lib.Serializer[types.PyGloveStudy] = attrs.field(
      factory=pg_serializers.QuantizedMeasurementSerializer
  )

  @functools.cached_property
  def output_types(self) -> Dict[str, tf.DType]:
    return {
        'inputs': tf.string,
        'targets': tf.string,
    }

  @functools.cached_property
  def output_shapes(self) -> Dict[str, tf.TensorShape]:
    return {
        'inputs': tf.TensorShape([]),
        'targets': tf.TensorShape([]),
    }

  @functools.cached_property
  def empty_output(self) -> Dict[str, tf.Tensor]:
    return {
        'inputs': tf.constant('', dtype=tf.string),
        'targets': tf.constant('', dtype=tf.string),
    }

  def to_features(self, study: types.PyGloveStudy, /) -> Dict[str, tf.Tensor]:
    inputs = self._encoder_serializer.to_str(study)
    targets = self._decoder_serializer.to_str(study)

    return {
        'inputs': tf.constant(inputs, dtype=tf.string),
        'targets': tf.constant(targets, dtype=tf.string),
    }
