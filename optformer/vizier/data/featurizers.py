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

"""Vizier Featurizers."""

import functools
from typing import Sequence

import attrs
from optformer.common import serialization as s_lib
from optformer.common.data import featurizers
from optformer.common.data import filters
from optformer.vizier.data import augmenters
import tensorflow as tf
from vizier import pyvizier as vz


# `slots=False` allows `functools.cached_property` to work properly.
@attrs.define(slots=False)
class VizierStudyFeaturizer(featurizers.Featurizer[vz.ProblemAndTrials]):
  """Directly featurizes a generic study."""

  # ---------------------------------------------------------------------------
  # Inputs/Targets serializers act on an entire study for generality.
  # Written as factories to allow randomization/data augmentation in training.
  # ---------------------------------------------------------------------------
  _inputs_serializer_factory: s_lib.SerializerFactory[vz.ProblemAndTrials] = (
      attrs.field(init=True)
  )
  _targets_serializer_factory: s_lib.SerializerFactory[vz.ProblemAndTrials] = (
      attrs.field(init=True)
  )
  # ---------------------------------------------------------------------------
  # Dedicated augmenters and filters.
  # ---------------------------------------------------------------------------
  _study_augmenters: Sequence[augmenters.VizierAugmenter] = attrs.field(
      default=tuple(),
      kw_only=True,
  )
  _require_idempotent_augmenters = attrs.field(default=False, kw_only=True)

  _study_filters: Sequence[filters.Filter[vz.ProblemAndTrials]] = attrs.field(
      factory=tuple, kw_only=True
  )
  _features_filters: Sequence[filters.Filter[dict[str, tf.Tensor]]] = (
      attrs.field(factory=tuple, kw_only=True)
  )

  def __attrs_post_init__(self):
    if self._require_idempotent_augmenters:
      for augmenter in self._study_augmenters:
        if not isinstance(augmenter, augmenters.VizierIdempotentAugmenter):
          raise ValueError(
              f'Invalid augmenter that is not idempotent: {augmenter}'
          )

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

  def to_features(self, study: vz.ProblemAndTrials, /) -> dict[str, tf.Tensor]:
    for study_augmenter in self._study_augmenters:
      # NOTE: Study may be modified in-place rather than copied.
      study = study_augmenter.augment_study(study)

    for study_filter in self._study_filters:
      if not study_filter(study):
        raise ValueError(f'{study_filter} rejected study.')

    inputs = self._inputs_serializer_factory().to_str(study)
    targets = self._targets_serializer_factory().to_str(study)
    features = {
        'inputs': tf.constant(inputs, dtype=tf.string),
        'targets': tf.constant(targets, dtype=tf.string),
    }

    for features_filter in self._features_filters:
      if not features_filter(features):
        raise ValueError(f'{features_filter} rejected features.')

    return features
