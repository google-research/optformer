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

"""Featurizer for Vizier; used for training only."""

import functools
from typing import Sequence

import attrs
import numpy as np
from optformer.common.data import featurizers
from optformer.common.data import filters
from optformer.embed_then_regress import normalization
from optformer.embed_then_regress.vizier import serializers
from optformer.vizier.data import augmenters
import tensorflow.google.compat.v2 as tf
from vizier import pyvizier as vz


VizierFilter = filters.Filter[vz.ProblemAndTrials]


@attrs.define(init=True, kw_only=True)
class ICLFeaturizer(featurizers.Featurizer[vz.ProblemAndTrials]):
  """Converts a Vizier study to strings suitable for ICL training."""

  min_context: int = attrs.field(default=5)
  max_context: int = attrs.field(default=100)
  max_trials: int = attrs.field(default=120)

  warper: normalization.StatefulWarper = attrs.field(
      factory=normalization.default_warper
  )

  _prefilters: Sequence[VizierFilter] = attrs.field(factory=list)
  _augmenters: Sequence[augmenters.VizierAugmenter] = attrs.field(factory=list)
  _postfilters: Sequence[VizierFilter] = attrs.field(factory=list)

  @functools.cached_property
  def element_spec(self) -> dict[str, tf.TensorSpec]:
    return {
        'x': tf.TensorSpec(shape=(None,), dtype=tf.string),  # L
        'y': tf.TensorSpec(shape=(None,), dtype=tf.float32),  # L
        'metadata': tf.TensorSpec(shape=(), dtype=tf.string),  # Scalar
        'mask': tf.TensorSpec(shape=(None,), dtype=tf.bool),  # L
    }

  @functools.cached_property
  def empty_output(self) -> dict[str, tf.Tensor]:
    return {
        'x': tf.constant([''], dtype=tf.string),
        'y': tf.constant([0.0], dtype=tf.float32),
        'metadata': tf.constant('', dtype=tf.string),
        'mask': tf.constant([False], dtype=tf.bool),
    }

  def to_features(self, study: vz.ProblemAndTrials, /) -> dict[str, tf.Tensor]:
    # pylint:disable=invalid-name
    for study_filter in self._prefilters:
      if not study_filter(study):
        raise ValueError(f'{study_filter} rejected study.')

    for study_augmenter in self._augmenters:
      # NOTE: Study may be modified in-place rather than copied.
      study = study_augmenter.augment_study(study)

    for study_filter in self._postfilters:
      if not study_filter(study):
        raise ValueError(f'{study_filter} rejected study.')

    if not study.trials:
      raise ValueError('Study has no trials.')

    # Limit maximum sequence length.
    study.trials[:] = study.trials[: self.max_trials]

    problem = study.problem
    ss_str = serializers.SearchSpaceSerializer().to_str(problem.search_space)
    m_name = problem.metric_information.item().name
    L = len(study.trials)

    xs = []
    ys = []
    x_serializer = serializers.XSerializer(study.problem.search_space)
    for trial in study.trials:
      xs.append(x_serializer.to_str(trial))
      ys.append(trial.final_measurement_or_die.metrics[m_name].value)

    num_context = np.random.randint(self.min_context, self.max_context)

    # Edit masking.
    mask = np.ones(L, dtype=bool)
    # Apply random permutation.
    perm = np.random.permutation(L)
    xs = [xs[i] for i in perm]
    ys = [ys[i] for i in perm]
    mask[num_context:] = False

    # Warp y-values.
    ys = np.array(ys)
    self.warper.train(ys[:num_context])
    ys = self.warper.warp(ys)

    if np.isnan(ys).any():
      raise ValueError(f'Y values contain NaN: {ys}')

    return {
        'x': tf.constant(xs, dtype=tf.string),
        'y': tf.constant(ys, dtype=tf.float32),
        'metadata': tf.constant(ss_str, dtype=tf.string),
        'mask': tf.constant(mask, dtype=tf.bool),
    }
