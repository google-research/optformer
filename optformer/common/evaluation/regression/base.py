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

"""Base classes for regression evaluation."""

import abc
from typing import Generic, Sequence, TypeVar
import numpy as np
from optformer.common.data import augmenters
from optformer.common.data import filters
import tensorflow_datasets as tfds

_T = TypeVar('_T')


class RegressorEvaluator(abc.ABC, Generic[_T]):
  """Evaluates a regressor's prediction performance."""

  TRAIN = tfds.Split.TRAIN
  VALIDATION = tfds.Split.VALIDATION
  TEST = tfds.Split.TEST

  # These are class-global to enforce fairness across all evaluators.
  FILTERS: Sequence[filters.Filter[_T]]
  AUGMENTERS: Sequence[augmenters.Augmenter[_T]]

  @abc.abstractmethod
  def evaluate_metric(
      self, objs: dict[str, _T]
  ) -> dict[str, float | np.ndarray]:
    """Obtains prediction metrics.

    Args:
      objs: Mapping from split to regular obj. It is the job of the subclass to
        determine how to best use the objs.

    Returns:
      Mapping from metric name to metric value.
    """

  @classmethod
  def validate_objs(cls, objs: dict[str, _T]) -> bool:
    try:
      return all([filt(s) for s in objs.values() for filt in cls.FILTERS])  # pylint:disable=g-complex-comprehension
    except ValueError:
      return False

  @classmethod
  def augment_objs(cls, objs: dict[str, _T]) -> dict[str, _T]:
    """NOTE: Object might have been augmented in-place."""
    for augmenter in cls.AUGMENTERS:
      objs = {k: augmenter.augment(v) for k, v in objs.items()}
    return objs
