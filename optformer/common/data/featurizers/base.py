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

"""Featurizers for creating TensorDicts, eventually to be used in data pipelines."""
import abc
from typing import Dict, Generic, TypeVar

import tensorflow as tf


_T = TypeVar('_T')


class Featurizer(Generic[_T], abc.ABC):
  """Converts an object (ex: study) into features.

  `to_features()` always returns a dictionary consistent with `output_shapes`
  and `output_types`, regardless of the input.
  """

  @abc.abstractmethod
  def to_features(self, obj: _T, /) -> Dict[str, tf.Tensor]:
    """Returns features and raises ValueError in case of failure."""

  @property
  @abc.abstractmethod
  def output_types(self) -> Dict[str, tf.DType]:
    """Returns the dtypes of values returned by to_features()."""

  @property
  @abc.abstractmethod
  def output_shapes(self) -> Dict[str, tf.TensorShape]:
    """Returns the shapes of values returned by to_features()."""

  @property
  @abc.abstractmethod
  def empty_output(self) -> Dict[str, tf.Tensor]:
    """Empty output to use in case an error is raised.

    Example use:
      featurizer: Featurizer[tf.Tensor]
      def map_fn(entry: tf.Tensor) -> tuple[bool, Dict[str, tf.Tensor]]:
        try:
          return True, featurizer.to_features(entry)
        except Exception:
          return False, featurizer.empty_output
      dataset.map(featurizer.to_features).filter(lambda x: x[0])
    """
