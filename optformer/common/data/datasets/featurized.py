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

"""Classes for applying featurizers over datasets."""

from typing import Sequence

from absl import logging
import attrs
from optformer.common.data import featurizers
from optformer.common.data.datasets import base
import tensorflow as tf


@attrs.define
class FeaturizedDatasetFn(base.DatasetFn[tf.data.Dataset]):
  """Featurizes a dataset."""

  featurizer: featurizers.Featurizer = attrs.field()

  def __call__(self, source: tf.data.Dataset) -> tf.data.Dataset:
    """Returns dataset processed via a Featurizer.

    Args:
      source: Dataset whose unit of data is a valid input to the featurizer.

    Returns:
      Immediate TF dataset, after applying the Featurizer and filtering out
      empty features. Each unit of data will be a `Dict[str, tf.Tensor]`.
    """
    ds = source

    # Apply the featurizer.
    def featurize_fn(s) -> Sequence[tf.Tensor]:
      # `tf.numpy_function` requires output type as Sequences, not dicts. Shapes
      # must also be consistent across all return statements. First output is
      # a bool indicating success.
      try:
        return (True, *self.featurizer.to_features(s).values())  # pytype: disable=bad-return-type  # py311-upgrade
      except Exception as e:  # pylint:disable=broad-exception-caught
        logging.exception('Failed to featurize: %s', e)
        return (False, *self.featurizer.empty_output.values())  # pytype: disable=bad-return-type  # py311-upgrade

    t_out = (tf.bool, *self.featurizer.output_types.values())
    ds = ds.map(
        lambda s: tf.numpy_function(featurize_fn, [s], t_out),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Filter empty (failed) results.
    ds = ds.filter(lambda success, *_: success)

    # NOTE: Downstream tokenization requires inputs w/ known shapes.
    def set_shapes(values: Sequence[tf.Tensor]) -> Sequence[tf.Tensor]:
      for v, s in zip(values, self.featurizer.output_shapes.values()):
        v.set_shape(s)
      return values

    # Drop the success bool and re-provide shape on each value.
    ds = ds.map(lambda _, *v: set_shapes(v))

    # Reconstruct the dict from tuple.
    return ds.map(lambda *v: dict(zip(self.featurizer.output_types.keys(), v)))
