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

from typing import Dict

from optformer.common.data import featurizers
from optformer.common.data.datasets import generator
import seqio
import tensorflow as tf

from absl.testing import absltest


class DoNothingFeaturizer(featurizers.Featurizer[str]):

  def to_features(self, obj: str) -> Dict[str, tf.Tensor]:
    return {'key': tf.constant(obj, dtype=tf.string)}

  @property
  def output_types(self) -> Dict[str, tf.DType]:
    return {'key': tf.string}

  @property
  def output_shapes(self) -> Dict[str, tf.TensorShape]:
    return {'key': tf.TensorShape([])}

  @property
  def empty_output(self) -> Dict[str, tf.Tensor]:
    return {'key': tf.constant('', dtype=tf.string)}


class GeneratorDatasetFnTest(absltest.TestCase):

  def test_buffer(self):
    buffer = ['hello', 'goodbye']
    gen = (s for s in buffer)

    featurizer = DoNothingFeaturizer()
    dataset_fn = generator.GeneratorDatasetFn(featurizer=featurizer)

    dataset = dataset_fn(gen)
    expected = [{'key': b'hello'}, {'key': b'goodbye'}]
    seqio.test_utils.assert_dataset(dataset, expected)


if __name__ == '__main__':
  absltest.main()
