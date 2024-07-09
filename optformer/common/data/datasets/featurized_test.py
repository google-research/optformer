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

from optformer.common.data import featurizers
from optformer.common.data.datasets import featurized
import seqio
import tensorflow as tf

from absl.testing import absltest
from absl.testing import parameterized

BAD_STRING = 'bad_string'


class FeaturizedDatasetFnTest(parameterized.TestCase):

  @parameterized.parameters((0,), (1,), (2,))
  def test_e2e(self, rank: int):
    objs = ['hello', 'goodbye']

    self.featurizer = featurizers.IdentityFeaturizer(rank)
    self.dataset_fn = featurized.FeaturizedDatasetFn(self.featurizer)

    ds = tf.data.Dataset.from_tensor_slices(objs)
    ds = self.dataset_fn(ds)

    expected = [self.featurizer.to_features(s) for s in objs]

    seqio.test_utils.assert_dataset(ds, expected)
    for k, v in ds.element_spec.items():
      self.assertSequenceEqual(
          v.shape,
          [None for _ in range(rank)],
          msg=f'{k} must have rank {rank}.',
      )


if __name__ == '__main__':
  absltest.main()
