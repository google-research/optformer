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

from optformer.common.data.processors import partitioning
import tensorflow as tf
from absl.testing import absltest


class PartitionerTest(absltest.TestCase):

  def test_e2e(self):
    partitioner = partitioning.Partitioner(
        split_ratios={'train': 0.8, 'validation': 0.1, 'test': 0.1}
    )

    features = tf.constant(list(range(10)))
    partitioned_features = partitioner(features)

    self.assertLen(partitioned_features['train'], 8)
    self.assertLen(partitioned_features['validation'], 1)
    self.assertLen(partitioned_features['test'], 1)


if __name__ == '__main__':
  absltest.main()
