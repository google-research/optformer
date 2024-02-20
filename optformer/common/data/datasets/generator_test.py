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
from optformer.common.data.datasets import generator
import seqio

from absl.testing import absltest


class GeneratorDatasetFnTest(absltest.TestCase):

  def test_buffer(self):
    buffer = ['hello', 'goodbye']
    gen = (s for s in buffer)

    featurizer = featurizers.IdentityFeaturizer()
    dataset_fn = generator.GeneratorDatasetFn(featurizer=featurizer)

    dataset = dataset_fn(gen)
    expected = [
        {'inputs': b'hello', 'targets': b'hello'},
        {'inputs': b'goodbye', 'targets': b'goodbye'},
    ]
    seqio.test_utils.assert_dataset(dataset, expected)


if __name__ == '__main__':
  absltest.main()
