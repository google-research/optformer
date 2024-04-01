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

from optformer.pyglove.data.generators import algorithms
from absl.testing import absltest
from absl.testing import parameterized


class AlgorithmFactoriesTest(parameterized.TestCase):

  @parameterized.parameters(
      (algorithms.EvolutionaryAlgorithmFactory,),
  )
  def test_e2e(self, alg_factory_cls):
    alg_factory = alg_factory_cls()
    for seed in range(10):
      alg_factory(seed)


if __name__ == "__main__":
  absltest.main()
