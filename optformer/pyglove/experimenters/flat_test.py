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

from optformer.pyglove.experimenters import binomial
from optformer.pyglove.experimenters import permutation
import pyglove as pg
from absl.testing import absltest
from absl.testing import parameterized


class FlatExperimentersTest(parameterized.TestCase):

  @parameterized.parameters(
      (binomial.CoverageExperimenter,),
      (binomial.LogDeterminantExperimenter,),
      (binomial.ModularExperimenter,),
      (permutation.FSSExperimenter,),
      (permutation.LOPExperimenter,),
      (permutation.QAPExperimenter,),
      (permutation.QueenPlacementExperimenter,),
      (permutation.TSPExperimenter,),
  )
  def test_e2e(self, exptr_ctr):
    experimenter = exptr_ctr()
    algo = pg.evolution.hill_climb()
    for suggestion, feedback in pg.sample(
        experimenter.search_space(), algo, num_examples=100
    ):
      feedback(experimenter.evaluate(suggestion))

  @parameterized.parameters(
      (binomial.CoverageExperimenter,),
      (binomial.LogDeterminantExperimenter,),
      (binomial.ModularExperimenter,),
      (permutation.FSSExperimenter,),
      (permutation.LOPExperimenter,),
      (permutation.QAPExperimenter,),
      (permutation.QueenPlacementExperimenter,),
      (permutation.TSPExperimenter,),
  )
  def test_seeding(self, exptr_ctr):
    experimenter = exptr_ctr()
    random_iter = pg.random_sample(experimenter.search_space())

    for seed in range(5):
      suggestion = next(random_iter)

      experimenter_1 = exptr_ctr(seed=seed)
      experimenter_2 = exptr_ctr(seed=seed)
      self.assertEqual(
          experimenter_1.evaluate(suggestion),
          experimenter_2.evaluate(suggestion),
      )


if __name__ == "__main__":
  absltest.main()
