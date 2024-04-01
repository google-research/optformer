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

from optformer.pyglove.experimenters import nested
from optformer.pyglove.experimenters import permutation
import pyglove as pg
from absl.testing import absltest


class SwitchTest(absltest.TestCase):

  def test_switch_e2e(self):
    exp1 = permutation.LOPExperimenter()
    exp2 = permutation.TSPExperimenter()
    switch_exp = nested.SwitchExperimenter(experimenters=(exp1, exp2))

    random_iter = pg.random_sample(switch_exp.search_space())
    suggestion = next(random_iter)
    switch_exp.evaluate(suggestion)

  def test_multi_switch_e2e(self):
    exp1 = permutation.FSSExperimenter()
    exp2 = permutation.LOPExperimenter()
    exp3 = permutation.TSPExperimenter()

    multi_switch_exp = nested.MultiSwitchExperimenter(
        experimenters=(exp1, exp2, exp3), k=2
    )

    random_iter = pg.random_sample(multi_switch_exp.search_space())
    suggestion = next(random_iter)
    multi_switch_exp.evaluate(suggestion)


if __name__ == "__main__":
  absltest.main()
