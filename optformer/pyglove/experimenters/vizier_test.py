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

from optformer.pyglove.experimenters import vizier as vizier_exptr_lib
import pyglove as pg
from vizier.benchmarks import experimenters
from absl.testing import absltest


class VizierTest(absltest.TestCase):

  def test_vizier_to_pyglove_e2e(self):
    vizier_experimenter = experimenters.PestControlExperimenter()
    pyglove_experimenter = vizier_exptr_lib.VizierToPyGloveExperimenter(
        vizier_experimenter
    )
    algo = pg.evolution.hill_climb()
    for suggestion, feedback in pg.sample(
        pyglove_experimenter.search_space(), algo, num_examples=100
    ):
      feedback(pyglove_experimenter.evaluate(suggestion))


if __name__ == "__main__":
  absltest.main()
