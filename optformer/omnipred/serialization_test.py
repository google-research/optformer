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

from optformer.omnipred import serialization
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import grid
from vizier.testing import test_studies
from absl.testing import absltest


class OmniPredSerializersTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self.problem = vz.ProblemStatement(
        search_space=test_studies.flat_space_with_all_types(),
        metric_information=[
            vz.MetricInformation("x", goal=vz.ObjectiveMetricGoal.MAXIMIZE)
        ],
    )
    designer = grid.GridSearchDesigner(self.problem.search_space)
    self.trials = [t.to_trial() for t in designer.suggest(1)]
    self.trials[0].complete(vz.Measurement(metrics={"x": 10.0}))

    self.study = vz.ProblemAndTrials(self.problem, self.trials)

  def test_inputs_serializer(self):
    serializer = serialization.OmniPredInputsSerializer()
    out = serializer.to_str(self.study)
    expected = """{suggestion:{lineardouble:-1,logdouble:0.0001,integer:-2,categorical:"a",boolean:"False",discrete_double:-0.5,discrete_logdouble:1e-05,discrete_int:-1}},{objective:"x",problem_metadata:"{}"}"""
    self.assertEqual(out, expected)

  def test_targets_serializer(self):
    serializer = serialization.OmniPredTargetsSerializer()
    out = serializer.to_str(self.study)
    self.assertEqual(out, "<+><1><0><0><0><E-2>")


if __name__ == "__main__":
  absltest.main()
