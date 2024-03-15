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

from optformer.vizier.serialization import problem as problem_lib
from vizier import pyvizier as vz
from vizier.testing import test_studies
from absl.testing import absltest


class SerializersTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.problem = vz.ProblemStatement(
        search_space=test_studies.flat_space_with_all_types(),
        metric_information=[
            vz.MetricInformation('x1', goal=vz.ObjectiveMetricGoal.MAXIMIZE)
        ],
    )
    self.problem.metadata['team'] = 'computer_vision'

  def test_search_space_output(self):
    full_serializer = problem_lib.SearchSpaceSerializer(
        include_param_name=True,
        include_scale_type=True,
        include_bounds_and_feasibles=True,
    )
    out = full_serializer.to_str(self.problem.search_space)
    expected = """{P:"DOUBLE",N:"lineardouble",S:"LINEAR",m:-1,M:2}*{P:"DOUBLE",N:"logdouble",S:"LOG",m:0.0001,M:100}*{P:"INTEGER",N:"integer",S:"None",m:-2,M:2}*{P:"CATEGORICAL",N:"categorical",L:3,C:["a","aa","aaa"]}*{P:"CATEGORICAL",N:"boolean",L:2,C:["False","True"]}*{P:"DISCRETE",N:"discrete_double",S:"LINEAR",L:3,F:[-0.5,1,1.2]}*{P:"DISCRETE",N:"discrete_logdouble",S:"LOG",L:3,F:[1e-05,0.01,0.1]}*{P:"DISCRETE",N:"discrete_int",S:"LINEAR",L:3,F:[-1,1,2]}"""
    self.assertEqual(out, expected)

    minimal_serializer = problem_lib.SearchSpaceSerializer(
        include_param_name=False,
        include_scale_type=False,
        include_bounds_and_feasibles=False,
    )
    out = minimal_serializer.to_str(self.problem.search_space)
    expected = """{P:"DOUBLE"}*{P:"DOUBLE"}*{P:"INTEGER"}*{P:"CATEGORICAL",L:3}*{P:"CATEGORICAL",L:2}*{P:"DISCRETE",L:3}*{P:"DISCRETE",L:3}*{P:"DISCRETE",L:3}"""
    self.assertEqual(out, expected)

  def test_conditional_search_space_output(self):
    full_serializer = problem_lib.SearchSpaceSerializer(
        include_param_name=True,
        include_scale_type=True,
        include_bounds_and_feasibles=True,
    )
    out = full_serializer.to_str(test_studies.conditional_automl_space())
    expected = """{P:"CATEGORICAL",N:"model_type",L:2,C:["dnn","linear"],s:{linear:"{P:"DOUBLE",N:"learning_rate",S:"LOG",m:0.1,M:1}"}}"""
    self.assertEqual(out, expected)

  def test_metrics_config_output(self):
    full_serializer = problem_lib.MetricsConfigSerializer(
        include_metric_name=True
    )
    out = full_serializer.to_str(self.problem.metric_information)
    expected = """{O:"x1",G:"MAXIMIZE"}"""
    self.assertEqual(out, expected)

    minimal_serializer = problem_lib.MetricsConfigSerializer(
        include_metric_name=False
    )
    out = minimal_serializer.to_str(self.problem.metric_information)
    expected = """{G:"MAXIMIZE"}"""
    self.assertEqual(out, expected)


if __name__ == '__main__':
  absltest.main()
