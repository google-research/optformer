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

from typing import Any

from optformer.vizier.data import filters
from vizier.testing import test_studies

from absl.testing import absltest
from absl.testing import parameterized


# pyformat: disable
class ValidateParametersTest(parameterized.TestCase):

  @parameterized.parameters(
      ({'model_type': 'linear', 'learning_rate': 1.0,}, True),
      ({'model_type': 'linear', 'learning_rate': 1.0, 'badvalue': 'bad'}, False),
      ({'model_type': 'linear', 'learning_rate': 1.0, 'optimizer_type': 'adam'}, False),
      ({'model_type': 'dnn', 'learning_rate': 1.0, 'optimizer_type': 'evolution'}, True),
      ({'model_type': 'dnn', 'learning_rate': 1.0, 'optimizer_type': 'evolution', 'use_special_logic': False}, False),
  )
  def test_conditional_contains(self, parameters: dict[str, Any], result: bool):
    space = test_studies.conditional_automl_space()
    self.assertEqual(filters._validate_parameters(parameters, space), result)
# pyformat: enable

if __name__ == '__main__':
  absltest.main()
