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

import numpy as np
from optformer.original.numeric import nan_handling
from absl.testing import absltest
from absl.testing import parameterized


class ObjectiveImputerTest(parameterized.TestCase):

  @parameterized.parameters(
      (np.array([0.0, np.nan, 10.0]), np.array([0.0, -10.0, 10.0]), True),
      (np.array([0.0, np.nan, 0.0]), np.array([0.0, 0.0, 0.0]), True),
      (np.array([0.0, np.nan, 10.0]), np.array([0.0, 20.0, 10.0]), False),
      (np.array([0.0, np.nan, 0.0]), np.array([0.0, 0.0, 0.0]), True),
  )
  def test_output(self, x, y, maximize):
    nan_handler = nan_handling.ObjectiveImputer(
        penalty_multiplier=1.0, maximize=maximize
    )
    self.assertTrue(np.array_equal(nan_handler.map(x), y))

  def test_error(self):
    nan_handler = nan_handling.ObjectiveImputer()
    with self.assertRaises(ValueError):
      nan_handler.map(np.array([np.nan, np.nan]))

    with self.assertRaises(ValueError):
      nan_handler.map(np.array(1.0))

    with self.assertRaises(NotImplementedError):
      nan_handler.unmap(np.array([1.0, 2.0]))


if __name__ == "__main__":
  absltest.main()
