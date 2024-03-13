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

from optformer.original.numeric import scaling
from absl.testing import absltest
from absl.testing import parameterized


class UniformIntervalSamplerTest(parameterized.TestCase):

  def test_smoke(self):
    random_interval = scaling.UniformIntervalSampler((0.5, 0.6))()
    self.assertLen(random_interval, 2)

    self.assertGreaterEqual(random_interval[0], 0.0)
    self.assertLessEqual(random_interval[0], 1.0)

    self.assertGreaterEqual(random_interval[1], 0.0)
    self.assertLessEqual(random_interval[1], 1.0)

    interval_length = random_interval[1] - random_interval[0]
    self.assertGreaterEqual(interval_length, 0.5)
    self.assertLessEqual(interval_length, 0.6)


class LinearIntervalScalerTest(parameterized.TestCase):

  def setUp(self):
    self.linear_scaler = scaling.LinearIntervalScaler(
        source_interval=(0.0, 1.0), target_interval=(10.0, 20.0)
    )
    super().setUp()

  def test_output(self):
    self.assertEqual(self.linear_scaler.map(0.0), 10.0)
    self.assertEqual(self.linear_scaler.map(0.5), 15.0)
    self.assertEqual(self.linear_scaler.map(1.0), 20.0)

    self.assertEqual(self.linear_scaler.unmap(10.0), 0.0)
    self.assertEqual(self.linear_scaler.unmap(15.0), 0.5)
    self.assertEqual(self.linear_scaler.unmap(20.0), 1.0)

  def test_array_input(self):
    x = 0.3
    array_x = x * np.ones(10)
    x_out = self.linear_scaler.map(x)
    array_x_out = self.linear_scaler.map(array_x)
    self.assertTrue(np.all(array_x_out == x_out))

    y = 16.3
    array_y = y * np.ones(10)
    y_out = self.linear_scaler.unmap(y)
    array_y_out = self.linear_scaler.unmap(array_y)
    self.assertTrue(np.all(array_y_out == y_out))

  def test_reversibility(self):
    x = 0.5
    y = self.linear_scaler.map(x)
    self.assertEqual(x, self.linear_scaler.unmap(y))

  @parameterized.parameters((-1.0,), (1.2,))
  def test_map_error(self, x: float):
    with self.assertRaises(ValueError):
      self.linear_scaler.map(x)

  @parameterized.parameters((9.0,), (21.1,))
  def test_unmap_error(self, y: float):
    with self.assertRaises(ValueError):
      self.linear_scaler.unmap(y)

  def test_bad_source_bounds(self):
    with self.assertRaises(ValueError):
      scaling.LinearIntervalScaler(
          source_interval=(0.0, -1.0), target_interval=(10.0, 20.0)
      )

  def test_bad_target_bounds(self):
    with self.assertRaises(ValueError):
      scaling.LinearIntervalScaler(
          source_interval=(0.0, 1.0), target_interval=(21.0, 20.0)
      )

  def test_equal_bounds(self):
    scaler = scaling.LinearIntervalScaler(
        source_interval=(0.0, 0.0), target_interval=(10.0, 20.0)
    )
    self.assertEqual(scaler.map(0.0), 15.0)
    self.assertEqual(scaler.unmap(17.3), 0.0)

    scaler = scaling.LinearIntervalScaler(
        source_interval=(0.0, 1.0), target_interval=(21.0, 21.0)
    )
    self.assertEqual(scaler.unmap(21.0), 0.5)
    self.assertEqual(scaler.map(0.0), 21.0)

    scaler = scaling.LinearIntervalScaler(
        source_interval=(0.0, 0.0), target_interval=(10.0, 10.0)
    )
    self.assertEqual(scaler.map(0.0), 10.0)
    self.assertEqual(scaler.unmap(10.0), 0.0)


if __name__ == "__main__":
  absltest.main()
