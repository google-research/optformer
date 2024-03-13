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
from optformer.original.numeric import quantization
from absl.testing import absltest
from absl.testing import parameterized


class NormalizedQuantizerTest(parameterized.TestCase):

  def setUp(self):
    self.quantizer = quantization.NormalizedQuantizer(
        num_bins=1000, dequantization_shift=0.5
    )
    super().setUp()

  @parameterized.parameters((0.0, 0), (1.0, 999))
  def test_quantization(self, x: float, output: int):
    self.assertEqual(self.quantizer.map(x), output)

  @parameterized.parameters((0, 0.0005), (999, 0.9995))
  def test_dequantization(self, y: int, output: float):
    self.assertEqual(self.quantizer.unmap(y), output)

  @parameterized.parameters(
      (0.0, False), (0.2005, True), (0.5, False), (1.0, False)
  )
  def test_forward_non_reversibility(self, x: float, reversible: bool):
    y = self.quantizer.map(x)
    if reversible:
      self.assertEqual(x, self.quantizer.unmap(y))
    else:
      self.assertNotEqual(x, self.quantizer.unmap(y))

  @parameterized.parameters((0,), (500,), (999,))
  def test_backward_reversibility(self, y: int):
    x = self.quantizer.unmap(y)
    self.assertEqual(y, self.quantizer.map(x))

  def test_array_input(self):
    ones_array = np.ones(shape=10)
    x = 0.3
    array_x = x * ones_array

    out = self.quantizer.map(x)
    array_out = self.quantizer.map(array_x)
    self.assertTrue(np.all(array_out == out))

    y = 51
    array_y = y * np.ones(shape=10, dtype=np.int32)

    out = self.quantizer.unmap(y)
    array_out = self.quantizer.unmap(array_y)
    self.assertTrue(np.all(array_out == out))

  @parameterized.parameters((1.0000001,), (-0.2,))
  def test_quantize_domain_error(self, x: float):
    with self.assertRaises(ValueError):
      self.quantizer.map(x)

  @parameterized.parameters((1000,), (-1,))
  def test_dequantize_domain_error(self, y: int):
    with self.assertRaises(ValueError):
      self.quantizer.unmap(y)


if __name__ == "__main__":
  absltest.main()
