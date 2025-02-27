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
from optformer.common.serialization.numeric import text

from absl.testing import absltest
from absl.testing import parameterized


class ScientificFloatTextSerializerTest(parameterized.TestCase):

  @parameterized.parameters(
      (1.0, '1.0e+00', 1.0),
      (2.1, '2.1e+00', 2.1),
      (123.4, '1.23e+02', 1.23e02),
      (12345, '1.23e+04', 12300),
      (0.1234, '1.23e-01', 0.123),
      (-123.4, '-1.23e+02', -123),
      (-12345, '-1.23e+04', -12300.0),
      (-0.1234, '-1.23e-01', -0.123),
      (0.0, '0.0e+00', 0.0),
      (-0.4e-8, '-4.0e-09', -4e-09),
  )
  def test_default(self, f: float, serialized: str, deserialized: float):
    serializer = text.ScientificFloatTextSerializer()
    self.assertEqual(serializer.to_str(f), serialized)
    self.assertEqual(serializer.from_str(serialized), deserialized)

  @parameterized.parameters(
      (np.nan,),
      (-np.nan,),
  )
  def test_nan(self, f: float):
    serializer = text.ScientificFloatTextSerializer()
    self.assertEqual(serializer.to_str(f), 'nan')
    self.assertTrue(np.isnan(serializer.from_str('nan')))


class ExpandedScientificFloatSerializerTest(parameterized.TestCase):

  @parameterized.parameters(
      (1.0, '[+ 1 10e0]'),
      (2.1, '[+ 2 10e0 1 10e-1]'),
      (123.4, '[+ 1 10e2 2 10e1 3 10e0 4 10e-1]'),
      (12345, '[+ 1 10e4 2 10e3 3 10e2 4 10e1 5 10e0]'),
      (0.00001234, '[+ 1 10e-5 2 10e-6 3 10e-7 4 10e-8]'),
      (-123.456789, '[- 1 10e2 2 10e1 3 10e0 4 10e-1 5 10e-2 7 10e-3]'),
      (-12345, '[- 1 10e4 2 10e3 3 10e2 4 10e1 5 10e0]'),
      (-0.1234, '[- 1 10e-1 2 10e-2 3 10e-3 4 10e-4]'),
      (0.0, '[+ 0 10e0]'),
  )
  def test_default(self, f: float, serialized: str):
    serializer = text.ExpandedScientificFloatSerializer(precision=5)
    self.assertEqual(serializer.to_str(f), serialized)
    self.assertAlmostEqual(serializer.from_str(serialized), f, places=3)


class SimpleFloatTextSerializerTest(parameterized.TestCase):

  @parameterized.parameters(
      (1.0, '1.0', 1.0),
      (2.1, '2.1', 2.1),
      (123.4, '123.4', 123.4),
      (12345, '12345', 12345),
      (0.1234, '0.123', 0.123),
      (-123.4, '-123.4', -123.4),
      (-12345, '-12345', -12345.0),
      (-0.1234, '-0.123', -0.123),
      (0.0, '0.0', 0.0),
      (-0.4e-8, '-0.0', -0.0),
  )
  def test_default(self, f: float, serialized: str, deserialized: float):
    serializer = text.SimpleFloatTextSerializer(precision=3)
    self.assertEqual(serializer.to_str(f), serialized)
    self.assertEqual(serializer.from_str(serialized), deserialized)

  @parameterized.parameters(
      (np.nan,),
      (-np.nan,),
  )
  def test_nan(self, f: float):
    serializer = text.SimpleFloatTextSerializer()
    self.assertEqual(serializer.to_str(f), 'nan')
    self.assertTrue(np.isnan(serializer.from_str('nan')))


class NormalizedFloatSerializerTest(parameterized.TestCase):

  @parameterized.parameters(
      (0.0, 16, '0.00', 0.0),
      (1.0, 16, '0.ff', 0.9960837),
      (0.0, 10, '0.00000', 0.0),
      (1.0, 10, '0.99999', 1.0),
      (0.5, 10, '0.50000', 0.5),
      (0.2, 10, '0.20000', 0.2),
      (1 / 7, 7, '0.10000', 1 / 7),
      (1 / 3, 10, '0.33333', 1 / 3),
  )
  def test_default(
      self, f: float, base: int, serialized: str, deserialized: float
  ):
    serializer = text.NormalizedFloatSerializer(
        precision=len(serialized) - 2, base=base, digit_map='0123456789abcdef'
    )
    self.assertEqual(serializer.to_str(f), serialized)
    self.assertAlmostEqual(
        serializer.from_str(serialized), deserialized, places=4
    )

  @parameterized.parameters((2,), (8,), (10,), (16,), (36,))
  def test_different_bases(self, base: int):
    serializer = text.NormalizedFloatSerializer(
        precision=5, base=base, digit_map='0123456789abcdefghijklmnopqrstuvwxyz'
    )
    f = 0.5
    serialized = serializer.to_str(f)
    deserialized = serializer.from_str(serialized)
    self.assertAlmostEqual(deserialized, f)


if __name__ == '__main__':
  absltest.main()
