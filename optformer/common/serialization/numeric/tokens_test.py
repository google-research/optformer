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

from optformer.common.serialization.numeric import tokens
from absl.testing import absltest
from absl.testing import parameterized


class DigitByDigitFloatTokenSerializerTest(parameterized.TestCase):

  @parameterized.parameters(
      (123.4, '<+><1><2><3><4><E-1>', 123.4),
      (12345, '<+><1><2><3><4><E1>', 12340),
      (0.1234, '<+><1><2><3><4><E-4>', 0.1234),
      (-123.4, '<-><1><2><3><4><E-1>', -123.4),
      (-12345, '<-><1><2><3><4><E1>', -12340),
      (-0.1234, '<-><1><2><3><4><E-4>', -0.1234),
      (0.0, '<+><0><0><0><0><E0>', 0.0),
      (-0.0, '<+><0><0><0><0><E0>', 0.0),  # in python, 0.0 == -0.0
      (-0.4e-13, '<-><0><0><0><0><E0>', 0.0),  # note leading negative zero
  )
  def test_serialize(self, f: float, serialized: str, deserialized: float):
    serializer = tokens.DigitByDigitFloatTokenSerializer()
    self.assertEqual(serializer.to_str(f), serialized)
    self.assertAlmostEqual(serializer.from_str(serialized), deserialized)

  @parameterized.parameters((3, 10, 1.0e-8, 9.99e12), (1, 5, 1.0e-5, 9.0e5))
  def test_representation_range(
      self,
      num_digits: int,
      exponent_range: int,
      min_val: float,
      max_val: float,
  ):
    serializer = tokens.DigitByDigitFloatTokenSerializer(
        num_digits=num_digits,
        exponent_range=exponent_range,
    )
    self.assertEqual(serializer._max_abs_val, max_val)
    self.assertEqual(serializer._min_abs_val, min_val)

  @parameterized.parameters(
      (1.0e13, 3, 10, '<+><9><9><9><E10>'),
      (2.0e13, 3, 10, '<+><9><9><9><E10>'),
      (-1.0e13, 3, 10, '<-><9><9><9><E10>'),
      (-2.0e13, 3, 10, '<-><9><9><9><E10>'),
      (9.9e12, 3, 10, '<+><9><9><0><E10>'),
      (-9.9e12, 3, 10, '<-><9><9><0><E10>'),
      (5.0e5, 3, 10, '<+><5><0><0><E3>'),
      (1.1e-8, 3, 10, '<+><1><1><0><E-10>'),
      (0.9e-8, 3, 10, '<+><1><0><0><E-10>'),
      (0.5e-8, 3, 10, '<+><0><0><0><E0>'),
      (0.51e-8, 3, 10, '<+><1><0><0><E-10>'),
      (0.4e-8, 3, 10, '<+><0><0><0><E0>'),
      # rounding up below creats a negative sign for 0
      (-0.4e-8, 3, 10, '<-><0><0><0><E0>'),
      (-0.5e-8, 3, 10, '<-><0><0><0><E0>'),
      (-0.51e-8, 3, 10, '<-><1><0><0><E-10>'),
      (-0.8e-8, 3, 10, '<-><1><0><0><E-10>'),
      (-1.1e-8, 3, 10, '<-><1><1><0><E-10>'),
  )
  def test_tokenization_limit(
      self,
      f: float,
      num_digits: int,
      exponent_range: int,
      serialized: str,
  ):
    serializer = tokens.DigitByDigitFloatTokenSerializer(
        num_digits=num_digits,
        exponent_range=exponent_range,
    )
    self.assertEqual(serializer.to_str(f), serialized)

  def test_all_tokens_used(self):
    serializer = tokens.DigitByDigitFloatTokenSerializer(exponent_range=2)
    out = serializer.all_tokens_used()

    signs = ['<+>', '<->']
    digits = [f'<{i}>' for i in range(0, 10)]
    exponents = ['<E-2>', '<E-1>', '<E0>', '<E1>', '<E2>']
    self.assertEqual(list(out), signs + digits + exponents)


if __name__ == '__main__':
  absltest.main()
