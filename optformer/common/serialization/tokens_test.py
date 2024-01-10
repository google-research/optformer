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

from typing import Any, Sequence

from optformer.common.serialization import tokens

from absl.testing import absltest
from absl.testing import parameterized


class IntegerTokenTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.serializer = tokens.IntegerTokenSerializer()

  @parameterized.parameters(
      (42, '<42>'),
      (0, '<0>'),
      (-3, '<-3>'),
  )
  def test_serialize(self, x: int, expected: str):
    self.assertEqual(self.serializer.to_str(x), expected)

  @parameterized.parameters(
      ('<42>', 42),
      ('<0>', 0),
      ('<-3>', -3),
  )
  def test_deserialize(self, y: str, expected: int):
    self.assertEqual(self.serializer.from_str(y), expected)

  @parameterized.parameters(
      ('<42',),
      ('42>',),
      ('<hello>',),
  )
  def test_deserialize_error(self, y: str):
    with self.assertRaises(ValueError):
      self.serializer.from_str(y)

  @parameterized.parameters(
      (242,),
      (0,),
      (-356,),
  )
  def test_reversibility(self, x):
    y = self.serializer.to_str(x)
    self.assertEqual(self.serializer.from_str(y), x)


class StringTokenTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.serializer = tokens.StringTokenSerializer()

  @parameterized.parameters(
      ('hi', '<hi>'),
      ('', '<>'),
      ('3', '<3>'),
  )
  def test_serialize(self, x: str, expected: str):
    self.assertEqual(self.serializer.to_str(x), expected)

  @parameterized.parameters(
      ('<im spaced>', 'im spaced'),
      ('<>', ''),
      ('<-3>', '-3'),
  )
  def test_deserialize(self, y: str, expected: int):
    self.assertEqual(self.serializer.from_str(y), expected)

  @parameterized.parameters(
      ('<hi',),
      ('hi>',),
      ('hi',),
  )
  def test_deserialize_error(self, y: str):
    with self.assertRaises(ValueError):
      self.serializer.from_str(y)

  @parameterized.parameters(
      ('<hi>',),
      ('<>',),
      ('<-3>',),
  )
  def test_reversibility(self, x):
    y = self.serializer.to_str(x)
    self.assertEqual(self.serializer.from_str(y), x)


class UnitSequenceTokenTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.serializer = tokens.UnitSequenceTokenSerializer()

  @parameterized.parameters(
      ([0, 42, 'm'], '<0><42><m>'),
      ([5], '<5>'),
      (['m'], '<m>'),
      ([42], '<42>'),
      ([42, 1], '<42><1>'),
      ([], ''),
  )
  def test_serialization(self, obj: Sequence[Any], output: str):
    self.assertEqual(self.serializer.to_str(obj), output)

  @parameterized.parameters(
      ('<0><42><m>', [0, 42, 'm']),
      ('<-5>', [-5]),
      ('<m>', ['m']),
      ('<42>', [42]),
      ('<42><1>', [42, 1]),
      ('', []),
  )
  def test_deserialization(self, s: str, obj: Sequence[Any]):
    self.assertEqual(self.serializer.from_str(s), obj)

  @parameterized.parameters(
      ([0, 242, 'm', -1],),
      ([0],),
      ([],),
  )
  def test_reversibility(self, x: Sequence[Any]):
    y = self.serializer.to_str(x)
    self.assertEqual(self.serializer.from_str(y), x)


if __name__ == '__main__':
  absltest.main()
