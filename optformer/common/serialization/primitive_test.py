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
from optformer.common.serialization import primitive
from absl.testing import absltest
from absl.testing import parameterized


class PrimitiveSerializerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.slzr_default = primitive.PrimitiveSerializer()
    self.slzr_no_brackets = primitive.PrimitiveSerializer(
        include_sequence_brackets=False, include_dict_brackets=False
    )
    self.slzr_w_brackets = primitive.PrimitiveSerializer(
        include_sequence_brackets=True, include_dict_brackets=True
    )

  def test_dict_inputs(self):
    v = {
        'N': 'lr',
        'P': 2,
        'm': 3.4,
        'M': 5.0,
        'C': ['ABC', 'DEF'],
        'F': [1.2, 2.0, -1.1],
    }
    output = '{N:"lr",P:2,m:3.4,M:5,C:["ABC","DEF"],F:[1.2,2,-1.1]}'
    self.assertEqual(self.slzr_default.to_str(v), output)

  @parameterized.parameters(
      (2, '2'), (1.2345, '1.23'), ('hi', '"hi"'), (True, 'True'), (None, 'None')
  )
  def test_basic_inputs(self, v: primitive.PrimitiveType, output: str):
    self.assertEqual(self.slzr_no_brackets.to_str(v), output)

  @parameterized.parameters(
      (['hi', 2], '"hi",2'),
      (('hi', 2), '"hi",2'),
      ({'hi': 2}, 'hi:2'),
      (np.array([1, 2]), '1,2'),
      (np.array([1.5, 2.5]), '1.5,2.5'),
  )
  def test_containers_no_brackets(
      self, v: primitive.PrimitiveType, output: str
  ):
    self.assertEqual(self.slzr_no_brackets.to_str(v), output)

  @parameterized.parameters(
      (['hi', 2], '["hi",2]'),
      (('hi', 2), '["hi",2]'),
      ({'hi': 2}, '{hi:2}'),
      (np.array([1, 2]), '[1,2]'),
      (np.array([1.5, 2.5]), '[1.5,2.5]'),
  )
  def test_containers_with_brackets(
      self, v: primitive.PrimitiveType, output: str
  ):
    self.assertEqual(self.slzr_w_brackets.to_str(v), output)


if __name__ == '__main__':
  absltest.main()
