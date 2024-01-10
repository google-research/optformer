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

from optformer.common.data.processors import masking
import tensorflow as tf
from absl.testing import absltest
from absl.testing import parameterized


class ValueMaskTest(tf.test.TestCase):

  def test_basic(self):
    tensor = tf.constant([1, 2, 3])
    preprocessor = masking.ValueMask(masked_values=[2, 3])
    out = preprocessor(tensor)
    expected = tf.constant([1, 0, 0])
    self.assertAllEqual(out, expected)


class BetweenDelimitersMaskTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.preprocessor = masking.BetweenDelimitersMask(left=-1, right=1)

  @parameterized.parameters(
      (tf.constant([-1, 2, 2, 1, -1, 1]), tf.constant([0, 1, 1, 0, 0, 0])),
      (tf.constant([2, 2, 2, 2, 2, 2]), tf.constant([0, 0, 0, 0, 0, 0])),
  )
  def test_basic(self, tensor: tf.Tensor, expected: tf.Tensor):
    out = self.preprocessor(tensor)
    self.assertAllEqual(out, expected)

  @parameterized.parameters(
      (tf.constant([-1]),),
      (tf.constant([1, -1]),),
      (tf.constant([1]),),
  )
  def test_error(self, tensor: tf.Tensor):
    """Input must have immediately-matched left-right delimiters."""
    with self.assertRaises(tf.errors.OpError):
      self.preprocessor(tensor)


if __name__ == '__main__':
  absltest.main()
