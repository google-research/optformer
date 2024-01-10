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

import jax
from jax.experimental import checkify
import jax.numpy as jnp
import numpy as np
from optformer.common.inference import sequence_utils
from optformer.validation import checkify as _checkify
from absl.testing import absltest
from absl.testing import parameterized


class SequenceUtilsTest(parameterized.TestCase):

  def test_count_not_from(self):
    x = jnp.array([[0, 4, 5, 1, 0, 0], [0, 5, 7, 8, 1, 0]])
    expected = [2, 3]

    jit_count_not_from = jax.jit(sequence_utils.count_not_from)
    out = jit_count_not_from(x, not_from=(0, 1))
    np.testing.assert_array_equal(expected, out)

  def test_shift_right(self):
    x = jnp.array([[1, 2, 3, 0, 0], [0, 0, 3, 4, 5]])
    expected = [[42, 100, 1, 2, 3], [42, 100, 0, 0, 3]]

    jit_shift_right = jax.jit(sequence_utils.shift_right)
    out = jit_shift_right(x, insert_left=(42, 100))
    np.testing.assert_array_equal(expected, out)

  def test_find(self):
    x = jnp.array([[0, 0, 42, 42, 42], [42, 42, 0, 0, 0], [0, 0, 0, 0, 0]])
    expected = [2, 0, -1]

    jit_find = jax.jit(sequence_utils.find)
    out = jit_find(x, elem=42, not_found=-1)
    np.testing.assert_array_equal(expected, out)

  def test_rfind(self):
    x = jnp.array([[42, 42, 42, 0, 0], [0, 0, 0, 42, 42], [0, 0, 0, 0, 0]])
    expected = [2, 4, -1]

    jit_rfind = jax.jit(sequence_utils.rfind)
    out = jit_rfind(x, elem=42, not_found=-1)
    np.testing.assert_array_equal(expected, out)

  @absltest.skip("This might require dynamic_update?")
  def test_append_to_output(self):
    x = jnp.array([[1, 2, 0, 0], [1, 2, 3, 0], [0, 0, 0, 0]])
    expected = jnp.array([[1, 2, 42, 43], [1, 2, 3, 0], [42, 43, 0, 0]])

    jit_append_to_output = jax.jit(sequence_utils.append_to_output)
    out = jit_append_to_output(x, elems=[42, 43])
    np.testing.assert_array_equal(expected, out)

  def test_rpad(self):
    x = jnp.array([[1, 2], [3, 4]])
    target = jnp.ones([3, 3])
    expected = jnp.array([[1, 2, 0], [3, 4, 0]])

    jit_rpad = jax.jit(sequence_utils.rpad)
    out = jit_rpad(x, target)
    np.testing.assert_array_equal(expected, out)

  def test_value_mask(self):
    x = jnp.array([[1, 2, 3], [4, 5, 6]])

    jit_value_mask = jax.jit(sequence_utils.value_mask)
    out = jit_value_mask(x, masked_values=[2, 3])
    expected = jnp.array([[1, 0, 0], [1, 1, 1]])
    np.testing.assert_array_equal(expected, out)

  def test_slice_update(self):
    x = jnp.array([[1, 2, 3], [4, 5, 6]])
    jit_slice_update = jax.jit(sequence_utils.slice_update)

    out = jit_slice_update(x, start=0, elems=[42, 43])
    expected = jnp.array([[42, 43, 3], [42, 43, 6]])
    np.testing.assert_array_equal(expected, out)


class ReduceEqTest(parameterized.TestCase):
  """Tests for `reduce_eq`."""

  def setUp(self):
    super().setUp()
    self._reduce_eq = _checkify.check_and_jit(sequence_utils.reduce_eq)
    _checkify.enable_checks(True)

  def tearDown(self):
    super().tearDown()
    _checkify.enable_checks(False)

  def test_raise_error(self):
    x = jnp.array([1, 1, 1, 1, 1, 2])
    with self.assertRaises(checkify.JaxRuntimeError):
      _ = self._reduce_eq(x)

  def test_good_array(self):
    x = jnp.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    out = self._reduce_eq(x)
    expected = jnp.array([1, 2, 3])
    np.testing.assert_array_equal(expected, out)


class BetweenMaskTest(absltest.TestCase):

  def test_good_input(self):
    x = jnp.array([[-1, 42, 42, 1, -1, 42, 1], [42, 42, 42, 42, 42, 42, 42]])
    jit_between_mask = _checkify.check_and_jit(sequence_utils.between_mask)

    out = jit_between_mask(x, left=-1, right=1)
    expected = jnp.array([[0, 1, 1, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0]])
    np.testing.assert_array_equal(expected, out)

  def test_checkify_imbalanced_delimiters(self):
    _checkify.enable_checks(True)
    jit_between_mask = _checkify.check_and_jit(sequence_utils.between_mask)
    x = jnp.array([[-1, 42, 42, 1, -1, 42, 0], [42, 42, 42, 42, 42, 42, 42]])
    with self.assertRaises(checkify.JaxRuntimeError):
      _ = jit_between_mask(x, left=-1, right=1)
    _checkify.enable_checks(False)

  def test_checkify_misordered_delimiters(self):
    _checkify.enable_checks(True)
    jit_between_mask = _checkify.check_and_jit(sequence_utils.between_mask)
    x = jnp.array([[-1, 42, 42, 1, 1, 42, -1], [42, 42, 42, 42, 42, 42, 42]])
    with self.assertRaises(checkify.JaxRuntimeError):
      _ = jit_between_mask(x, left=-1, right=1)
    _checkify.enable_checks(False)


def _create_readable_array(shape) -> jax.Array:
  """Returns an array where array[i,j,k] = ijk."""
  x = np.zeros(shape)
  for e, k in enumerate(reversed(shape)):
    if k >= 10:
      raise ValueError("Cannot handle digits >= 10.")
    for i in range(k):
      index = [slice(None)] * len(x.shape)
      index[x.ndim - e - 1] = slice(i, i + 1)
      x[tuple(index)] += (i) * 10**e

  return jnp.asarray(x, dtype=jnp.int32)


class DynamicSliceBroadcastTest(parameterized.TestCase):

  def test_basic_1d(self):
    actual = sequence_utils.dynamic_slice_broadcast(
        jnp.array([0, 1, 2, 3]), jnp.array(1), 3
    )
    np.testing.assert_array_equal([1, 2, 3], actual)

  def test_x3d_index1d(self):
    x = _create_readable_array([2, 3, 5])
    fn = jax.jit(
        sequence_utils.dynamic_slice_broadcast,
        static_argnums=[2],
    )
    actual = fn(x, jnp.array([0, 1, 1]), 2)
    expected = jnp.array(
        [
            # 0, 1, 1 -> we should see [0, 1], [1, 2], [1, 2]
            # as the last digit.
            [[0, 1], [1, 2], [1, 2]],
            [[0, 1], [1, 2], [1, 2]],
        ],
    )
    self.assertSequenceEqual(actual.shape, (2, 3, 2))
    np.testing.assert_array_equal(expected, actual % 10)

  def test_x4d_index1d(self):
    x = _create_readable_array([5, 2, 3, 4])
    fn = jax.jit(
        sequence_utils.dynamic_slice_broadcast,
        static_argnums=[2],
    )
    actual = fn(x, jnp.array([0, 1, 1]), 2)
    self.assertSequenceEqual(actual.shape, (5, 2, 3, 2))
    expected = jnp.array(
        [
            # 0, 1, 1 -> we should see [0, 1], [1, 2], [1, 2] as
            # the last digit.
            [[[0, 1], [1, 2], [1, 2]], [[0, 1], [1, 2], [1, 2]]],
            [[[0, 1], [1, 2], [1, 2]], [[0, 1], [1, 2], [1, 2]]],
            [[[0, 1], [1, 2], [1, 2]], [[0, 1], [1, 2], [1, 2]]],
            [[[0, 1], [1, 2], [1, 2]], [[0, 1], [1, 2], [1, 2]]],
            [[[0, 1], [1, 2], [1, 2]], [[0, 1], [1, 2], [1, 2]]],
        ],
    )
    np.testing.assert_array_equal(expected, actual % 10)

  def test_x4d_index2d(self):
    x = _create_readable_array([5, 2, 3, 4])
    fn = jax.jit(
        sequence_utils.dynamic_slice_broadcast,
        static_argnums=[2],
    )
    actual = fn(x, jnp.array([[0, 1, 1], [1, 2, 1]]), 2)
    self.assertSequenceEqual(actual.shape, (5, 2, 3, 2))
    expected = jnp.array([
        [[[0, 1], [1, 2], [1, 2]], [[1, 2], [2, 3], [1, 2]]],
        [[[0, 1], [1, 2], [1, 2]], [[1, 2], [2, 3], [1, 2]]],
        [[[0, 1], [1, 2], [1, 2]], [[1, 2], [2, 3], [1, 2]]],
        [[[0, 1], [1, 2], [1, 2]], [[1, 2], [2, 3], [1, 2]]],
        [[[0, 1], [1, 2], [1, 2]], [[1, 2], [2, 3], [1, 2]]],
    ])
    np.testing.assert_array_equal(expected, actual % 10)


if __name__ == "__main__":
  absltest.main()
