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

"""Tests for runtime.py."""

from typing import Any, Sequence

from optformer.validation import runtime

from absl.testing import absltest
from absl.testing import parameterized


class RuntimeTest(parameterized.TestCase):

  @parameterized.parameters(
      ([], True),
      ([1], True),
      ([1, 2], False),
      ([1, 1, 1], True),
      (('a', 'a'), True),
      (('a', 'b', 'c'), False),
  )
  def test_all_elements_same(self, x: Sequence[Any], is_same: bool):
    if is_same:
      runtime.assert_all_elements_same(x)
    else:
      with self.assertRaises(ValueError):  # pylint:disable=g-error-prone-assert-raises
        runtime.assert_all_elements_same(x)


if __name__ == '__main__':
  absltest.main()
