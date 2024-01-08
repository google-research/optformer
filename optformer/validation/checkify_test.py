# Copyright 2022 Google LLC.
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

from optformer.validation import checkify as _checkify
from absl.testing import absltest


class CheckifyTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.trace_count = 0

  def test_traces_only_once(self):
    def f(x):
      self.trace_count += 1
      return x + 1

    _checkify.check_and_jit(f)(0.1)
    _checkify.check_and_jit(f)(0.1)
    self.assertEqual(1, self.trace_count)


if __name__ == "__main__":
  absltest.main()
