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

from optformer.vizier.serialization import trial as trial_lib
from vizier import pyvizier as vz
from absl.testing import absltest


class DefaultParametersSerializerTest(absltest.TestCase):

  def test_e2e(self):
    serializer = trial_lib.DefaultParametersSerializer()
    parameters = vz.ParameterDict(a=3, b=5.0, c="hello")
    s = serializer.to_str(parameters)

    self.assertEqual(s, """{"a": 3, "b": 5.0, "c": "hello"}""")
    self.assertEqual(serializer.from_str(s), parameters)


if __name__ == "__main__":
  absltest.main()
