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

"""Tests for feedback serializers."""

from optformer.pyglove import types
from optformer.pyglove.serialization import feedback
import pyglove as pg
from absl.testing import absltest


def _make_trial(x: int, objective: float) -> types.PyGloveTrial:
  return types.PyGloveTrial(suggestion=x, objective=objective)


class QuantizedMeasurementSerializerTest(absltest.TestCase):

  def test_quantized_measurement_serializer(self):
    search_space = pg.one_of(candidates=[0, 1, 2, 3])
    trials = [
        _make_trial(1, 0.0),
        _make_trial(2, 4.0),
        _make_trial(3, 10.0),
    ]
    study = types.PyGloveStudy(search_space=search_space, trials=trials)
    serializer = feedback.QuantizedMeasurementSerializer()
    serialized = serializer.to_str(study)
    self.assertEqual(serialized, '<0><400><999>')


if __name__ == '__main__':
  absltest.main()
