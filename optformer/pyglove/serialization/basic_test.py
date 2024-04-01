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

from optformer.pyglove import types
from optformer.pyglove.serialization import basic
import pyglove as pg
from absl.testing import absltest


def _generate_study() -> types.PyGloveStudy:
  search_space = pg.one_of(candidates=[0, 1, 2, 3])
  suggestion = next(pg.random_sample(search_space, seed=0))
  objective = 1.0

  trial = types.PyGloveTrial(suggestion=suggestion, objective=objective)
  return types.PyGloveStudy(search_space=search_space, trials=[trial])


class JsonSerializerTest(absltest.TestCase):

  def test_serialization(self):
    study = _generate_study()
    expected = """{"_type": "optformer.pyglove.types.PyGloveStudy", "search_space": {"_type": "hyper.OneOf", "name": null, "hints": null, "num_choices": 1, "candidates": [0, 1, 2, 3], "choices_distinct": true, "choices_sorted": false, "where": null}, "trials": [{"_type": "optformer.pyglove.types.PyGloveTrial", "suggestion": 3, "objective": 1.0}]}"""
    out = basic.ObjectJSONSerializer().to_str(study)
    self.assertEqual(expected, out)


if __name__ == '__main__':
  absltest.main()
