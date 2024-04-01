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

from optformer.pyglove.data.generators import studies
from absl.testing import absltest


class StudyFactoriesTest(absltest.TestCase):

  def test_synthetic(self):
    num_trials = 20
    study_factory = studies.SyntheticStudyFactory(num_trials=num_trials)

    for seed in range(5):
      study = study_factory(seed=seed)

      study_same = study_factory(seed=seed)
      study_different = study_factory(seed=seed + 1)
      self.assertEqual(study, study_same)
      self.assertNotEqual(study, study_different)

      self.assertLen(study.trials, num_trials)


if __name__ == "__main__":
  absltest.main()
