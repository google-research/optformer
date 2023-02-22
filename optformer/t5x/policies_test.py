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

"""Tests for policies."""
from optformer.t5x import inference_utils
from optformer.t5x import policies
from vizier import algorithms as vza
from vizier import benchmarks
from absl.testing import absltest


class PoliciesTest(absltest.TestCase):

  @absltest.skip("Checkpoint must be installed manually.")
  def test_e2e(self):
    experimenter = benchmarks.IsingExperimenter(lamda=0.01)

    inference_model = inference_utils.InferenceModel.from_checkpoint(
        **policies.BBOB_INFERENCE_MODEL_KWARGS)
    designer = policies.OptFormerDesigner(
        experimenter.problem_statement(), inference_model=inference_model)

    for _ in range(2):
      suggestions = designer.suggest(1)
      trials = [suggestion.to_trial() for suggestion in suggestions]
      experimenter.evaluate(trials)
      designer.update(
          completed=vza.CompletedTrials(trials), all_active=vza.ActiveTrials()
      )


if __name__ == '__main__':
  absltest.main()
