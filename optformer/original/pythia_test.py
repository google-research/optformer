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

from optformer.original import algorithms
from optformer.original import featurizers
from optformer.original import vocabs
from optformer.t5x import inference
from optformer.t5x import testing
from optformer.vizier.algorithms import pythia as optformer_pythia
from vizier import pythia
from vizier import pyvizier as vz
from vizier._src.algorithms.testing import test_runners
from vizier.testing import test_studies
from absl.testing import absltest


class PolicyWrapperTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.problem = vz.ProblemStatement(
        search_space=test_studies.flat_space_with_all_types(),
    )
    metric = vz.MetricInformation(name="", goal=vz.ObjectiveMetricGoal.MAXIMIZE)
    self.problem.metric_information.append(metric)

    self.featurizer = featurizers.get_eval_featurizer()
    self.vocab = vocabs.QuantizedVocabulary(vocabs.VOCAB_TEST_MODEL_FILE)
    self.model = testing.small_encoder_decoder(vocab=self.vocab)

    self.inference_config = inference.InferenceConfig(
        self.model, self.featurizer, inference.initial_train_state(self.model)
    )

    # Made small to optimize test speed.
    self.max_sequence_length = 128

  def test_suggest_smoke(self):
    algorithm = algorithms.DirectSamplingAlgorithm(
        problem=self.problem,
        inference_config=self.inference_config,
        num_suggest_samples=1,
        max_sequence_length=self.max_sequence_length,
    )

    def policy_factory(
        supporter: pythia.PolicySupporter,
    ) -> optformer_pythia.OptFormerPolicy:
      return optformer_pythia.OptFormerPolicy(algorithm, supporter)

    runner = test_runners.RandomMetricsRunner(
        self.problem, iters=2, batch_size=1
    )
    trials = runner.run_policy(policy_factory)
    self.assertLen(trials, 2)


if __name__ == "__main__":
  absltest.main()
