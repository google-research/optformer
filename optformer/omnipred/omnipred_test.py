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

import numpy as np
from optformer.common.data import vocabs as common_vocabs
from optformer.omnipred import omnipred
from optformer.omnipred import serialization
from optformer.omnipred import vocabs
from optformer.t5x import finetuning
from optformer.t5x import inference
from optformer.t5x import testing
from optformer.vizier.data import featurizers
from vizier import pyvizier as vz
from absl.testing import absltest


class OmnipredTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    m = vz.MetricInformation(name='obj', goal=vz.ObjectiveMetricGoal.MINIMIZE)
    self.study = vz.ProblemAndTrials(
        problem=vz.ProblemStatement(metric_information=vz.MetricsConfig([m])),
        trials=[vz.Trial(parameters={'a': 5, 'b': 'xyz'})],
    )
    # Embedding table can be larger than actual vocab size.
    self.vocab = vocabs.FloatMetricVocabulary(
        common_vocabs.VOCAB_TEST_MODEL_FILE
    )
    self.num_embeddings = self.vocab.vocab_size + 5

    # Randomly initialized small model w/ deterministic seed.
    self.model = testing.small_encoder_decoder(self.vocab, self.num_embeddings)
    self.featurizer = featurizers.VizierStudyFeaturizer(
        serialization.OmniPredInputsSerializer,
        serialization.OmniPredTargetsSerializer,
    )

    self.inference_config = inference.InferenceConfig(
        self.model, self.featurizer, inference.initial_train_state(self.model)
    )

    self.regressor = omnipred.OmniPred(
        self.inference_config, num_samples=1, max_inputs_length=128
    )

  def test_predict(self):
    self.assertEqual(np.median(self.regressor.predict(self.study)), 4.861e-07)

  def test_finetune(self):
    finetuner = finetuning.Finetuner(
        self.model,
        self.regressor.dataset_fn,
        learning_rate=1.0,
        max_num_epochs=2,
        batch_size=4,
        batch_per_tpu=2,
    )

    self.inference_config.train_state = finetuner.finetune(
        3 * [self.study], 3 * [self.study], self.inference_config.train_state
    )
    self.assertEqual(np.median(self.regressor.predict(self.study)), 5.831e-07)


if __name__ == '__main__':
  absltest.main()
