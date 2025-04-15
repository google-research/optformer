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

import dataclasses

import attrs
import jax
import numpy as np
from optformer.common import serialization as s_lib
from optformer.common.data import featurizers
from optformer.common.data import vocabs as common_vocabs
from optformer.common.serialization import numeric
from optformer.omnipred import omnipred
from optformer.omnipred import vocabs
from optformer.t5x import finetuning
from optformer.t5x import inference
from optformer.t5x import testing
import tensorflow as tf

from absl.testing import absltest

jax.config.update('jax_threefry_partitionable', False)


# Example input type. Can be anything as long as featurizer is implemented.
@dataclasses.dataclass(kw_only=True)
class _ExampleType:
  x: str
  y: float


@attrs.define
class _ExampleFeaturizer(featurizers.Featurizer[_ExampleType]):
  """Example featurizer."""

  float_serializer: s_lib.Serializer[float] = attrs.field(
      factory=numeric.DigitByDigitFloatTokenSerializer
  )

  def to_features(self, obj: _ExampleType) -> dict[str, tf.Tensor]:
    return {
        'inputs': tf.constant(obj.x),
        'targets': tf.constant(self.float_serializer.to_str(obj.y)),
    }

  @property
  def element_spec(self) -> dict[str, tf.TensorSpec]:
    return {
        'inputs': tf.TensorSpec(shape=(), dtype=tf.string),
        'targets': tf.TensorSpec(shape=(), dtype=tf.string),
    }

  @property
  def empty_output(self) -> dict[str, tf.Tensor]:
    return {
        'inputs': tf.constant('', dtype=tf.float32),
        'targets': tf.constant('', dtype=tf.float32),
    }


class OmnipredTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.featurizer = _ExampleFeaturizer()
    self.example = _ExampleType(x='hello', y=2.0)

    # Embedding table can be larger than actual vocab size.
    self.vocab = vocabs.FloatMetricVocabulary(
        common_vocabs.VOCAB_TEST_MODEL_FILE
    )
    self.num_embeddings = self.vocab.vocab_size + 5

    # Randomly initialized small model w/ deterministic seed.
    self.model = testing.small_encoder_decoder(self.vocab, self.num_embeddings)

    self.inference_config = inference.InferenceConfig(
        self.model, self.featurizer, inference.initial_train_state(self.model)
    )

    self.regressor = omnipred.OmniPred(
        self.inference_config, num_samples=1, max_inputs_length=128
    )

  def test_predict(self):
    self.assertEqual(np.median(self.regressor.predict(self.example)), 4.861e-07)

  def test_finetune(self):
    finetuner = finetuning.Finetuner(
        self.model,
        self.regressor.dataset_fn,
        learning_rate=1.0,
        max_num_epochs=2,
        batch_size=4,
        batch_per_tpu=2,
    )

    # Finetune on the exact same example.
    self.inference_config.train_state = finetuner.finetune(
        3 * [self.example],
        3 * [self.example],
        self.inference_config.train_state,
    )

    # Model weights have changed, prediction will be closer example's y-value.
    self.assertEqual(np.median(self.regressor.predict(self.example)), -4.831)


if __name__ == '__main__':
  absltest.main()
