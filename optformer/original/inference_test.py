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

import functools

import jax
import jax.numpy as jnp
import numpy as np
from optformer.common.data import datasets
from optformer.common.data import vocabs as common_vocabs
from optformer.original import featurizers
from optformer.original import inference
from optformer.original import vocabs
from optformer.t5x import testing as t5x_testing
import seqio
from t5x import decoding
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import grid
from vizier.testing import test_studies

from absl.testing import absltest
from absl.testing import parameterized

_VOCAB_TEST_MODEL_FILE = common_vocabs.VOCAB_TEST_MODEL_FILE


class LogitRestrictorsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    # Vocab can have special strings and embedding table can be larger than
    # actual vocab size.
    self.vocab = vocabs.QuantizedVocabulary(_VOCAB_TEST_MODEL_FILE)
    self.num_embeddings = self.vocab.vocab_size + 42

    def tokens_to_logits(decoding_state: decoding.DecodingState):
      del decoding_state
      # Uniform distribution over targets from model
      logits = np.zeros(
          shape=(self.num_decodes, self.num_embeddings),
          dtype=np.float32,
      )
      return logits, {}

    self.tokens_to_logits = tokens_to_logits

    self.num_decodes = 2

  def test_suggestion_logit_restrictor(self):
    search_space = test_studies.flat_space_with_all_types()
    logit_restrictor = inference._SuggestionLogitRestrictor(
        search_space, vocab=self.vocab
    )

    # Model will fill-in all zero'ed positions.
    inputs = np.zeros(
        shape=(self.num_decodes, search_space.num_parameters()), dtype=np.int32
    )

    # Check JIT-ing and perform sampling.
    partial_fn = functools.partial(
        decoding._temperature_sample_single_trial,
        cache={},
        tokens_to_logits=self.tokens_to_logits,
        eos_id=1,
        prng_key=jax.random.PRNGKey(0),
        topk=0,
        max_decode_steps=search_space.num_parameters(),
        logit_callback_fn=logit_restrictor,
    )
    jit_partial_fn = jax.jit(partial_fn)
    sampled_sequences, _ = jit_partial_fn(inputs)

    quantized_values = sampled_sequences - self.vocab.quantization_vocab_index

    # This makes sense as the first 2 params are DOUBLE, the third is INTEGER,
    # while the rest of the params are CATEGORICAL / DISCRETE.
    expected = [[550, 419, 337, 0, 1, 1, 2, 0], [861, 814, 61, 2, 1, 0, 2, 1]]
    np.testing.assert_array_equal(expected, quantized_values)

  def test_measurement_logit_restrictor(self):
    m1 = vz.MetricInformation(name="x1", goal=vz.ObjectiveMetricGoal.MAXIMIZE)
    m2 = vz.MetricInformation(name="x2", goal=vz.ObjectiveMetricGoal.MINIMIZE)
    metrics_config = vz.MetricsConfig(metrics=[m1, m2])

    logit_restrictor = inference._MeasurementLogitRestrictor(
        metrics_config, vocab=self.vocab
    )

    # Model will fill-in all zero'ed positions.
    inputs = np.zeros(
        shape=(self.num_decodes, len(metrics_config)), dtype=np.int32
    )

    # Check JIT-ing and perform sampling.
    partial_fn = functools.partial(
        decoding._temperature_sample_single_trial,
        cache={},
        tokens_to_logits=self.tokens_to_logits,
        eos_id=1,
        prng_key=jax.random.PRNGKey(0),
        topk=0,
        max_decode_steps=len(metrics_config),
        logit_callback_fn=logit_restrictor,
    )
    jit_partial_fn = jax.jit(partial_fn)
    sampled_sequences, _ = jit_partial_fn(inputs)

    quantized_values = sampled_sequences - self.vocab.initial_extra_token_id

    expected = [[550, 419], [861, 814]]
    np.testing.assert_array_equal(expected, quantized_values)


def _flat_all_param_study() -> vz.ProblemAndTrials:
  problem = vz.ProblemStatement(
      search_space=test_studies.flat_space_with_all_types(),
      metric_information=[
          vz.MetricInformation("m", goal=vz.ObjectiveMetricGoal.MAXIMIZE),
      ],
  )

  designer = grid.GridSearchDesigner(problem.search_space)
  trials = [t.to_trial() for t in designer.suggest(2)]
  trials[0].complete(vz.Measurement(metrics={"m": 0.0}))
  trials[1].complete(vz.Measurement(metrics={"m": 1.0}))

  return vz.ProblemAndTrials(problem, trials)


class InferencerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.study = _flat_all_param_study()

    # Vocab can have special strings and embedding table can be larger than
    # actual vocab size.
    self.vocab = vocabs.QuantizedVocabulary(_VOCAB_TEST_MODEL_FILE)
    self.num_embeddings = self.vocab.vocab_size + 42

    self.max_sequence_length = 32
    self.dataset_fn = datasets.E2EInferenceDatasetFn(
        featurizer=featurizers.get_eval_featurizer(),
        input_vocabulary=self.vocab,
        output_vocabulary=self.vocab,
        feature_converter=seqio.feature_converters.EncDecFeatureConverter(
            pack=False
        ),
        max_inputs_length=self.max_sequence_length,
        max_targets_length=self.max_sequence_length,
    )

    # Randomly initialized small model w/ deterministic seed.
    self.model = t5x_testing.small_encoder_decoder(
        self.vocab, self.num_embeddings
    )

    self.num_samples = 3
    self.inferencer = inference.QuantizedInferencer(
        problem=self.study.problem,
        model=self.model,
        weights=self.model.get_initial_variables(
            rng=jax.random.PRNGKey(42),
            input_shapes={
                "encoder_input_tokens": (1, 2),
                "decoder_input_tokens": (1, 2),
            },
        )["params"],
        num_suggest_samples=self.num_samples,
        num_measurement_samples=self.num_samples,
    )

  def test_sample_suggestions(self):
    # Prepare Jax-ified input
    dataset = self.dataset_fn([self.study])
    dataset = dataset.batch(1)
    jax_study = next(dataset.as_numpy_iterator())

    # Perform inference.
    rng = jax.random.PRNGKey(42)
    jax_sugg, full = self.inferencer.sample_suggestion_tokens(jax_study, rng)

    # Remove batch=1 outer axis.
    jax_sugg = jnp.squeeze(jax_sugg, axis=0)
    full = jnp.squeeze(full, axis=0)

    suggestion_shape = (
        self.num_samples,
        self.study.problem.search_space.num_parameters(),
    )
    self.assertEqual(jax_sugg.shape, suggestion_shape)
    self.assertEqual(full.shape, (self.num_samples, self.max_sequence_length))

    # Make sure suggestion token IDs are from quantized set.
    quantized_values = jax_sugg - self.vocab.quantization_vocab_index
    np.testing.assert_array_less(-1, quantized_values)
    np.testing.assert_array_less(
        quantized_values, self.vocab.num_quantization_bins
    )
    # Makes sense; first two params are DOUBLE, third is INTEGER bound,
    # last are CATEGORICAL / DISCRETE w/ few feasible values.
    expected = [
        [211, 836, 440, 1, 1, 1, 1, 0],
        [67, 412, 727, 0, 1, 0, 1, 0],
        [719, 557, 343, 0, 1, 0, 1, 1],
    ]
    np.testing.assert_array_equal(expected, quantized_values)

  def test_sample_measurements(self):
    # NOTE: This is a SMOKE test. The index might not be properly aligned to
    # produce the actual measurement tokens.

    # Prepare Jax-ified input
    dataset = self.dataset_fn([self.study])
    dataset = dataset.batch(1)
    jax_study = next(dataset.as_numpy_iterator())

    # JIT-check and perform inference.
    rng = jax.random.PRNGKey(42)
    jax_meas, full = self.inferencer.sample_measurement_tokens(jax_study, rng)

    # Remove batch=1 outer axis.
    jax_meas = jnp.squeeze(jax_meas, axis=0)
    full = jnp.squeeze(full, axis=0)

    measurement_shape = (
        self.num_samples,
        len(self.study.problem.metric_information),
    )
    self.assertEqual(jax_meas.shape, measurement_shape)
    self.assertEqual(full.shape, (self.num_samples, self.max_sequence_length))

    # Make sure metric token IDs are from quantized set.
    quantized_values = jax_meas - self.vocab.quantization_vocab_index
    np.testing.assert_array_less(-1, quantized_values)
    np.testing.assert_array_less(
        quantized_values, self.vocab.num_quantization_bins
    )
    expected = [[211], [67], [719]]
    np.testing.assert_array_equal(expected, quantized_values)

  @parameterized.parameters(
      (np.array(4),),
      # np.array([4, 6])),  # TODO: Fix non-scalar case.
  )
  def test_jax_regressor(self, index: np.ndarray):
    # NOTE: This is a SMOKE test. The index might not be properly aligned to
    # produce the logits for the actual measurement token.
    batch_size = 2

    # Prepare Jax-ified input.
    dataset = self.dataset_fn(batch_size * [self.study])
    dataset = dataset.batch(batch_size)
    jax_study = next(dataset.as_numpy_iterator())

    # JIT-check and perform inference.
    jit_regress = jax.jit(self.inferencer.regress)
    dist, full_output = jit_regress(jax_study, index)

    # Check distribution works.
    mean_pred = dist.mean()
    np.testing.assert_array_less(0.0, mean_pred)
    np.testing.assert_array_less(mean_pred, self.vocab.num_quantization_bins)

    # Check output shape.
    self.assertEqual(
        full_output.shape,
        (batch_size, self.max_sequence_length, self.num_embeddings),
    )


if __name__ == "__main__":
  absltest.main()
