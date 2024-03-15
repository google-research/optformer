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

import random

from absl import logging
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from optformer.common.data import vocabs as common_vocabs
from optformer.common.inference import sequence_utils as seq_utils
from optformer.original import algorithms
from optformer.original import featurizers
from optformer.original import vocabs
from optformer.t5x import inference
from optformer.t5x import testing
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import grid
from vizier._src.algorithms.designers.gp import acquisitions
from vizier._src.algorithms.optimizers import eagle_strategy as es
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier._src.jax import types
from vizier.testing import test_studies

from absl.testing import absltest

# NOTE: Some tests don't work with the test vocabulary.
_VOCAB_TEST_MODEL_FILE = common_vocabs.VOCAB_TEST_MODEL_FILE


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


class AlgorithmTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.study = _flat_all_param_study()

    # Vocab has special Vizier tokens and embedding table can be larger than
    # actual vocab size.
    self.vocab = vocabs.QuantizedVocabulary(_VOCAB_TEST_MODEL_FILE)
    self.num_embeddings = self.vocab.vocab_size + random.randint(0, 10)

    # Randomly initialized small model w/ deterministic seed.
    model = testing.small_encoder_decoder(self.vocab)

    featurizer = featurizers.get_eval_featurizer()
    self.inference_config = inference.InferenceConfig(
        model, featurizer, inference.initial_train_state(model)
    )

    # Made small to optimize test speed.
    self.max_sequence_length = 128


class DirectSamplingAlgorithmTest(AlgorithmTest):

  def test_suggest(self):
    algorithm = algorithms.DirectSamplingAlgorithm(
        problem=self.study.problem,
        inference_config=self.inference_config,
        num_suggest_samples=2,
        max_sequence_length=self.max_sequence_length,
    )

    for i in range(2):
      suggestion = algorithm.stateless_suggest(1, self.study.trials)[0]
      logging.info("Suggestion %d is: %s", i, suggestion)

      # Verify suggestion is valid.
      contained = self.study.problem.search_space.contains(
          suggestion.parameters
      )
      self.assertTrue(contained)


class SelfRankingTest(AlgorithmTest):

  def setUp(self):
    super().setUp()
    self.num_suggest_samples = 2
    self.algorithm = algorithms.SelfRankingAlgorithm(
        problem=self.study.problem,
        inference_config=self.inference_config,
        num_suggest_samples=self.num_suggest_samples,
        max_sequence_length=self.max_sequence_length,
    )

  def test_suggest(self):
    for i in range(2):
      suggestion = self.algorithm.stateless_suggest(1, self.study.trials)[0]
      logging.info("Suggestion %d is: %s", i, suggestion)

      # Verify suggestion is valid.
      contained = self.study.problem.search_space.contains(
          suggestion.parameters
      )
      self.assertTrue(contained)

  def test_separation_index(self):
    dataset = self.algorithm._inference_dataset_fn([self.study])
    dataset = dataset.batch(1)
    batch = next(dataset.as_numpy_iterator())
    _, full_suggestions = self.algorithm._inferencer.sample_suggestion_tokens(
        batch
    )
    full_suggestions = jnp.squeeze(full_suggestions, axis=0)  # [S, L]

    # Check that the first token in output is `329`, used for spaces.
    initial_tokens = jnp.array(
        [self.algorithm._inferencer.vocab.encode(" ")[0]]
        * self.num_suggest_samples
    )
    np.testing.assert_array_equal(
        full_suggestions[:, 0],
        initial_tokens,
    )
    # Check that the separation index is as expected.
    sep_index = seq_utils.count_not_from(full_suggestions)
    # The separation index should be placed after 1+11+11+9 = 32 tokens.
    #  ,|,p,p,p,p,p,p,p,p,*,m,|,p,p,p,p,p,p,p,p,*,m,|,p,p,p,p,p,p,p,p,_
    np.testing.assert_array_equal(
        sep_index,
        jnp.array([32] * self.num_suggest_samples),
    )


class LevelSetTest(AlgorithmTest):

  def test_suggest(self):
    algorithm = algorithms.LevelSetAlgorithm(
        problem=self.study.problem,
        inference_config=self.inference_config,
        num_suggest_samples=2,
        max_sequence_length=self.max_sequence_length,
    )
    for i in range(2):
      suggestion = algorithm.stateless_suggest(1, self.study.trials)[0]
      logging.info("Suggestion %d is: %s", i, suggestion)

      # Verify suggestion is valid.
      contained = self.study.problem.search_space.contains(
          suggestion.parameters
      )
      self.assertTrue(contained)


class VizierOptimizerTest(AlgorithmTest):

  def setUp(self):
    super().setUp()
    optimizer_factory = vb.VectorizedOptimizerFactory(
        strategy_factory=es.VectorizedEagleStrategyFactory(),
        max_evaluations=10,
        suggestion_batch_size=3,
        use_fori=False,
    )
    self.algorithm = algorithms.VizierOptimizerAlgorithm(
        problem=self.study.problem,
        inference_config=self.inference_config,
        acquisition_function=acquisitions.UCB(),
        optimizer_factory=optimizer_factory,
        max_sequence_length=self.max_sequence_length,
    )

  def test_suggest(self):
    suggestions = self.algorithm.stateless_suggest(
        count=1,
        history=self.study.trials,
    )
    self.assertLen(suggestions, 1)
    # Verify suggestion is valid.
    contained = self.study.problem.search_space.contains(
        suggestions[0].parameters
    )
    self.assertTrue(contained)

  def test_inefficient_suggest(self):
    suggestions = self.algorithm._inefficient_suggest(
        count=1, history=self.study.trials
    )
    self.assertLen(suggestions, 1)
    # Verify suggestion is valid.
    contained = self.study.problem.search_space.contains(
        suggestions[0].parameters
    )
    self.assertTrue(contained)

  @parameterized.parameters(1, 5)
  def test_create_features_tokens(self, batch_size):
    """Test converting features to trial tokens."""
    empty_features = self.algorithm._converter.to_features([])
    n_cont_features = empty_features.continuous.shape[-1]
    n_cat_features = empty_features.categorical.shape[-1]
    xs = types.ModelInput(
        continuous=types.PaddedArray.as_padded(
            np.random.uniform(0.0, 1.0, size=(batch_size, n_cont_features))
        ),
        categorical=types.PaddedArray.as_padded(
            np.zeros((batch_size, n_cat_features), dtype=types.INT_DTYPE)
        ),
    )
    trials_tokens = self.algorithm._tokenize_suggestion_features(xs)
    self.assertLen(trials_tokens, batch_size)

    # Check that the param tokens are within a valid range.
    params_tokens = trials_tokens[:, :-1]  # Remove last xy_separator token.
    np.testing.assert_array_less(
        self.vocab.quantization_vocab_index - 1, params_tokens
    )

    # Check last token is XY-separator.
    xy_sep_tokens = trials_tokens[:, -1]
    np.testing.assert_array_equal(
        xy_sep_tokens, self.algorithm._xy_separator_id
    )

  def test_y_value_index(self):
    """Test that the y-value index computation is correct."""
    batch = self.algorithm._studies_to_batch([self.study])
    regress_index = (
        seq_utils.rfind(
            batch["decoder_input_tokens"], self.algorithm._trial_separator_id
        )
        + self.algorithm._num_parameters
        + 1
    )
    # decoder_input_tokens:
    # 0,<329>,|,p,p,p,p,p,p,p,p,*,m,|,p,p,p,p,p,p,p,p,*,m,|,p,p,p,p,p,p,p,p,*,_
    np.testing.assert_array_equal(
        regress_index,
        jnp.array([33]),
    )

  def test_add_trials_tokens_to_batch(self):
    batch = self.algorithm._studies_to_batch([self.study])
    xy_sep = self.algorithm._xy_separator_id
    tokens = [
        [32001, 32002, 32003, 32004, xy_sep],
        [32011, 32012, 32013, 32014, xy_sep],
    ]
    tokens = jnp.array(tokens)

    new_batch = self.algorithm._append_tokenized_trials(batch, tokens)
    self.assertTrue(new_batch["decoder_input_tokens"].shape[0], 2)
    # 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21
    # 0 329 |  p1 p1 p1 p1 *  m1 |  p2 p2 p2 p2 *  m2 |  t  t  t  t  *
    jnp.array_equal(
        new_batch["decoder_input_tokens"][0][17 : 21 + 1], jnp.array(tokens[0])
    )
    jnp.array_equal(
        new_batch["decoder_input_tokens"][1][17 : 21 + 1], jnp.array(tokens[1])
    )


if __name__ == "__main__":
  absltest.main()
