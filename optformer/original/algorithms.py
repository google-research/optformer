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

"""Vizier Algorithms using quantization method."""

import abc
import functools
import typing
from typing import Dict, Mapping, Optional, Sequence

import attrs
import jax
import jax.numpy as jnp
from optformer.common.data import datasets
from optformer.common.inference import sequence_utils as seq_utils
from optformer.original import inference as inferencer_lib
from optformer.original import numeric
from optformer.original import serializers as os_lib
from optformer.t5x import inference as t5x_inference_lib
from optformer.vizier import serialization as vzs
from optformer.vizier.algorithms import base
from t5x import models
from tensorflow_probability.substrates import jax as tfp
from vizier import pyvizier as vz
from vizier._src.algorithms.designers.gp import acquisitions
from vizier._src.algorithms.optimizers import eagle_strategy as es
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier._src.jax import types
from vizier.pyvizier import converters

tfd = tfp.distributions


# eq=False allows jitting but won't use jit-compilation cache. Not an issue if
# we only use one instance of the class.
@attrs.define(init=False, kw_only=True)
class QuantizedVizierAlgorithm(base.Algorithm):
  """Base class for Vizier algorithms using quantization."""

  # Number of parameters denoted `P` in shape comments.
  problem: vz.ProblemStatement = attrs.field()

  _inferencer: inferencer_lib.QuantizedInferencer = attrs.field()
  _inference_dataset_fn: datasets.E2EInferenceDatasetFn[vz.ProblemAndTrials] = (
      attrs.field()
  )
  _deserializer: os_lib.QuantizedSuggestionSerializer = attrs.field()

  # Only used in Bayesian Optimization variants.
  _acquisition_function: acquisitions.AcquisitionFunction = attrs.field()

  _rng: jax.Array = attrs.field()

  def __init__(
      self,
      problem: vz.ProblemStatement,
      inference_config: t5x_inference_lib.InferenceConfig,
      *,
      num_suggest_samples: int = 256,
      num_measurement_samples: int = 256,
      acquisition_function: Optional[acquisitions.AcquisitionFunction] = None,
      max_sequence_length: int = 8192,
      seed: int = 42,
  ):
    """Custom init to reduce field ownership."""
    inferencer = inferencer_lib.QuantizedInferencer(
        problem=problem,
        model=typing.cast(models.EncoderDecoderModel, inference_config.model),
        weights=inference_config.train_state.params,
        num_suggest_samples=num_suggest_samples,
        num_measurement_samples=num_measurement_samples,
    )
    inference_dataset_fn = inference_config.get_dataset_fn(
        max_sequence_length, max_sequence_length
    )
    deserializer = os_lib.QuantizedSuggestionSerializer(
        search_space=problem.search_space,
        quantizer=numeric.NormalizedQuantizer(
            num_bins=inferencer.vocab.num_quantization_bins
        ),
    )

    self.__attrs_init__(
        problem=problem,
        inferencer=inferencer,
        inference_dataset_fn=inference_dataset_fn,
        deserializer=deserializer,
        acquisition_function=acquisition_function or acquisitions.UCB(),
        rng=jax.random.PRNGKey(seed),
    )

  @abc.abstractmethod
  def stateless_suggest(
      self, count: int, history: Sequence[vz.Trial]
  ) -> Sequence[vz.TrialSuggestion]:
    """Can be wrapped into a Vizier Designer or Pythia Policy later on."""

  @property
  def _num_parameters(self) -> int:
    return self.problem.search_space.num_parameters()

  @functools.cached_property
  def _xy_separator_id(self) -> int:
    s = self._deserializer.token_serializer.to_str(
        [vzs.TrialTokenSerializer.XY_SEPARATOR]
    )
    token_ids = self._inferencer.vocab.encode(s)
    return token_ids[1]

  @functools.cached_property
  def _trial_separator_id(self) -> int:
    s = self._deserializer.token_serializer.to_str(
        [os_lib.QuantizedTrialsSerializer.TRIAL_SEPARATOR]
    )
    token_ids = self._inferencer.vocab.encode(s)
    return token_ids[1]

  def _studies_to_batch(
      self, studies: Sequence[vz.ProblemAndTrials]
  ) -> Dict[str, jnp.ndarray]:
    dataset = self._inference_dataset_fn(studies)
    dataset = dataset.batch(len(studies))
    return next(dataset.as_numpy_iterator())

  def _quantized_suggestion_to_pyvizier(
      self, jax_suggestion: jnp.ndarray
  ) -> vz.TrialSuggestion:
    """Converts jax suggestion back to PyVizier."""
    s = self._inferencer.vocab.decode(jax_suggestion.tolist())
    return self._deserializer.from_str(s)

  def _regressed_acquisition_function(
      self,
      new_batch: Mapping[str, jnp.ndarray],
      index: jnp.ndarray,
  ) -> types.Array:
    dist, _ = self._inferencer.regress(new_batch, index)
    return self._acquisition_function(dist)


class DirectSamplingAlgorithm(QuantizedVizierAlgorithm):
  """Samples suggestion from inferencer directly."""

  def stateless_suggest(
      self, count: int, history: Sequence[vz.Trial]
  ) -> Sequence[vz.TrialSuggestion]:
    if count > 1:
      raise NotImplementedError('Currently does not support batching.')

    # Convert study to numpy.
    study = vz.ProblemAndTrials(self.problem, history)
    batch = self._studies_to_batch([study])

    jax_suggestion = self._jax_suggest(batch)
    return [self._quantized_suggestion_to_pyvizier(jax_suggestion)]

  def _jax_suggest(self, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    # [B=1, S, P]
    self._rng, subk = jax.random.split(self._rng)
    jax_suggestions, _ = self._inferencer.sample_suggestion_tokens(batch, subk)
    return jax_suggestions[0, -1, :]  # [P]  Obtain sample w/ highest logprob.


class SelfRankingAlgorithm(QuantizedVizierAlgorithm):
  """Outputs the argmax sample w.r.t. regressor's acquisition function."""

  def stateless_suggest(
      self, count: int, history: Sequence[vz.Trial]
  ) -> Sequence[vz.TrialSuggestion]:
    if count > 1:
      raise NotImplementedError('Currently do not support batching.')

    # Convert study to numpy.
    study = vz.ProblemAndTrials(self.problem, history)
    batch = self._studies_to_batch([study])

    best_jax_sugg = self._jax_suggest(batch)
    return [self._quantized_suggestion_to_pyvizier(best_jax_sugg)]

  def _jax_suggest(self, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    self._rng, subk = jax.random.split(self._rng)
    # [B=1, S, P], [B=1, S, L]
    jax_sugg, full_sugg = self._inferencer.sample_suggestion_tokens(batch, subk)
    jax_sugg = jnp.squeeze(jax_sugg, axis=0)  # [S, P]
    full_sugg = jnp.squeeze(full_sugg, axis=0)  # [S, L]

    # Add measurement separator token to prepare for regression.
    # Need to modify suggestion `[..., p1, p2, p3] -> [..., p1, p2, p3, *]` to
    # prompt the next `m`.
    sep_inds = seq_utils.count_not_from(full_sugg)  # [S]
    sep_inds = seq_utils.reduce_eq(sep_inds)  # Scalar

    # [S, L]
    regress_prompt = full_sugg.at[:, sep_inds].set(self._xy_separator_id)

    # Prepare for input by left-appending the `0`.
    regress_prompt = seq_utils.shift_right(regress_prompt)  # [S, L]
    sep_inds += 1

    # Create new batch prompt with all entries of shape [S, L].
    new_batch = {k: jnp.squeeze(v, axis=0) for k, v in batch.items()}
    new_batch = seq_utils.broadcast_batch(
        new_batch, [self._inferencer.num_suggest_samples]
    )
    new_batch['decoder_input_tokens'] = regress_prompt

    # Compute acquisition scores. Shape [S].
    acq_scores = self._regressed_acquisition_function(new_batch, sep_inds)

    # Find best suggestion according to acquisition scores.
    best_idx = jnp.argmax(acq_scores)  # Scalar
    return jax_sugg[best_idx]  # [P]


class LevelSetAlgorithm(QuantizedVizierAlgorithm):
  """Uses a level-set sampler which predicts `x` given `y`, such that y == f(x).

  The algorithm queries high y-values (defaulted to quantized max) to produce
  promising suggestions.
  """

  def stateless_suggest(
      self, count: int, history: Sequence[vz.Trial]
  ) -> Sequence[vz.TrialSuggestion]:
    # NOTE: `TrialSerializer` should be using `yx` ordering.
    if count > 1:
      raise NotImplementedError('Currently does not support batching.')

    # Convert study to numpy.
    study = vz.ProblemAndTrials(self.problem, history)
    batch = self._studies_to_batch([study])

    jax_suggestion = self._jax_suggest(batch)
    return [self._quantized_suggestion_to_pyvizier(jax_suggestion)]

  def _jax_suggest(self, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    # [0, ..., |, 0, 0, 0, ...] -> [0, ..., |, y, *, 0, ...]
    decoder_inp = jnp.squeeze(batch['decoder_input_tokens'], axis=0)  # [L]
    start = seq_utils.count_not_from(decoder_inp) + 1  # Scalar
    new_toks = [self._target_quantized_y, self._xy_separator_id]
    decoder_inp = seq_utils.slice_update(decoder_inp, start, new_toks)
    batch['decoder_input_tokens'] = jnp.expand_dims(decoder_inp, axis=0)

    self._rng, subk = jax.random.split(self._rng)
    # [B=1, S, P]
    jax_suggestions, _ = self._inferencer.sample_suggestion_tokens(batch, subk)
    return jax_suggestions[0, -1, :]  # [P]  Obtain sample w/ highest logprob.

  @functools.cached_property
  def _target_quantized_y(self) -> int:
    """Token ID of the highest quantized (ex: 999) value."""
    zero = self._inferencer.vocab.quantization_vocab_index
    num_bins = self._inferencer.vocab.num_quantization_bins
    return zero + num_bins - 1


@attrs.define(init=False)
class VizierOptimizerAlgorithm(QuantizedVizierAlgorithm):
  """Uses Eagle optimizer on optformer acquisition function.

  The default acquisition function is UCB and the default optimizer is Eagle.
  The class assumes linear y scaling and uniformly spaced quantization scheme.
  """

  def __init__(
      self,
      problem: vz.ProblemStatement,
      inference_config: t5x_inference_lib.InferenceConfig,
      *,
      acquisition_function: Optional[acquisitions.AcquisitionFunction] = None,
      optimizer_factory: Optional[vb.VectorizedOptimizerFactory] = None,
      max_sequence_length: int = 8192,
      seed: int = 42,
  ):
    super().__init__(
        problem,
        inference_config,
        acquisition_function=acquisition_function,
        max_sequence_length=max_sequence_length,
        seed=seed,
    )

    self._converter = converters.TrialToModelInputConverter.from_problem(
        problem,
        scale=True,
        max_discrete_indices=0,
        flip_sign_for_minimization_metrics=True,
    )

    optimizer_factory = optimizer_factory or vb.VectorizedOptimizerFactory(
        strategy_factory=es.VectorizedEagleStrategyFactory(),
        use_fori=False,
    )
    self._optimizer = optimizer_factory(self._converter)

  def stateless_suggest(
      self, count: int, history: Sequence[vz.Trial]
  ) -> Sequence[vz.TrialSuggestion]:
    """Suggest trials by running optimization on Optformer acquisition fn."""
    if count > 1:
      raise NotImplementedError('Batching unsupported currently.')

    # Create batch from history trials.
    previous_study = vz.ProblemAndTrials(self.problem, history)
    batch = self._studies_to_batch([previous_study])

    # Location of `*` during input == Location of `y` during output.
    # [..., |, p1, p2, p3, *, y]
    regress_index = (
        seq_utils.rfind(batch['decoder_input_tokens'], self._trial_separator_id)
        + self._num_parameters
        + 1
    ).item()

    def score_fn(xs: types.ModelInput, seed: jax.Array) -> types.Array:
      del seed
      tokenized_trials = self._tokenize_suggestion_features(xs)
      new_batch = self._append_tokenized_trials(batch, tokenized_trials)
      return self._regressed_acquisition_function(new_batch, regress_index)

    best_candidates: vb.VectorizedStrategyResults = self._optimizer(
        score_fn, prior_features=self._converter.to_features(history)
    )
    return vb.best_candidates_to_trials(best_candidates, self._converter)

  def _inefficient_suggest(
      self, count: int, history: Sequence[vz.Trial]
  ) -> Sequence[vz.TrialSuggestion]:
    """Redundantly tokenizes history repeatedly; for debugging."""
    if count > 1:
      raise NotImplementedError('Batching unsupported currently.')

    def score_fn(
        xs: types.ModelInput, seed: Optional[jax.Array] = None
    ) -> types.Array:
      del seed
      histories = [
          list(history) + [vz.Trial(params)]
          for params in self._converter.to_parameters(xs)
      ]
      studies = [
          vz.ProblemAndTrials(self.problem, trials) for trials in histories
      ]
      batch = self._studies_to_batch(studies)

      regress_index = seq_utils.rfind(
          batch['decoder_input_tokens'], self._trial_separator_id
      )
      regress_index = seq_utils.reduce_eq(regress_index)
      return self._regressed_acquisition_function(batch, regress_index)

    best_candidates: vb.VectorizedStrategyResults = self._optimizer(
        score_fn, prior_features=self._converter.to_features(history)
    )
    return vb.best_candidates_to_trials(best_candidates, self._converter)

  def _tokenize_suggestion_features(self, xs: types.ModelInput) -> jnp.ndarray:
    trials = [vz.Trial(params) for params in self._converter.to_parameters(xs)]
    studies = [vz.ProblemAndTrials(self.problem, [trial]) for trial in trials]
    tokenized_trials = self._studies_to_batch(studies)['decoder_input_tokens']

    # Slice to obtain [p1, p2, p3, *]
    xy_sep_ind = seq_utils.rfind(tokenized_trials, self._xy_separator_id)[0]
    trial_start_ind = xy_sep_ind - self._num_parameters
    return tokenized_trials[:, trial_start_ind : xy_sep_ind + 1]  # [N, P + 1]

  def _append_tokenized_trials(
      self,
      batch: Mapping[str, jnp.ndarray],
      tokenized_trials: jnp.ndarray,
  ) -> Mapping[str, jnp.ndarray]:
    """Append trial tokens to 'decoder_input_tokens'."""
    batch = {k: jnp.squeeze(v, axis=0) for k, v in batch.items()}  # [L]
    num_trials = len(tokenized_trials)

    # TODO: Replace w/ `seq_utils.append()`.
    # Compute beginning index for appending (accounting for initial '0')
    append_index = seq_utils.count_not_from(batch['decoder_input_tokens']) + 1
    append_len = tokenized_trials.shape[-1]

    batch = seq_utils.broadcast_batch(batch, [num_trials])  # [N, L]
    new_input_tokens = batch['decoder_input_tokens']
    new_input_tokens = new_input_tokens.at[
        :, append_index : append_index + append_len
    ].set(tokenized_trials)

    batch['decoder_input_tokens'] = new_input_tokens

    return batch
