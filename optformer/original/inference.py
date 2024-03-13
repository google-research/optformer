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

"""Quantization-based (i.e. relative tokenization) inferencer."""

import functools
import typing
from typing import Any, Mapping, Optional, Tuple

import attrs
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
from optformer.common.inference import sequence_utils as seq_utils
from optformer.original import vocabs
from optformer.t5x import decoding
from t5x import models
from tensorflow_probability.substrates import jax as tfp
from vizier import pyvizier as vz

tfd = tfp.distributions


@attrs.define(auto_attribs=False)
class _SuggestionLogitRestrictor(decoding.IndexLogitRestrictor):
  """Restricts logit values according to current parameter being sampled.

  For a given search space with P parameters, to produce a trial suggestion, the
  model needs to autoregressively sample P times.

  Each parameter p's logit must be restricted to the valid range of
  quantized values (<0>, <1>,... <q_p>).
  """

  search_space: vz.SearchSpace = attrs.field()
  vocab: vocabs.QuantizedVocabulary = attrs.field()

  _logits_masks: jnp.ndarray = attrs.field(init=False)

  # Replaces `search_space.num_parameters()` with a fixed constant to avoid
  # re-jitting the __call__ function for different search spaces.
  MAX_NUM_PARAMETERS: int = 100

  def __attrs_post_init__(self):
    """Compute the valid quantized value masks for each parameter config."""
    masks = np.zeros((self.MAX_NUM_PARAMETERS, self.vocab.vocab_size))
    start = self.vocab.quantization_vocab_index

    for p, pc in enumerate(self.search_space.parameters):
      if pc.type in [vz.ParameterType.DOUBLE, vz.ParameterType.INTEGER]:
        masks[p, start : start + self.vocab.num_quantization_bins] = 1.0
      else:  # CATEGORICAL, DISCRETE
        masks[p, start : start + pc.num_feasible_values] = 1.0

    self._logits_masks = jnp.array(masks)

  def logit_mask(self, index: Int[Array, "BS"]) -> Float[Array, "BS V"]:
    return self._logits_masks[index]


@attrs.define(auto_attribs=False)
class _MeasurementLogitRestrictor(decoding.IndexLogitRestrictor):
  """Restricts logit values to legitimate metric tokens.

  For a given MetricsConfig w/ M parameters, model must autoregressively sample
  M times.

  Each metric m's logit must be restricted to the valid range of
  quantized values (<0>, <1>,... <999>).
  """

  metrics_config: vz.MetricsConfig = attrs.field()
  vocab: vocabs.QuantizedVocabulary = attrs.field()

  _logits_masks: jnp.ndarray = attrs.field(init=False)

  # Replaces `len(metric_information)` w/ fixed constant to avoid re-jitting the
  # __call__ function for different metric configs.
  MAX_NUM_METRICS: int = 100

  def __attrs_post_init__(self):
    """Compute valid quantized value masks for each metric config."""
    masks = np.zeros((self.MAX_NUM_METRICS, self.vocab.vocab_size))
    start = self.vocab.quantization_vocab_index

    # TODO: Add infeasibility token prediction.
    for m in range(len(self.metrics_config)):
      masks[m, start : start + self.vocab.num_quantization_bins] = 1.0

    self._logits_masks = jnp.array(masks)

  def logit_mask(self, index: Int[Array, "BS"]) -> Float[Array, "BS V"]:
    return self._logits_masks[index]


def _quantized_model_validator(
    instance: Any,
    attribute: attrs.Attribute,
    value: models.EncoderDecoderModel,
) -> None:
  """Verify if T5X model can be used in quantized settings."""
  del instance, attribute
  if not isinstance(value.output_vocabulary, vocabs.QuantizedVocabulary):
    raise ValueError(
        f"Model {value} of type {type(value)} must use quantized output vocab."
    )
  # TODO: Add check to see if decoder_fn (all variants:
  # original, functools.partial, etc.) is temperature_sample.


# eq=False allows hashing and 'self' as hashable static arg for jitting.
@attrs.define(kw_only=True, eq=False)
class QuantizedInferencer:
  """Low-level T5X-based class for performing inference in Jax-array space.

  NOTE: Most functions in this class assume the jnp.arrays are correct values
  without validation.

  Otherwise a misalignment will cause e.g. the model to "want" to predict a
  separator token, but be forced to predict a parameter token. The code will
  run, but the actual algorithm will be poor.

  Input preprocessing (vz.Study -> Batch) and output postprocessing (jnp.array
  -> vz.Suggestion) should be performed outside of this class.
  """

  # Shape comments: num_params = `P`, num_metrics = `M`.
  problem: vz.ProblemStatement = attrs.field()

  # Expected to use `temperature_sample` as `decode_fn``.
  model: models.EncoderDecoderModel = attrs.field(
      validator=_quantized_model_validator
  )
  weights: models.PyTree = attrs.field()

  num_suggest_samples: int = attrs.field()
  num_measurement_samples: int = attrs.field()

  # Created after init.
  _suggestion_restrictor: _SuggestionLogitRestrictor = attrs.field(init=False)
  _measurement_restrictor: _MeasurementLogitRestrictor = attrs.field(init=False)

  def __attrs_post_init__(self):
    self._suggestion_restrictor = _SuggestionLogitRestrictor(
        self.problem.search_space, self.vocab
    )
    self._measurement_restrictor = _MeasurementLogitRestrictor(
        self.problem.metric_information, self.vocab
    )

  @functools.partial(jax.jit, static_argnames=["self"])
  def sample_suggestion_tokens(
      self,
      history: Mapping[str, jnp.ndarray],
      rng: Optional[jax.Array] = None,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return self._sample_tokens(
        batch=history,
        num_decode_steps=self.problem.search_space.num_parameters(),
        num_samples=self.num_suggest_samples,
        logit_callback_fn=self._suggestion_restrictor,
        rng=rng,
    )

  @functools.partial(jax.jit, static_argnames=["self"])
  def sample_measurement_tokens(
      self,
      history: Mapping[str, jnp.ndarray],
      rng: Optional[jax.Array] = None,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return self._sample_tokens(
        batch=history,
        num_decode_steps=len(self.problem.metric_information),
        num_samples=self.num_measurement_samples,
        logit_callback_fn=self._measurement_restrictor,
        rng=rng,
    )

  def _sample_tokens(
      self,
      batch: Mapping[str, jnp.ndarray],
      num_decode_steps: int,
      num_samples: int,
      logit_callback_fn: decoding.IndexLogitRestrictor,
      rng: Optional[jax.Array] = None,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Temperature samples tokens.

    The `predict_batch_with_aux` works in the following: if e.g. our search
    space has 2 parameters and `index` = 3:

    decoder_input_tokens: [0, #, #, #, 0, 0, 0, ...]  # Starts w/ 0.
    output: [#, #, #, p1, p2, 0, ...]  # Same shape as input.

    where `p1, p2` are the 2 new generated parameter tokens. `0` corresponds
    to BOS/blank tokens, and `#` correspond to non-BOS tokens.

    [p1, p2] will also be conveniently returned via slicing starting from
    `index`.

    Args:
      batch: Should be features representing (batched) study histories, which
        immediately prompts suggestion. Sequence tensors should be of shape [B,
        L].
      num_decode_steps: Length of slice (e.g. number of parameters or number of
        metrics).
      num_samples: Number of samples to generate in beam dimension.
      logit_callback_fn: Restricts possible token to generate based on index.
      rng:

    Returns:
      1. Sampled quantized suggestion tokens. Shape [B, S, P].
      2. Full sequence (history + new suggestion tokens). Shape [B, S, L].
    """
    index = seq_utils.count_not_from(batch["decoder_input_tokens"])  # [B]

    # Setup decoding args and perform T5X decoding.
    decoder_params = {
        "max_decode_steps": num_decode_steps,
        "logit_callback_fn": functools.partial(logit_callback_fn, shift=index),
    }

    full, _ = self.model.predict_batch_with_aux(
        params=self.weights,
        batch=batch,
        rng=rng,
        decoder_params=decoder_params,
        return_all_decodes=True,
        num_decodes=num_samples,
        prompt_with_targets=True,
    )  # [B, S, L]

    # Also obtain generated slice.
    index = jnp.broadcast_to(index, full.shape[:-1])  # [B, S]
    sliced = seq_utils.dynamic_slice_broadcast(full, index, num_decode_steps)

    return sliced, full

  # TODO: Fix non-scalar index case.
  def regress(
      self,
      history: Mapping[str, jnp.ndarray],
      index: Optional[jnp.ndarray] = None,
  ) -> Tuple[tfd.Distribution, Float[Array, "*B L V"]]:
    """Computes objective distribution at a specific index of the sequence.

    To be relatively agnostic to the type of serialization, this function
    assumes nothing about the batch input. Example: if `index=3`, we will obtain
    the logits of `?` from the following (where `#` can be any token) output:

    [#, #, #, ?, #, #, #, ...]

    Args:
      history: Should be features representing parallel length-aligned studies,
        which immediately prompts objective function regression. Shape [..., L].
      index: Indices for computing quantized logits. If None, picks rightmost
        non-BOS token. Shape [...] or scalar.

    Returns:
      1. Distribution using objective function logits at index. Shape [...].
      2. Original full logits. Shape [..., L, V].
    """

    studies = history
    if index is None:
      index = seq_utils.count_not_from(studies["decoder_input_tokens"]) - 1

    # pylint: disable=protected-access
    full_logits: Float[Array, "*B L V"] = self.model._compute_logits(
        params=self.weights, batch=studies
    )
    # pylint: enable=protected-access

    # Obtain logit slice and squeeze slice axis
    logit_slice: Float[Array, "*B 1 V"] = jax.lax.dynamic_slice_in_dim(
        full_logits, index, 1, axis=-2
    )
    logit_slice: Float[Array, "*B V"] = jnp.squeeze(logit_slice, axis=-2)

    # Restrict logit slice to only quantized and create distribution.
    quant_begin = self.vocab.quantization_vocab_index
    quant_end = quant_begin + self.vocab.num_quantization_bins
    logit_slice: Float[Array, "*B Q"] = logit_slice[..., quant_begin:quant_end]
    dist = tfd.FiniteDiscrete(
        jnp.arange(self.vocab.num_quantization_bins),
        logits=logit_slice,
    )

    return dist, full_logits

  @property
  def vocab(self) -> vocabs.QuantizedVocabulary:
    return typing.cast(vocabs.QuantizedVocabulary, self.model.output_vocabulary)


# TODO: Implement multitrial sampler for long-range planning.
