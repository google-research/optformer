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

"""Subclasses derived from t5x models.

Overrides the feature converter class. Also, the loss function and metrics
computation methods are overridden to compute the metrics separately for
parameter and function predictions.
"""

import functools
from typing import Any, Callable, Mapping, MutableMapping, Optional, Tuple, Union

import clu.metrics as clu_metrics
from flax import linen as nn
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
from optformer.t5x import decoding as vizier_decoding
from optformer.t5x import feature_converters
from optformer.t5x.utils import TargetType
import seqio
from t5x import decoding
from t5x import losses
from t5x import models
from t5x import optimizers

MetricsMap = models.MetricsMap
PyTreeDef = models.PyTreeDef
DecodeFnCallable = models.DecodeFnCallable
DecodingState = vizier_decoding.SamplingLoopState
_NoValueSentinel = object()


def logit_mask_callback_fn(logits: jnp.ndarray,
                           state: DecodingState,
                           logits_mask: jnp.ndarray,
                           beam_size: int = 0,
                           flatten_batch_dim: bool = False) -> jnp.ndarray:
  """Apply a mask to the given index of the provided logits.

  Args:
    logits: [batch_size, vocabulary_size] array of the predicted logits if
      flatten_dim is true, else we have [batch_size, beam_size, vocabulary_size]
      as the array dimensions.
    state: decoding loop state.
    logits_mask: [batch_size, sequence_len, vocabulary_size] array with -inf at
      positions to be masked out.
    beam_size: size of number of decode samples.
    flatten_batch_dim: whether or not to flatten the beam dimension.
  Returns:
    Masked logits whose shape depends on the shape of logits as described above.
  """
  # The logit sequence index to apply the mask to.
  current_idx = state.mask_idx

  cur_step_mask = jnp.asarray(
      logits_mask[:, None, current_idx, :], dtype=logits.dtype)
  cur_step_mask = jnp.repeat(cur_step_mask, beam_size, axis=1)
  if flatten_batch_dim:
    cur_step_mask = jnp.reshape(cur_step_mask, (-1, cur_step_mask.shape[-1]))
  return logits + cur_step_mask


class VizierEncoderDecoderModel(models.EncoderDecoderModel):
  """Subclass of EncoderDecoderModel for Vizier tasks."""

  FEATURE_CONVERTER_CLS = feature_converters.VizierEncDecFeatureConverter

  def __init__(
      self,
      module: nn.Module,
      input_vocabulary: seqio.Vocabulary,
      output_vocabulary: seqio.Vocabulary,
      optimizer_def: optimizers.OptimizerDefType,
      decode_fn: DecodeFnCallable = vizier_decoding.temperature_sample,
      feature_converter_cls: Optional[Callable[...,
                                               seqio.FeatureConverter]] = None,
      label_smoothing: float = 0.0,
      z_loss: float = 0.0,
      loss_normalizing_factor: Optional[float] = None,
  ):
    super().__init__(
        module=module,
        input_vocabulary=input_vocabulary,
        output_vocabulary=output_vocabulary,
        optimizer_def=optimizer_def,
        decode_fn=decode_fn,
        feature_converter_cls=feature_converter_cls,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        loss_normalizing_factor=loss_normalizing_factor,
    )

  def loss_fn(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.random.KeyArray],
      label_smoothing: Optional[float] = None,
      z_loss: Optional[float] = None,
      loss_normalizing_factor: Union[Optional[float],
                                     object] = _NoValueSentinel,
  ) -> Tuple[jnp.ndarray, MetricsMap]:
    return loss_fn_with_additional_metrics(
        model=self,
        params=params,
        batch=batch,
        dropout_rng=dropout_rng,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        loss_normalizing_factor=loss_normalizing_factor)

  def predict_batch_with_aux(
      self,
      params: PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      rng: Optional[jax.random.KeyArray] = None,
      decoder_params: Optional[MutableMapping[str, Any]] = None,
      return_all_decodes: bool = False,
      num_decodes: int = 1,
      prompt_with_targets: bool = False
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Predict with fast decoding beam search on a batch.

    Here we refer to "parameters" for values that can be compiled into the
    model dynamically, as opposed to static configuration settings that require
    a recompile. For example, the model weights and the decoder brevity-penalty
    are parameters and can be modified without requiring a recompile. The number
    of layers, the batch size and the decoder beam size are configuration
    options that require recompilation if changed.

    This method can be used with a customizable decoding function as long as it
    follows the signature of `DecodeFnCallable`. In order to provide a unified
    interface for the decoding functions, we use a generic names. For example, a
    beam size is a concept unique to beam search. Conceptually, it corresponds
    to the number of sequences returned by the beam search.  Therefore, the
    generic argument `num_decodes` corresponds to the beam size if
    `self._decode_fn` is a beam search. For temperature sampling, `num_decodes`
    corresponds to the number of independent sequences to be sampled. Typically
    `num_decodes = 1` is used for temperature sampling.

    If `return_all_decodes = True`, the return tuple contains the predictions
    with a shape [batch, num_decodes, max_decode_len] and the scores (i.e., log
    probability of the generated sequence) with a shape [batch, num_decodes].

    If `return_all_decodes = False`, the return tuple contains the predictions
    with a shape [batch, max_decode_len] and the scores with a shape [batch].

    `decoder_params` can be used to pass dynamic configurations to
    `self.decode_fn`. An example usage is to pass different random seed (i.e.,
    `jax.random.PRNGKey(seed)` with different `seed` value). This can be done by
    setting `decoder_params['decode_rng'] = jax.random.PRNGKey(seed)`.

    If `prompt_with_targets = True`, then `decoder_prompt_inputs` is initialized
    from the batch's `decoder_input_tokens`. The EOS is stripped to avoid
    decoding to stop after the prompt by matching to `output_vocabulary.eos_id`.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      rng: an optional RNG key to use during prediction, which is passed as
        'decode_rng' to the decoding function.
      decoder_params: additional (model-independent) parameters for the decoder.
      return_all_decodes: whether to return the entire beam or just the top-1.
      num_decodes: the number of beams to use in beam search.
      prompt_with_targets: Whether the force decode decoder_inputs.

    Returns:
      A tuple containing:
        the batch of predictions, with the entire beam if requested
        an auxiliary dictionary of decoder scores
    """

    # Prepare zeroed-out autoregressive cache.
    # [batch, input_len]
    inputs = batch['encoder_input_tokens']
    # [batch, target_len]
    target_shape = batch['decoder_input_tokens'].shape
    target_type = batch['decoder_input_tokens'].dtype
    _, variables_with_cache = self.module.apply(
        {'params': params},
        jnp.ones(inputs.shape, inputs.dtype),
        jnp.ones(target_shape, target_type),
        jnp.ones(target_shape, target_type),
        decode=True,
        enable_dropout=False,
        mutable=['cache'])

    cache = variables_with_cache['cache']

    # Prepare transformer fast-decoder call for beam search: for beam search, we
    # need to set up our decoder model to handle a batch size equal to
    # batch_size * num_decodes, where each batch item's data is expanded
    # in-place rather than tiled.
    # i.e. if we denote each batch element subtensor as el[n]:
    # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
    # [batch * num_decodes, input_len, emb_dim]
    encoded = self.module.apply({'params': params},
                                inputs,
                                enable_dropout=False,
                                method=self.module.encode)
    encoded_inputs = decoding.flat_batch_beam_expand(encoded, num_decodes)

    # [batch * num_decodes, input_len]
    raw_inputs = decoding.flat_batch_beam_expand(inputs, num_decodes)
    tokens_ids_to_logits = functools.partial(
        self._compute_logits_from_slice,
        params=params,
        encoded_inputs=encoded_inputs,
        raw_inputs=raw_inputs,
        max_decode_length=target_shape[1])

    if decoder_params is None:
      decoder_params = {}
    if rng is not None:
      if decoder_params.get('decode_rng') is not None:
        raise ValueError(
            f'Got RNG both from the `rng` argument ({rng}) and '
            f"`decoder_params['decode_rng']` ({decoder_params['decode_rng']}). "
            'Please specify one or the other.')
      decoder_params['decode_rng'] = rng

    # `decoder_prompt_inputs` is initialized from the batch's
    # `decoder_input_tokens`. The EOS is stripped to avoid decoding to stop
    # after the prompt by matching to `output_vocabulary.eos_id`.
    # These inputs are ignored by the beam search decode fn.
    if prompt_with_targets:
      decoder_prompt_inputs = batch['decoder_input_tokens']
      decoder_prompt_inputs = decoder_prompt_inputs * (
          decoder_prompt_inputs != self.output_vocabulary.eos_id)
    else:
      decoder_prompt_inputs = jnp.zeros_like(batch['decoder_input_tokens'])

    # Setup the logits mask
    if batch['logits_mask'] is not None:
      local_logit_callback = functools.partial(
          logit_mask_callback_fn,
          logits_mask=batch['logits_mask'],
          beam_size=num_decodes,
          flatten_batch_dim=True)
      decoder_params['logit_callback_fn'] = local_logit_callback

    # TODO: rename the returned value names to more generic ones.
    # Using the above-defined single-step decoder function, run a
    # beam search over possible sequences given input encoding.
    # decodes: [batch, num_decodes, max_decode_len + 1]
    # scores: [batch, num_decodes]
    scanned = hasattr(self.module, 'scan_layers') and self.module.scan_layers
    decodes, scores = self._decode_fn(
        inputs=decoder_prompt_inputs,
        cache=cache,
        tokens_to_logits=tokens_ids_to_logits,
        eos_id=self.output_vocabulary.eos_id,
        num_decodes=num_decodes,
        cache_offset=1 if scanned else 0,
        **decoder_params)

    # Beam search returns [n_batch, n_beam, n_length] with beam dimension sorted
    # in increasing order of log-probability.
    # Return the highest scoring beam sequence.
    if return_all_decodes:
      return decodes, {'scores': scores}
    else:
      return decodes[:, -1, :], {'scores': scores[:, -1]}


def loss_fn_with_additional_metrics(
    model,
    params: PyTreeDef,
    batch: Mapping[str, jnp.ndarray],
    dropout_rng: Optional[jnp.ndarray],
    label_smoothing: Optional[float],
    z_loss: Optional[float],
    loss_normalizing_factor: Union[Optional[float],
                                   object]
) -> Tuple[jnp.ndarray, MetricsMap]:
  """Shared loss function implementation by both Vizier architectures."""
  # pylint: disable=protected-access

  # Default these to the constructor values. In the future, they may be
  # removed as parameters for `loss_fn`.
  label_smoothing = (
      model._label_smoothing if label_smoothing is None else label_smoothing)
  z_loss = model._z_loss if z_loss is None else z_loss
  if loss_normalizing_factor is _NoValueSentinel:
    loss_normalizing_factor = model._loss_normalizing_factor

  logits = model._compute_logits(params, batch, dropout_rng)

  weights = batch.get('decoder_loss_weights', None)
  targets_types = batch.get('decoder_target_types', None)
  if targets_types is None:
    parameter_weights = None
    function_weights = None
  else:
    parameter_weights = (
        targets_types == TargetType.PARAMETER.value).astype(jnp.int32)
    function_weights = (
        targets_types == TargetType.FUNCTION.value).astype(jnp.int32)
    if weights is not None:
      parameter_weights = parameter_weights * weights
      function_weights = function_weights * weights

  (loss, z_loss, _, p_cross_ent_loss, f_cross_ent_loss
   ) = compute_weighted_cross_entropy(
       logits,
       targets=batch['decoder_target_tokens'],
       weights=weights,
       parameter_weights=parameter_weights,
       function_weights=function_weights,
       label_smoothing=label_smoothing,
       z_loss=z_loss,
       loss_normalizing_factor=loss_normalizing_factor)

  metrics = compute_base_metrics(
      logits=logits,
      targets=batch['decoder_target_tokens'],
      mask=weights,
      loss=loss,
      parameter_cross_ent_loss=p_cross_ent_loss,
      parameter_mask=parameter_weights,
      function_cross_ent_loss=f_cross_ent_loss,
      function_mask=function_weights,
      z_loss=z_loss)
  return loss, metrics


def compute_weighted_cross_entropy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    parameter_weights: Optional[jnp.ndarray] = None,
    function_weights: Optional[jnp.ndarray] = None,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    loss_normalizing_factor: Optional[float] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Compute weighted cross entropy and entropy for log probs and targets.

  Compute separate cross entropy loss for parameter and function predictions.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch, length].
   parameter_weights: None or array of shape [batch, length].
   function_weights: None or array of shape [batch, length].
   label_smoothing: label smoothing constant, used to determine the on and off
     values.
   z_loss: coefficient for auxiliary z-loss loss term.
   loss_normalizing_factor: Constant to divide loss by. If not specified, loss
     will not be normalized. Intended for backward compatibility with T5-MTF
     training. Should not normally be used.

  Returns:
    Tuple of scalar loss, z_loss, weight sum, parameter cross entropy loss,
      and function cross entropy loss.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  vocab_size = logits.shape[-1]
  confidence = 1.0 - label_smoothing
  low_confidence = (1.0 - confidence) / (vocab_size - 1)
  normalizing_constant = -(
      confidence * jnp.log(confidence) +
      (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20))
  soft_targets = common_utils.onehot(
      targets, vocab_size, on_value=confidence, off_value=low_confidence)
  total_loss, total_z_loss = losses.cross_entropy_with_logits(
      logits, soft_targets, z_loss=z_loss)
  total_loss = total_loss - normalizing_constant

  cross_ent_loss = total_loss - total_z_loss

  weight_sum = np.prod(targets.shape)
  if weights is not None:
    total_loss = total_loss * weights
    total_z_loss = total_z_loss * weights
    weight_sum = jnp.sum(weights)

  if parameter_weights is not None:
    p_cross_ent_loss = cross_ent_loss * parameter_weights
    p_cross_ent_loss = jnp.sum(p_cross_ent_loss)
  else:
    p_cross_ent_loss = jnp.array(0., dtype=cross_ent_loss.dtype)

  if function_weights is not None:
    f_cross_ent_loss = cross_ent_loss * function_weights
    f_cross_ent_loss = jnp.sum(f_cross_ent_loss)
  else:
    f_cross_ent_loss = jnp.array(0., dtype=cross_ent_loss.dtype)

  # By default, we do not normalize loss based on anything.
  # We don't normalize based on batch size because the optimizers we use are
  # pretty much scale invariant, so this simplifies things.
  # We don't normalize based on number of non-padding tokens in order to treat
  # each token as equally important regardless of sequence length.
  if loss_normalizing_factor:
    total_loss /= loss_normalizing_factor
    total_z_loss /= loss_normalizing_factor
  return (jnp.sum(total_loss), jnp.sum(total_z_loss), weight_sum,
          p_cross_ent_loss, f_cross_ent_loss)


def compute_base_metrics(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
    loss: jnp.ndarray,
    parameter_cross_ent_loss: jnp.ndarray,
    parameter_mask: jnp.ndarray,
    function_cross_ent_loss: jnp.ndarray,
    function_mask: jnp.ndarray,
    z_loss: Optional[jnp.ndarray] = None,
) -> MetricsMap:
  """Compute summary metrics."""
  metrics = models.compute_base_metrics(
      logits=logits,
      targets=targets,
      mask=mask,
      loss=loss,
      z_loss=z_loss)

  # Add additional metrics.
  if z_loss is not None:
    nonpadding_tokens = np.prod(targets.size)
    if mask is not None:
      nonpadding_tokens = jnp.sum(mask)
    metrics.update({
        'cross_ent_loss_per_nonpadding_target_token':
            clu_metrics.Average(
                total=jnp.sum(loss - z_loss),
                count=nonpadding_tokens),
    })

  if parameter_mask is not None:
    parameter_tokens = jnp.maximum(jnp.sum(parameter_mask), 1e-8)
  else:
    parameter_tokens = jnp.array(1e-8, dtype=parameter_cross_ent_loss.dtype)
  if function_mask is not None:
    function_tokens = jnp.maximum(jnp.sum(function_mask), 1e-8)
  else:
    function_tokens = jnp.array(1e-8, dtype=function_cross_ent_loss.dtype)
  metrics.update({
      'parameter_cross_ent_loss_per_token':
          clu_metrics.Average(
              total=parameter_cross_ent_loss,
              count=parameter_tokens),
      'parameter_accuracy':
          clu_metrics.Accuracy.from_model_output(
              logits=logits,
              labels=targets.astype(jnp.int32),
              mask=parameter_mask),
      'function_cross_ent_loss_per_token':
          clu_metrics.Average(
              total=function_cross_ent_loss,
              count=function_tokens),
      'function_accuracy':
          clu_metrics.Accuracy.from_model_output(
              logits=logits,
              labels=targets.astype(jnp.int32),
              mask=function_mask),
  })
  return metrics
