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

"""OmniPred predictor and sampler."""

import functools
import typing
from typing import Callable, Generic, Mapping, Tuple, TypeVar
import attrs
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
from optformer.common.data import datasets
from optformer.omnipred import vocabs
from optformer.t5x import decoding
from optformer.t5x import inference


@attrs.define(auto_attribs=False)
class _OmniPredLogitRestrictor(decoding.IndexLogitRestrictor):
  """Forces decoding of only necessary float tokens."""

  vocab: vocabs.FloatMetricVocabulary = attrs.field(init=True)
  _logits_masks: Float[Array, 'L V'] = attrs.field(init=False)

  def __attrs_post_init__(self):
    logits_masks = np.zeros((self.vocab.decode_length, self.vocab.vocab_size))

    # Deal with initial token always used.
    logits_masks[0, self.vocab.initial_token_id] = 1.0

    # Only turns on index-dependent custom tokens representing floats.
    for i in range(self.vocab.deserializer.num_tokens_per_obj):  # pytype:disable=attribute-error
      tokens_used = self.vocab.deserializer.tokens_used(i)  # pytype:disable=attribute-error
      ids = [self.vocab.extra_token_id(t) for t in tokens_used]
      logits_masks[i + 1, ids] = 1.0

    self._logits_masks = jnp.array(logits_masks)

  def logit_mask(self, index: Int[Array, 'BS']) -> Float[Array, 'BS V']:
    return self._logits_masks[index]


_T = TypeVar('_T')


@attrs.define
class OmniPred(Generic[_T]):
  """Predicts tokens representing a float."""

  inference_config: inference.InferenceConfig[_T] = attrs.field()

  num_samples: int = attrs.field(default=64, kw_only=True)
  max_inputs_length: int = attrs.field(default=2048, kw_only=True)

  rng: jax.Array = attrs.field(
      kw_only=True, factory=lambda: jax.random.PRNGKey(42)
  )

  _jit_predict_batch_with_aux = attrs.field(init=False)
  _dataset_fn = attrs.field(init=False)

  def __attrs_post_init__(self):
    # Setup logit restriction.
    logit_callback_fn = _OmniPredLogitRestrictor(self._vocab)
    predict_batch_with_aux = functools.partial(
        self.inference_config.model.predict_batch_with_aux,
        decoder_params={
            'max_decode_steps': self._vocab.decode_length,
            'logit_callback_fn': logit_callback_fn,
        },
    )

    # jit `EncoderDecoderModel` functions.
    self._jit_predict_batch_with_aux = jax.jit(
        predict_batch_with_aux,
        static_argnames=['return_all_decodes', 'num_decodes'],
    )

    self._dataset_fn = self.inference_config.get_dataset_fn(
        self.max_inputs_length, self._vocab.decode_length
    )

  def _sample_tokens(
      self, batch: Mapping[str, jnp.ndarray]
  ) -> Tuple[Int[Array, 'B S L'], Float[Array, 'B S']]:
    """Returns token samples and their logprobs."""
    self.rng, subkey = jax.random.split(self.rng)
    sampled_tokens, aux = self._jit_predict_batch_with_aux(
        params=self.inference_config.train_state.params,
        batch=batch,
        rng=subkey,
        return_all_decodes=True,
        num_decodes=self.num_samples,
    )
    return sampled_tokens, aux['scores']

  def predict(
      self, prompt: _T, aggregator_fn: Callable[[np.ndarray], float] = np.median
  ) -> float:
    """Give a pointwise prediction, aggregated from samples."""
    ds = self._dataset_fn([prompt]).batch(1)
    batch = next(ds.as_numpy_iterator())

    toks, _ = self._sample_tokens(batch)
    toks = jnp.squeeze(toks, axis=0)  # [S, L]

    fs = np.array([self._vocab.decode_to_object(t) for t in toks])
    return aggregator_fn(fs)

  @property
  def _vocab(self) -> vocabs.FloatMetricVocabulary:
    vocab = self.inference_config.model.output_vocabulary
    return typing.cast(vocabs.FloatMetricVocabulary, vocab)

  @property
  def dataset_fn(self) -> datasets.E2EInferenceDatasetFn[_T]:
    return self._dataset_fn
