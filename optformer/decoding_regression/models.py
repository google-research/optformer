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

"""Regression models."""

import jaxtyping as jt
import numpy as np
from optformer.decoding_regression import vocabs
import scipy as sp
import tensorflow as tf

keras = tf.keras
NEG_INF = -1e7


def sample_from_probs(prob_vector: jt.Float[jt.Array, 'P']):
  return np.random.choice(range(len(prob_vector)), p=prob_vector)


# Vectorize the function to apply it over the batch dimension
vectorized_sample = np.vectorize(sample_from_probs, signature='(n)->()')


class AttentionDecoder(keras.Model):
  """Attention-based decoder."""

  def __init__(
      self,
      encoder: keras.Model,
      vocab: vocabs.FloatVocab,
      units: int = 128,
      num_layers: int = 1,
      num_heads: int = 1,
      dropout: float = 0.0,
  ):
    super().__init__()
    self._encoder = encoder
    self._vocab = vocab
    self._units = units  # D
    self._num_layers = num_layers
    self._num_heads = num_heads
    self._dropout = dropout

    self._enc_proj = keras.layers.Dense(self._units)
    self._token_emb = keras.layers.Embedding(
        input_dim=self._vocab.size, output_dim=units
    )
    self._pos_emb = keras.layers.Embedding(
        input_dim=self._vocab.token_length, output_dim=units
    )

    self._transformer_layers = []
    for _ in range(num_layers):
      attn = keras.layers.MultiHeadAttention(
          num_heads=self._num_heads,
          key_dim=self._units,
          value_dim=self._units,
          dropout=self._dropout,
      )
      ffn = keras.Sequential([
          keras.layers.Dense(units, activation='relu'),
          keras.layers.Dense(units),
      ])
      layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
      layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
      self._transformer_layers.append((attn, ffn, layernorm1, layernorm2))

    self._output_proj = keras.layers.Dense(self._vocab.size)

  def call(self, inputs):  # pytype:disable=signature-mismatch
    """Returns probabilities over the next token."""

    encoder_input, decoded_ids = inputs  # [B, F] and [B, L]
    encoded = self._encoder(encoder_input)  # [B, F']

    encoded = self._enc_proj(encoded)  # [B, D]
    decoded = self._token_emb(decoded_ids)  # [B, L-1, D]

    # Add it to the decoded sequence.
    encoded = tf.expand_dims(encoded, axis=1)  # [B, 1, D]
    seq = tf.concat([encoded, decoded], axis=1)  # [B, L, D]

    # Add positional embeddings to the decoded sequence
    positions = tf.range(start=0, limit=tf.shape(seq)[1], delta=1)
    pos_embeddings = self._pos_emb(positions)
    seq += pos_embeddings

    # Apply transformer layers.
    for attn, ffn, layernorm1, layernorm2 in self._transformer_layers:
      out = attn(seq, seq, use_causal_mask=True)
      out = layernorm1(out + seq)
      ffn_out = ffn(out)
      seq = layernorm2(out + ffn_out)  # Update seq for the next layer

    # Project to logits.
    return self._output_proj(seq)  # [B, L, V]

  def decode(
      self,
      encoder_input: jt.Float[jt.Array, 'B F'],
      temperature: float = 1.0,
  ) -> jt.Float[jt.Array, 'B']:
    """Performs temperature sampling."""
    # pylint: disable=invalid-name
    B = encoder_input.shape[0]
    token_ids = -1 * np.ones((B, self._vocab.token_length), dtype=int)

    for i in range(self._vocab.token_length):
      logits = self.predict([encoder_input, token_ids[:, :i]])
      current_logits = logits[:, -1, :]
      # Logit restriction
      current_logits[:, ~self._vocab.logit_mask(i)] = NEG_INF

      # [B, V]
      probs = sp.special.softmax(temperature * current_logits, axis=-1)

      # Sample tokens.
      sampled_ids = vectorized_sample(probs)
      token_ids[:, i] = np.array(sampled_ids)

    return np.array([self._vocab.from_int(toks) for toks in token_ids])

  def greedy_decode(
      self, encoder_input: jt.Float[jt.Array, 'B F']
  ) -> jt.Float[jt.Array, 'B']:
    """Performs greedy decoding."""
    # pylint: disable=invalid-name
    B = encoder_input.shape[0]
    token_ids = -1 * np.ones((B, self._vocab.token_length), dtype=int)

    for i in range(self._vocab.token_length):
      logits = self.predict([encoder_input, token_ids[:, :i]])
      current_logits = logits[:, -1, :]
      # Logit restriction
      current_logits[:, ~self._vocab.logit_mask(i)] = NEG_INF

      # Pick argmax instead.
      sampled_ids = np.argmax(current_logits, axis=-1)  # [B]
      token_ids[:, i] = np.array(sampled_ids)

    return np.array([self._vocab.from_int(toks) for toks in token_ids])


def weighted_sparse_categorical_crossentropy(labels, logits, weights=None):
  """Weighted version of sparse categorical cross entropy."""

  ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
  if weights is None:
    return ce

  weights = tf.cast(weights, dtype=tf.float32)
  normalized_weights = (
      labels.shape[-1] * weights / tf.reduce_sum(weights, axis=-1)
  )
  return ce * normalized_weights
