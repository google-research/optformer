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

"""Transformer model for ICL regression."""

import functools
from typing import Callable
from flax import linen as nn
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np

Array = jnp.ndarray | np.ndarray

# NOTE: Lower initialization is **extremely** important. We need to start off
# with reasonably scaled output distribution and prevent exploding gradients /
# bad initial loss values. Also needs to be consistent across entire model in
# order to use the same learning rate.
default_kernel_init = nn.initializers.truncated_normal(stddev=0.02)
Dense = functools.partial(nn.Dense, kernel_init=default_kernel_init)
EPS = 1e-7
AnyTensor = jt.Float[jax.Array, '*A']


class Block(nn.Module):
  """Standard attention block with customizable mask."""

  d_model: int  # D
  num_heads: int  # H
  hidden_dim: int  # F
  dropout_rate: float

  def setup(self):
    self.pre_attn_norm = nn.LayerNorm()
    self.attn = nn.SelfAttention(
        num_heads=self.num_heads,
        qkv_features=self.d_model,
        dropout_rate=self.dropout_rate,
        kernel_init=default_kernel_init,
        out_kernel_init=default_kernel_init,
    )

    self.pre_ffw_norm = nn.LayerNorm()
    self.ffw = nn.Sequential(
        [Dense(self.hidden_dim), nn.relu, Dense(self.d_model)]
    )

    self.dropout = nn.Dropout(rate=self.dropout_rate)

  def __call__(
      self,
      x: jt.Float[jax.Array, 'B* L D'],
      mask: jt.Float[jax.Array, 'B* H QL KVL'] | None = None,
      deterministic: bool | None = None,
      rng: jax.Array | None = None,
  ) -> jt.Float[jax.Array, 'B* L D']:
    # Pre-attention normalization
    norm1 = self.pre_attn_norm(x)
    # Self-attention layer
    attn = self.attn(
        norm1, mask=mask, deterministic=deterministic, dropout_rng=rng
    )
    x = x + attn  # Residual connection
    # Pre-feed-forward normalization
    norm2 = self.pre_ffw_norm(x)
    # Feed-forward layer
    ff = self.ffw(norm2)
    x = x + ff  # Residual connection

    # Optionally, apply dropout
    if self.dropout_rate > 0.0:
      x = self.dropout(x, deterministic, rng)

    return x


class ICLTransformer(nn.Module):
  """ICL Transformer model for regression."""

  d_model: int  # D
  ffw_dim_ratio: int  # F // D
  nhead: int  # H
  dropout: float
  num_layers: int
  std_transform_fn: Callable[[AnyTensor], AnyTensor]
  embedder_factory: Callable[[], nn.Module]  # __call__: [B, T] -> [B, D]

  def setup(self):
    # For embedding x and metadata tokens.
    self.embedder = self.embedder_factory()

    # X, Y, and concatenated X,Y embedders.
    self.x_proj = nn.Sequential(
        [Dense(self.d_model), nn.relu, Dense(self.d_model)]
    )
    self.y_proj = nn.Sequential(
        [Dense(self.d_model), nn.relu, Dense(self.d_model)]
    )
    self.xy_proj = nn.Sequential(
        [Dense(self.d_model * 2), nn.relu, Dense(self.d_model)]
    )

    # Attention blocks with customizable masks.
    self.encoder_layers = [
        Block(
            d_model=self.d_model,
            num_heads=self.nhead,
            dropout_rate=self.dropout,
            hidden_dim=int(self.d_model * self.ffw_dim_ratio),
        )
        for _ in range(self.num_layers)
    ]

    # Predict mean and logstd.
    self.mean_logstd_head = nn.Sequential(
        [Dense(self.d_model), nn.relu, Dense(2)]
    )

  def __call__(
      self,
      x: jt.Int[jax.Array, 'B L T'],  # T = number of tokens.
      y: jt.Float[jax.Array, 'B L'],
      metadata: jt.Int[jax.Array, 'B T'],  # Study-level tokenized metadata.
      mask: jt.Bool[jax.Array, 'B L'],
      deterministic: bool | None = None,
      rng: jax.Array | None = None,
  ) -> tuple[jt.Float[jax.Array, 'B L'], jt.Float[jax.Array, 'B L']]:
    # pylint: disable=invalid-name

    B, L, T = x.shape
    x = jnp.reshape(x, (B * L, T))
    x = self.embed(x)  # [B*L, E]
    x = jnp.reshape(x, (B, L, -1))  # [B, L, E]

    metadata = self.embed(metadata)  # [B, E]
    metadata = jnp.expand_dims(metadata, axis=1)  # [B, 1, E]
    metadata = jnp.repeat(metadata, L, axis=1)  # [B, L, E]
    x = jnp.concatenate((x, metadata), axis=-1)  # [B, L, 2E]

    xt_emb = self.x_proj(x)  # [B, L, D]

    # Force 0.0 values for target points using the mask.
    y = y * mask  # [B, L], element-wise multiplication

    y = jnp.expand_dims(y, axis=-1)  # [B, L, 1]
    yt_emb = self.y_proj(y)  # [B, L, D]
    xy_emb = self.xy_proj(jnp.concatenate((xt_emb, yt_emb), axis=-1))

    # Broadcast mask to all heads and additional axis.
    # All tokens attend to context tokens: mask[:, :num_ctx] = True
    # and no token attends to target tokens: mask[:, num_ctx:] = False
    mask = jnp.repeat(jnp.expand_dims(mask, axis=1), L, axis=1)  # [B, L, L]
    mask = jnp.expand_dims(mask, axis=1)  # [B, 1, L, L]

    out = xy_emb
    for layer in self.encoder_layers:
      out = layer(out, mask, deterministic, rng)

    mean, std = jnp.split(self.mean_logstd_head(out), 2, axis=-1)  # [B L 1]
    std = self.std_transform_fn(self.std_transform)(std) + EPS

    mean = jnp.squeeze(mean, axis=-1)
    std = jnp.squeeze(std, axis=-1)
    return mean, std

  @nn.remat  # Reduce memory consumption during backward pass.
  def embed(self, tokens: jt.Int[jax.Array, 'X T']):
    return self.embedder(tokens)
