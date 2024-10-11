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

from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np

Array = jnp.ndarray | np.ndarray


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
    )

    self.pre_ffw_norm = nn.LayerNorm()
    self.ffw = nn.Sequential(
        [nn.Dense(self.self.hidden_dim), nn.relu, nn.Dense(self.d_model)]
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
  token_embedder: nn.Module  # __call__: [B, T] -> [B, D]
  freeze_embedder: bool = True

  def setup(self):
    # X, Y, and concatenated X,Y embedders.
    self.x_embedder = nn.Sequential(
        [nn.Dense(self.d_model), nn.relu, nn.Dense(self.d_model)]
    )
    self.y_embedder = nn.Sequential(
        [nn.Dense(self.d_model), nn.relu, nn.Dense(self.d_model)]
    )
    self.x_y_embedder = nn.Sequential(
        [nn.Dense(self.d_model * 2), nn.relu, nn.Dense(self.d_model)]
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
        [nn.Dense(self.d_model), nn.relu, nn.Dense(2)]
    )

  def __call__(
      self,
      xt: jt.Int[jax.Array, 'B L T'],  # T = number of tokens.
      yt: jt.Float[jax.Array, 'B L 1'],
      mt: jt.Float[jax.Array, 'B T'],  # Study-level tokenized metadata.
      mask: jt.Float[jax.Array, 'B 1 L L'],  # Broadcasted to all heads.
      deterministic: bool | None = None,
      rng: jax.Array | None = None,
  ) -> tuple[jt.Float[jax.Array, 'B L'], jt.Float[jax.Array, 'B L']]:
    # pylint: disable=invalid-name
    # All tokens attend to context tokens: mask[:, :num_ctx] = True
    # and no token attends to target tokens: mask[:, num_ctx:] = False
    B, L, T = xt.shape
    xt = jnp.reshape(xt, (B * L, T))

    if self.freeze_embedder:
      xt = lax.stop_gradient(self.token_embedder(xt))  # [B*L, E]
    xt = jnp.reshape(xt, (B, L, -1))  # [B, L, E]

    if self.freeze_embedder:
      mt = lax.stop_gradient(self.token_embedder(mt))  # [B, E]
    mt = jnp.expand_dims(mt, axis=1)  # [B, 1, E]
    mt = jnp.repeat(mt, L, axis=1)  # [B, L, E]
    xt = jnp.concatenate((xt, mt), axis=-1)  # [B, L, 2E]

    xt_emb = self.x_embedder(xt)
    yt_emb = self.y_embedder(yt)
    xy_emb = self.x_y_embedder(jnp.concatenate((xt_emb, yt_emb), axis=-1))

    out = xy_emb
    for layer in self.encoder_layers:
      out = layer(out, mask, deterministic, rng)

    mean, log_std = jnp.split(self.mean_logstd_head(out), 2, axis=-1)  # [B L 1]
    std = jnp.exp(log_std)

    mean = jnp.squeeze(mean, axis=-1)
    std = jnp.squeeze(std, axis=-1)
    return mean, std
