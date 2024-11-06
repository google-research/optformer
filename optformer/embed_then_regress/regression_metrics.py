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

"""Compute regression-related metrics."""

import jax
import jax.numpy as jnp
import jaxtyping as jt

EPS = 1e-7
Scalar = jt.Float[jax.Array, '']


def masked_mean(
    values: jt.Float[jax.Array, 'B L'],
    target_mask: jt.Bool[jax.Array, 'B L'],
) -> jt.Float[jax.Array, 'B']:
  """Calculate means, only considering mask=True values."""
  values = values * target_mask  # [B, L]
  return jnp.sum(values, axis=1) / jnp.sum(target_mask, axis=1)  # [B]


def pointwise_mse(
    mu: jt.Float[jax.Array, 'B L'],
    ys: jt.Float[jax.Array, 'B L'],
    target_mask: jt.Bool[jax.Array, 'B L'],
) -> Scalar:
  """Pointwise MSE."""
  squared_error = jnp.square(ys - mu)
  mse = masked_mean(squared_error, target_mask)
  return jnp.mean(mse)  # [B] -> Scalar


def pointwise_r2(
    mu: jt.Float[jax.Array, 'B L'],
    ys: jt.Float[jax.Array, 'B L'],
    target_mask: jt.Bool[jax.Array, 'B L'],
) -> Scalar:
  """Pointwise R2."""
  # Calculate centered values.

  mu_mean = jnp.expand_dims(masked_mean(mu, target_mask), axis=-1)  # [B, 1]
  ys_mean = jnp.expand_dims(masked_mean(ys, target_mask), axis=-1)  # [B, 1]

  mu_centered = (mu - mu_mean) * target_mask  # [B, L]
  ys_centered = (ys - ys_mean) * target_mask  # [B, L]

  # Calculate covariance and standard deviations.
  covariance = jnp.sum(mu_centered * ys_centered, axis=1)  # [B]
  std_mu = jnp.sqrt(jnp.sum(mu_centered**2, axis=1))  # [B]
  std_ys = jnp.sqrt(jnp.sum(ys_centered**2, axis=1))  # [B]

  # Calculate correlation coefficient
  corrcoef = covariance / (std_mu * std_ys + EPS)  # [B]
  return jnp.mean(corrcoef**2)  # [B] -> Scalar


def default_metrics(
    mu: jt.Float[jax.Array, 'B L'],
    ys: jt.Float[jax.Array, 'B L'],
    target_mask: jt.Bool[jax.Array, 'B L'],
) -> dict[str, Scalar]:
  """Default metrics."""
  return {
      'pointwise_mse': pointwise_mse(mu, ys, target_mask),
      'pointwise_r2': pointwise_r2(mu, ys, target_mask),
  }
