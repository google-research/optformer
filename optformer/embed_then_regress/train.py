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

"""Training functions."""

import functools
from typing import Mapping
from clu import metric_writers
import flax
import flax.typing as flax_typing
import jax
import jax.numpy as jnp
import numpy as np
import optax
from optformer.embed_then_regress import configs
from optformer.embed_then_regress import icl_transformer
import tensorflow as tf


@flax.struct.dataclass
class TrainState:
  step: int
  params: flax_typing.FrozenVariableDict
  opt_state: optax.OptState


def create_train_state(
    model: icl_transformer.ICLTransformer,
    optimizer: optax.GradientTransformation,
    batch: Mapping[str, np.ndarray],
) -> TrainState:
  params = model.init(jax.random.PRNGKey(0), **batch)
  return TrainState(0, params, optimizer.init(params))


def loss_fn(
    params: flax_typing.FrozenVariableDict,
    model: icl_transformer.ICLTransformer,
    batch: Mapping[str, np.ndarray],
    rng: jax.Array,
) -> tuple[jax.Array, Mapping[str, float]]:
  """Loss function with metrics."""
  # pylint: disable=invalid-name
  mean, std = model.apply(params, rng=rng, **batch)
  loss = -jax.scipy.stats.norm.logpdf(batch['batch_y'], mean, std)  # B, N, 1

  # Only compute loss over target ys. Mask is Bx1xNxN where mask[i, j] = True
  # if # j is a context token and False otherwise.
  B, N, _ = loss.shape
  target_mask = 1 - batch['mask'][:, 0, 0, :].reshape((B, N, 1))
  loss = jnp.sum(loss * target_mask) / jnp.sum(target_mask)
  return loss, {}  # TODO: Get more metrics.


def train_step(
    model: icl_transformer.ICLTransformer,
    optimizer: optax.GradientTransformation,
    train_state: TrainState,
    batch: Mapping[str, np.ndarray],
    rng: jax.Array,
) -> tuple[TrainState, jax.Array, Mapping[str, float]]:
  """Train for a single step."""
  dropout_rng, new_rng = jax.random.split(rng)

  loss_fn_with_rng = functools.partial(
      loss_fn, model=model, batch=batch, rng=dropout_rng
  )

  grad_fn = jax.value_and_grad(loss_fn_with_rng, has_aux=True)
  (_, metrics), grads = grad_fn(train_state.params)

  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grads = jax.lax.pmean(grads, axis_name='batch')
  updates, new_opt_state = optimizer.update(
      grads, train_state.opt_state, train_state.params
  )
  new_params = optax.apply_updates(train_state.params, updates)
  new_state = TrainState(train_state.step + 1, new_params, new_opt_state)

  return new_state, new_rng, metrics


def train(
    model_config: configs.ModelConfig,
    train_config: configs.TrainingConfig,
    train_iter: tf.data.NumpyIterator,
    valid_iter: tf.data.NumpyIterator | None = None,
):
  """Training loop."""
  writer = metric_writers.create_default_writer(
      train_config.workdir, just_logging=jax.process_index() > 0
  )

  optimizer = train_config.create_optimizer()
  model = model_config.create_model()
  rng = jax.random.PRNGKey(train_config.seed)

  f_train_step = functools.partial(train_step, model=model, optimizer=optimizer)
  p_train_step = jax.pmap(f_train_step, axis_name='batch')

  train_state = create_train_state(model, optimizer, next(train_iter))
  for step in range(train_config.max_steps):
    batch = next(train_iter)
    train_state, rng, metrics = p_train_step(
        model, optimizer, train_state, batch, rng
    )
    writer.write_scalars(step, metrics)

    if step % train_config.validation_interval == 0 and valid_iter:
      valid_batch = next(valid_iter)
      _, valid_metrics = loss_fn(train_state.params, model, valid_batch, rng)
      writer.write_scalars(step, valid_metrics)
