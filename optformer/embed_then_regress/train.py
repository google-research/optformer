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
from typing import Any, Mapping
from clu import metric_writers
import flax
import flax.typing as flax_typing
import jax
import jax.numpy as jnp
import numpy as np
import optax
from optformer.common.data import datasets as ds_lib
from optformer.embed_then_regress import checkpointing as ckpt_lib
from optformer.embed_then_regress import configs
from optformer.embed_then_regress import icl_transformer
from optformer.embed_then_regress import metrics as metrics_lib
import tensorflow as tf


Scalar = jnp.ndarray | np.ndarray | float


def multi_gpu() -> bool:
  return jax.device_count() > 1


def replicate(x: Any) -> Any:
  return flax.jax_utils.replicate(x) if multi_gpu() else x


def unreplicate(x: Any) -> Any:
  return flax.jax_utils.unreplicate(x) if multi_gpu() else x


def pmean(x: Any) -> Any:
  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  return jax.lax.pmean(x, axis_name='batch') if multi_gpu() else x


def pmap(f: Any) -> Any:
  return jax.pmap(f, axis_name='batch') if multi_gpu() else jax.jit(f)


@flax.struct.dataclass
class TrainState:
  step: jax.Array  # Scalar
  params: flax_typing.FrozenVariableDict
  opt_state: optax.OptState
  rng: jax.Array


def compress_batch(batch: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
  """Compresses batch to reduce memory usage during initialization."""
  if multi_gpu():
    batch = jax.tree.map(lambda t: t[0], batch)  # Drop device dim.
  # Slice from all axis (batch, regressor, token).
  return jax.tree.map(
      lambda t: jax.lax.dynamic_slice(t, [0] * t.ndim, [1] * t.ndim), batch
  )


def create_train_state(
    model: icl_transformer.ICLTransformer,
    optimizer: optax.GradientTransformation,
    example_batch: Mapping[str, np.ndarray],
    seed: int = 0,
) -> TrainState:
  """Creates initial train state with possibility to override weights."""
  rng = jax.random.PRNGKey(seed)
  example_batch = compress_batch(example_batch)
  params = model.init(
      rng, method=model.fit, deterministic=False, **example_batch
  )

  opt_state = optimizer.init(params)
  return TrainState(jnp.array(0), params, opt_state, rng)


def loss_fn(
    params: flax_typing.FrozenVariableDict,
    model: icl_transformer.ICLTransformer,
    batch: Mapping[str, np.ndarray],
    training: bool,
    rng: jax.Array,
) -> tuple[jax.Array, Mapping[str, Scalar]]:
  """Loss function with metrics."""
  # pylint: disable=invalid-name
  mean, std = model.apply(
      params, deterministic=not training, rng=rng, method=model.fit, **batch
  )
  nlogprob = -jax.scipy.stats.norm.logpdf(batch['y'], mean, std)  # [B, L]

  # Only compute loss over target ys. Mask is BxL where True denotes context
  # token and False otherwise.
  target_mask = 1 - batch['mask']  # [B, L]
  target_nlogprob = nlogprob * target_mask  # [B, L]

  avg_nlogprob = metrics_lib.masked_mean(target_nlogprob, target_mask)
  loss = jnp.mean(avg_nlogprob)  # [B] -> Scalar

  metrics = metrics_lib.default_metrics(mean, batch['y'], target_mask)
  metrics['loss'] = loss
  return loss, metrics


def train_step(
    model: icl_transformer.ICLTransformer,
    optimizer: optax.GradientTransformation,
    train_state: TrainState,
    batch: Mapping[str, np.ndarray],
) -> tuple[TrainState, Mapping[str, Scalar]]:
  """Train for a single step."""
  dropout_rng, new_rng = jax.random.split(train_state.rng)

  loss_fn_with_rng = functools.partial(
      loss_fn, model=model, batch=batch, training=True, rng=dropout_rng
  )

  grad_fn = jax.value_and_grad(loss_fn_with_rng, has_aux=True)
  (_, metrics), grads = grad_fn(train_state.params)
  updates, new_opt_state = optimizer.update(
      pmean(grads), train_state.opt_state, train_state.params
  )
  new_params = optax.apply_updates(train_state.params, updates)
  new_state = TrainState(
      train_state.step + 1, new_params, new_opt_state, new_rng
  )

  metrics = {f'train_{k}': v for k, v in metrics.items()}
  return new_state, metrics


def eval_step(
    model: icl_transformer.ICLTransformer,
    train_state: TrainState,
    batch: Mapping[str, np.ndarray],
) -> Mapping[str, Scalar]:
  """Evaluation step."""
  _, metrics = loss_fn(
      train_state.params, model, batch, training=False, rng=train_state.rng
  )
  return {f'eval_{k}': v for k, v in metrics.items()}


def aggregate_metrics(
    metrics: list[Mapping[str, jnp.ndarray]] | Mapping[str, jnp.ndarray],
) -> Mapping[str, Scalar]:
  """Aggregates metrics (possibly from multiple gradient accumulation steps)."""
  if isinstance(metrics, list):
    metrics = jax.tree.map(lambda *args: jnp.stack(args), *metrics)
  metrics = jax.tree.map(jnp.mean, metrics)
  return {k: float(v) for k, v in metrics.items()}


def train(
    model_config: configs.ModelConfig,
    embedder_config: configs.T5EmbedderConfig,
    train_config: configs.TrainingConfig,
    data_config: configs.DataConfig,
):
  """Training loop."""
  # tf.data.AUTOTUNE can put data on GPU, causing extreme OOMs. Disable it.
  tf.config.set_visible_devices([], device_type='GPU')

  writer = metric_writers.create_default_writer(
      train_config.workdir, just_logging=jax.process_index() > 0
  )

  optimizer = train_config.create_optimizer()
  model = model_config.create_model(embedder_config)

  p_train_step = pmap(functools.partial(train_step, model, optimizer))
  p_eval_step = pmap(functools.partial(eval_step, model))

  ds_fn = data_config.seqio_dataset_fn()
  ds_fn = ds_lib.DistributedSeqioDatasetFn(ds_fn)

  train_ds = ds_fn('train', shuffle_files=True)
  train_ds = data_config.wrap_ds(train_ds, multi_gpu())
  train_it = train_ds.as_numpy_iterator()

  valid_ds = ds_fn('validation', shuffle_files=True)
  valid_ds = data_config.wrap_ds(valid_ds, multi_gpu())
  valid_it = valid_ds.as_numpy_iterator()

  init_train_state = create_train_state(
      model, optimizer, next(train_it), train_config.seed
  )

  # Set up checkpointing
  checkpoint_manager = ckpt_lib.get_checkpoint_manager(
      train_config.workdir,
      max_to_keep=train_config.max_to_keep_ckpts,
      best_fn=lambda metrics: metrics['eval_loss'],
      best_mode='min',
  )
  # Restore if available.
  train_state = ckpt_lib.restore_train_state(
      train_config.workdir, init_train_state
  )

  train_state = replicate(train_state)  # For pmap
  # Account for gradient accumulation by using step // grad_accum_steps.
  grad_accum_steps = train_config.grad_accum_steps
  eff_step = int(unreplicate(train_state.step)) // grad_accum_steps

  while eff_step < train_config.max_steps:
    if eff_step % train_config.validation_interval == 0:
      valid_agg_metrics = aggregate_metrics([
          p_eval_step(train_state, next(valid_it))
          for _ in range(grad_accum_steps)
      ])
      writer.write_scalars(eff_step, valid_agg_metrics)

      ckpt_train_state = unreplicate(train_state)
      checkpoint_manager.save(
          eff_step,
          items=dict(train_state=jax.tree.map(np.array, ckpt_train_state)),
          metrics=valid_agg_metrics,
      )

    all_train_metrics = []
    for _ in range(grad_accum_steps):
      train_state, train_metrics = p_train_step(train_state, next(train_it))
      all_train_metrics.append(train_metrics)
    writer.write_scalars(eff_step, aggregate_metrics(all_train_metrics))

    eff_step = int(unreplicate(train_state.step)) // grad_accum_steps
