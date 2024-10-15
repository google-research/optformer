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
from absl import logging
from clu import metric_writers
from etils import epath
import flax
import flax.typing as flax_typing
import jax
import jax.numpy as jnp
import numpy as np
import optax
from optformer.common.data import datasets as ds_lib
from optformer.embed_then_regress import configs
from optformer.embed_then_regress import icl_transformer
from orbax import checkpoint as orbax_checkpoint
import tensorflow as tf


Scalar = jnp.ndarray | np.ndarray | float
EPS = 1e-7


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
    weights_override: dict[str, jax.Array] | None = None,
) -> TrainState:
  """Creates initial train state with possibility to override weights."""
  rng = jax.random.PRNGKey(seed)
  example_batch = compress_batch(example_batch)
  with jax.default_device(jax.devices('cpu')[0]):
    params = model.init(rng, deterministic=False, **example_batch)

    if weights_override:
      params = params.unfreeze()
      params.update(weights_override)
      params = params.freeze(params)

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
  mean, std = model.apply(params, deterministic=not training, rng=rng, **batch)
  nlogprob = -jax.scipy.stats.norm.logpdf(batch['y'], mean, std + EPS)  # [B, L]

  # Only compute loss over target ys. Mask is Bx1xNxN where mask[i, j] = True
  # if j is a context token and False otherwise.
  target_mask = 1 - batch['mask'][:, 0, :]  # [B, L]
  target_nlogprob = nlogprob * target_mask  # [B, L]

  avg_nlogprob = jnp.sum(target_nlogprob, axis=1) / jnp.sum(target_mask, axis=1)
  loss = jnp.mean(avg_nlogprob)  # [B] -> Scalar
  # TODO: Get more metrics.
  return loss, {'loss': loss}


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


def get_checkpoint_manager(
    workdir: epath.PathLike,
) -> orbax_checkpoint.CheckpointManager:
  """Sets up Orbax checkpointing."""
  # The keys in this dict should match the keys in `checkpointed_state`.
  checkpointers = dict(
      train_state=orbax_checkpoint.PyTreeCheckpointer(),
  )
  checkpoint_dir = epath.Path(workdir) / 'checkpoints'
  return orbax_checkpoint.CheckpointManager(
      checkpoint_dir,
      checkpointers=checkpointers,
      options=orbax_checkpoint.CheckpointManagerOptions(create=True),
  )


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

  train_state = create_train_state(
      model,
      optimizer,
      next(train_it),
      train_config.seed,
  )

  # Set up checkpointing
  checkpoint_manager = get_checkpoint_manager(train_config.workdir)
  # Restore if available.
  latest_step = checkpoint_manager.latest_step()
  if latest_step is not None:
    restored_state = checkpoint_manager.restore(
        latest_step,
        items=dict(train_state=train_state),  # initial train_state template
    )
    train_state = restored_state['train_state']
    logging.info('Restored checkpoint from step %d', latest_step)

  train_state = replicate(train_state)  # For pmap
  step = int(unreplicate(train_state.step))
  while step < train_config.max_steps:
    train_state, train_metrics = p_train_step(train_state, next(train_it))
    writer.write_scalars(step, jax.tree.map(jnp.mean, train_metrics))

    if step % train_config.validation_interval == 0:
      valid_metrics = p_eval_step(train_state, next(valid_it))
      writer.write_scalars(step, jax.tree.map(jnp.mean, valid_metrics))

    if step % train_config.checkpoint_interval == 0:
      ckpt_train_state = unreplicate(train_state)
      checkpoint_manager.save(
          step, items=dict(train_state=jax.tree.map(np.array, ckpt_train_state))
      )
    step = int(unreplicate(train_state.step))
