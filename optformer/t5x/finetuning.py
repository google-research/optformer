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

"""Finetuning wrappers for T5X."""

import math
from typing import Generic, Optional, Sequence, TypeVar
from absl import logging
import attrs
import jax
from jax import numpy as jnp
from optformer.common.data import datasets
from t5x import models
from t5x import train_state as train_state_lib
from t5x import trainer as trainer_lib
import tensorflow as tf


def _detect_overfitting(losses: Sequence[float]) -> bool:
  if len(losses) <= 1:
    return False
  return losses[-1] > losses[-2]


_D = TypeVar('_D')


@attrs.define
class Finetuner(Generic[_D]):
  """Finetunes against training data until overfitting or max epochs."""

  model: models.BaseTransformerModel = attrs.field()
  inference_dataset_fn: datasets.E2EInferenceDatasetFn[_D] = attrs.field()

  learning_rate: float = attrs.field(default=1e-5)  # 10x lower than training.
  batch_size: int = attrs.field(default=256)  # Should match training.
  max_num_epochs: int = attrs.field(default=30)
  reset_optimizer_state: bool = attrs.field(default=False)
  use_early_stop: bool = attrs.field(default=True)

  seed: int = attrs.field(default=0)
  batch_per_tpu: Optional[int] = attrs.field(default=4)  # Varies by hardware.

  weight_metrics_computer: trainer_lib.WeightMetricsComputer = attrs.field(
      factory=trainer_lib.WeightMetricsComputer
  )

  # Post init fields.
  _num_microbatches: int | None = attrs.field(init=False)

  def __attrs_post_init__(self):
    self._num_microbatches = (
        self.batch_size // self.batch_per_tpu if self.batch_per_tpu else None
    )

  def finetune(
      self,
      train_data: Sequence[_D],
      valid_data: Sequence[_D],
      state: train_state_lib.FlaxOptimTrainState,
  ) -> train_state_lib.FlaxOptimTrainState:
    """Finetunes."""
    if self.reset_optimizer_state:
      logging.info('Resetting optimizer state and step count.')
      new_optimizer_state = self.model.optimizer_def.init_state(state.params)
      new_optimizer = state._optimizer.replace(state=new_optimizer_state)  # pylint: disable=protected-access
      state = state.replace(_optimizer=new_optimizer)

    rng = jax.random.PRNGKey(self.seed)
    jit_train_with_lr = jax.jit(
        trainer_lib.train_with_lr,
        static_argnames=[
            'model',
            'num_microbatches',
            'weight_metrics_computer',
        ],
    )
    num_grad_steps_per_epoch = math.ceil(len(train_data) / self.batch_size)
    train_iter = self._make_train_ds(train_data).as_numpy_iterator()

    valid_losses = []
    stepwise_metrics = []
    prev_state = state
    n_epoch = 0
    while n_epoch < self.max_num_epochs:
      valid_losses.append(self._loss(state.params, valid_data))

      if self.use_early_stop and _detect_overfitting(valid_losses):
        state = prev_state
        break

      prev_state = state
      for _ in range(num_grad_steps_per_epoch):
        rng, subkey = jax.random.split(rng)
        state, metrics = jit_train_with_lr(
            state,
            next(train_iter),
            learning_rate=jnp.array(self.learning_rate),
            dropout_rng=subkey,
            model=self.model,
            num_microbatches=self._num_microbatches,
            weight_metrics_computer=self.weight_metrics_computer,
        )
        stepwise_metrics.append(metrics)

      n_epoch += 1

    logging.info('Finetuned for %s epochs.', n_epoch)
    logging.info('Validation losses: %s', valid_losses)
    logging.info('Stepwise metrics: %s', stepwise_metrics)
    return state

  def _make_train_ds(self, data: Sequence[_D]) -> tf.data.Dataset:
    ds = self.inference_dataset_fn(data)
    ds = ds.shuffle(len(data), seed=self.seed)
    ds = ds.repeat()
    return ds.batch(self.batch_size)

  def _loss(self, params: models.PyTree, data: Sequence[_D]) -> float:
    """Compute total loss over data."""
    jit_loss_fn = jax.jit(self.model.loss_fn)
    num_batches = math.ceil(len(data) / self.batch_per_tpu)

    ds = self.inference_dataset_fn(data)
    ds = ds.batch(self.batch_per_tpu, drop_remainder=False)
    data_iter = ds.as_numpy_iterator()

    step_losses = []
    for _ in range(num_batches):
      loss, _ = jit_loss_fn(params, next(data_iter), dropout_rng=None)
      step_losses.append(loss)
    return jnp.sum(jnp.array(step_losses)).item()
