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
  use_early_stop: bool = attrs.field(default=True)

  seed: int = attrs.field(default=0)
  loss_batch_size: int = attrs.field(default=64)  # TPU memory limit
  batch_per_tpu: Optional[int] = attrs.field(default=4)  # TPU memory limit

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
    rng = jax.random.PRNGKey(self.seed)
    jit_train_with_lr = jax.jit(
        trainer_lib.train_with_lr,
        static_argnames=['model', 'num_microbatches'],
    )
    num_grad_steps_per_epoch = math.ceil(len(train_data) / self.batch_size)
    train_iter = self._make_train_ds(train_data).as_numpy_iterator()

    valid_losses = []
    prev_state = state  # Never used.
    for n_epoch in range(self.max_num_epochs):

      if self.use_early_stop:
        valid_losses.append(self._loss(state.params, valid_data))
        if _detect_overfitting(valid_losses):
          logging.info('Early stopping at epoch %d', n_epoch)
          return prev_state

      prev_state = state
      for _ in range(num_grad_steps_per_epoch):
        rng, subkey = jax.random.split(rng)
        state, _ = jit_train_with_lr(
            state,
            next(train_iter),
            learning_rate=jnp.array(self.learning_rate),
            dropout_rng=subkey,
            model=self.model,
            num_microbatches=self._num_microbatches,
        )

    return state

  def _make_train_ds(self, data: Sequence[_D]) -> tf.data.Dataset:
    ds = self.inference_dataset_fn(data)
    ds = ds.shuffle(len(data), seed=self.seed)
    ds = ds.repeat()
    return ds.batch(self.batch_size)

  def _loss(self, params: models.PyTree, data: Sequence[_D]) -> float:
    """Compute total loss over data."""
    jit_loss_fn = jax.jit(self.model.loss_fn)
    num_batches = math.ceil(len(data) / self.loss_batch_size)

    ds = self.inference_dataset_fn(data)
    ds = ds.batch(self.loss_batch_size, drop_remainder=False)
    data_iter = ds.as_numpy_iterator()

    step_losses = []
    for _ in range(num_batches):
      loss, _ = jit_loss_fn(params, next(data_iter), dropout_rng=None)
      step_losses.append(loss)
    return jnp.sum(jnp.array(step_losses)).item()
