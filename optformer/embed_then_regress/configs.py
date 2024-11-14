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

"""Configs and their corresponding `make`-like functions."""

import abc
import dataclasses
import functools
from typing import Callable
from flax import linen as nn
import jax
import jax.numpy as jnp
import jaxtyping as jt
import optax
from optformer.embed_then_regress import icl_transformer
from optformer.t5x import embedders
import seqio
import tensorflow as tf

DatasetFnCallable = seqio.DatasetFnCallable
FlaxT5Embedder = embedders.FlaxT5Embedder


@dataclasses.dataclass
class T5EmbedderConfig:
  """T5 embedder configuration."""

  t5_embedder: str = 'small'
  reduction: str = 'attention'
  freeze_encoder: bool = False

  def create_embedder_factory(self) -> Callable[[], nn.Module]:
    """Create t5-based embedder and send to ICLTransformer."""
    if self.t5_embedder == 'small':
      embedder_factory = FlaxT5Embedder.from_small
    elif self.t5_embedder == 'base':
      embedder_factory = FlaxT5Embedder.from_base
    elif self.t5_embedder == 'large':
      embedder_factory = FlaxT5Embedder.from_large
    elif self.t5_embedder == 'xl':
      embedder_factory = FlaxT5Embedder.from_xl
    elif self.t5_embedder == 'xxl':
      embedder_factory = FlaxT5Embedder.from_xxl
    else:
      raise ValueError(f'Unknown T5 embedder: {self.t5_embedder}')

    if self.reduction == 'pooling':
      reduction_factory = embedders.PoolingReduction
    elif self.reduction == 'attention':
      reduction_factory = embedders.AttentionReduction
    else:
      raise ValueError(f'Unknown reduction: {self.reduction}')

    return functools.partial(
        embedder_factory,
        reduction_factory=reduction_factory,
        freeze_encoder=self.freeze_encoder,
    )


@dataclasses.dataclass
class ModelConfig:
  """Model configuration."""

  d_model: int = 1024
  ffw_dim_ratio: int = 4
  nhead: int = 16
  dropout: float = 0.1
  num_layers: int = 8
  use_metadata: bool = True
  std_transform: str = 'exp'

  def create_model(
      self, embedder_config: T5EmbedderConfig
  ) -> icl_transformer.ICLTransformer:

    kwargs = dataclasses.asdict(self)
    kwargs.pop('std_transform')

    return icl_transformer.ICLTransformer(
        std_transform_fn=self.create_std_transform_fn(),
        embedder_factory=embedder_config.create_embedder_factory(),
        **kwargs,
    )

  def create_std_transform_fn(
      self,
  ) -> Callable[[jt.Float[jax.Array, '*A']], jt.Float[jax.Array, '*A']]:
    """Creates std transform function."""
    if self.std_transform == 'exp':
      return jnp.exp
    elif self.std_transform == 'exp10':
      return lambda x: jnp.exp(10.0 * x)
    elif self.std_transform == 'softplus':
      return jax.nn.softplus
    elif self.std_transform == 'softplus10':
      return lambda x: jax.nn.softplus(10.0 * x)
    elif self.std_transform == 'abs':
      return jnp.abs
    elif self.std_transform == 'abs10':
      return lambda x: jnp.abs(10.0 * x)
    elif self.std_transform == 'shifted_relu':
      return lambda x: jax.nn.relu(x + 1.0)
    elif self.std_transform == 'shifted_relu10':
      return lambda x: jax.nn.relu(10.0 * x + 1.0)
    else:
      raise ValueError(f'Unknown std_transform: {self.std_transform}')


@dataclasses.dataclass
class TrainingConfig:
  """Training configuration."""

  base_lr: float = 3e-5
  warmup_steps: int = 10000
  max_steps: int = 100000
  weight_decay: float = 1e-5
  gradient_clip: float = 0.5
  grad_accum_steps: int = 1

  seed: int = 42

  validation_interval: int = 100
  max_to_keep_ckpts: int = 5
  workdir = '../checkpoints'

  def create_optimizer(self) -> optax.GradientTransformation:
    """Creates optimizer with learning rate scheduling."""
    learning_rate_fn = self._create_cosine_lr_fn()
    optimizer = optax.adamw(
        learning_rate_fn, b2=0.95, weight_decay=self.weight_decay
    )

    multi_steps = optax.MultiSteps(optimizer, self.grad_accum_steps)
    optimizer = multi_steps.gradient_transformation()

    optimizer = optax.chain(
        optax.clip_by_global_norm(self.gradient_clip), optimizer
    )
    return optimizer

  def _create_cosine_lr_fn(self) -> optax.Schedule:
    """Creates learning rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=self.base_lr,
        transition_steps=self.warmup_steps,
    )
    cosine_steps = max(self.max_steps - self.warmup_steps, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=self.base_lr, decay_steps=cosine_steps
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn], boundaries=[self.warmup_steps]
    )

    # Account for gradient accumulation by using count // grad_accum_steps.
    gradient_accum_schedule_fn = lambda count: schedule_fn(
        count // self.grad_accum_steps
    )
    return gradient_accum_schedule_fn


@dataclasses.dataclass
class DataConfig(abc.ABC):
  """Data configuration, to be subclassed for each task."""

  # Maximum batch size for A100 GPU w/ trainable small T5 embedder.
  per_device_batch_size: int = 4
  max_token_length: int = 256

  def wrap_ds(
      self, ds: tf.data.Dataset, multi_gpu: bool = False
  ) -> tf.data.Dataset:
    """This should be used at the trainer level."""
    ds = self._tokenize_ds(ds)
    ds = ds.batch(
        self.per_device_batch_size,
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if multi_gpu:  # Device count leading dimension, required by jax.pmap.
      ds = ds.batch(
          jax.local_device_count(),
          drop_remainder=True,
          num_parallel_calls=tf.data.AUTOTUNE,
      )
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

  def _tokenize_ds(self, ds: tf.data.Dataset) -> tf.data.Dataset:
    """Tokenizes trajectories."""
    vocab = self.create_vocab()
    output_features = {
        'x': seqio.Feature(vocab, add_eos=False),
        'metadata': seqio.Feature(vocab, add_eos=False),
    }
    ds = seqio.preprocessors.tokenize(
        ds,
        output_features=output_features,
        copy_pretokenized=False,
        with_eos=False,
    )

    feature_lengths = {
        'x': self.max_token_length,
        'metadata': self.max_token_length,
    }
    # Fancy logic since trimming/padding only affects the first dimension.
    transpose_x_only = lambda d: {
        k: tf.transpose(v.to_tensor()) if k == 'x' else v for k, v in d.items()
    }
    ds = ds.map(transpose_x_only, num_parallel_calls=tf.data.AUTOTUNE)
    ds = seqio.trim_and_pad_dataset(ds, feature_lengths)
    ds = ds.map(
        lambda d: {k: tf.transpose(v) if k == 'x' else v for k, v in d.items()},
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds

  @abc.abstractmethod
  def seqio_dataset_fn(self) -> DatasetFnCallable:
    """Creates seqio dataset fn for iterating (unbatched) examples.

    SeqIO DatasetFn also allows shuffling / splitting. Once batched, these
    examples should directly be callable by the model.

    Returns:
      SeqIO dataset fn.
    """

  @abc.abstractmethod
  def create_vocab(self) -> seqio.Vocabulary:
    """Returns the vocabulary used for tokenization."""
