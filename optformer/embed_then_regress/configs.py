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

import dataclasses
import optax
from optformer.embed_then_regress import icl_transformer


@dataclasses.dataclass
class ModelConfig:
  """Model configuration."""

  d_model: int = 1024
  ffw_dim_ratio: int = 4
  nhead: int = 16
  dropout: float = 0.1
  num_layers: int = 8

  def create_model(self) -> icl_transformer.ICLTransformer:
    return icl_transformer.ICLTransformer(**dataclasses.asdict(self))


@dataclasses.dataclass
class TrainingConfig:
  """Training configuration."""

  learning_rate: float = 5e-4  # Optimal for a batch size of 128
  warmup_steps: int = 10000
  max_steps: int = 100000
  weight_decay: float = 1e-5
  gradient_clip: float = 0.5

  min_n_context: int = 10
  max_n_context: int = 100

  seed: int = 42

  validation_interval: int = 1000
  workdir = '../checkpoints'

  def create_optimizer(self) -> optax.GradientTransformation:
    learning_rate_fn = self._create_cosine_lr_fn()
    optimizer = optax.adamw(
        learning_rate_fn, b2=0.95, weight_decay=self.weight_decay
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(self.gradient_clip), optimizer
    )
    return optimizer

  def _create_cosine_lr_fn(self) -> optax.Schedule:
    """Creates learning rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=self.learning_rate,
        transition_steps=self.warmup_steps,
    )
    cosine_steps = max(self.max_steps - self.warmup_steps, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=self.learning_rate, decay_steps=cosine_steps
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn], boundaries=[self.warmup_steps]
    )
    return schedule_fn
