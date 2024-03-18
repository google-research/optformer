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

"""Inference-related classes."""

from typing import Optional

from absl import flags
from absl import logging
import attrs
import jax
from paxml import base_experiment
from paxml import base_task
from paxml import checkpoints
from paxml import partitioning
from paxml import train_states
from praxis import py_utils
import tensorflow as tf


@attrs.define
class InferenceConfig:
  """Minimum set of objects required for end-to-end inference."""

  checkpoint_dir: str = attrs.field()
  checkpoint_step: Optional[int] = attrs.field()
  experiment: base_experiment.BaseExperiment = attrs.field()

  pathways_bns: Optional[str] = attrs.field(default=None)

  # Internal state below
  partitioner: partitioning.Partitioner = attrs.field(init=False)
  train_state: train_states.TrainState = attrs.field(init=False)
  task: base_task.BaseTask = attrs.field(init=False)

  def __attrs_post_init__(self):
    logging.info('Connecting to Pathways...')
    self._init_pathways()
    logging.info('Creating partitioner...')
    self._init_partitioner()
    logging.info('Loading checkpoint...')
    self._load_checkpoint()

  def _init_pathways(self):
    tf.config.experimental.set_visible_devices([], 'GPU')
    jax.config.update('jax_xla_backend', 'pathways')
    if self.pathways_bns:
      jax.config.update('jax_backend_target', self.pathways_bns)
    jax.config.update('jax_platform_name', 'cpu')
    py_utils.set_globally_use_rbg_prng_key()
    logging.info(jax.local_devices())
    flags.FLAGS.pmap_use_tensorstore = True

  def _init_partitioner(self):
    """Initializes the partitioner."""
    self.task = self.experiment.task().Instantiate()
    self.partitioner = partitioning.create_partitioner(
        self.task, reshard_inputs=True
    )
    input_specs_provider = (
        self.experiment.get_input_specs_provider_params().Instantiate()
    )
    self.partitioner.setup(
        self.task,
        init_key=jax.random.PRNGKey(1234),
        train_inputs_shape_dtype=input_specs_provider.get_input_specs(),
    )

  def _load_checkpoint(self):
    metadata = self.partitioner.get_train_state_metadata()
    self.train_state = checkpoints.restore_checkpoint(
        metadata.padded_global_shapes,
        self.checkpoint_dir,
        global_mesh=self.partitioner.global_mesh,
        checkpoint_type=checkpoints.CheckpointType.GDA,
        state_specs=metadata.partition_specs,
        step=self.checkpoint_step,
    )
