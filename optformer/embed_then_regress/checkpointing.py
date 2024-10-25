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

"""Checkpointing API."""

from typing import Any
from absl import logging
from etils import epath
from orbax import checkpoint as orbax_checkpoint


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


def restore_train_state(
    workdir: epath.PathLike,
    initial_train_state: dict[str, Any] | None = None,
    *,
    step: int | None = None,
) -> dict[str, Any]:
  """Loads params from checkpoint workdir."""
  initial_train_state = initial_train_state or {'train_state': None}

  checkpoint_manager = get_checkpoint_manager(workdir)
  step = step if step is not None else checkpoint_manager.latest_step()
  if step is None:
    logging.info('No last step found. Orbax has not run yet.')
    return initial_train_state

  checkpointed_state = checkpoint_manager.restore(
      step, items=initial_train_state
  )
  logging.info('Restored checkpoint from step %d', step)
  return checkpointed_state['train_state']
