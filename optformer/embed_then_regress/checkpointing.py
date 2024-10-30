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
    workdir: epath.PathLike, **options_kwargs
) -> orbax_checkpoint.CheckpointManager:
  """Sets up Orbax checkpointing."""
  checkpoint_dir = epath.Path(workdir) / 'checkpoints'
  return orbax_checkpoint.CheckpointManager(
      checkpoint_dir,
      checkpointers={'train_state': orbax_checkpoint.PyTreeCheckpointer()},
      options=orbax_checkpoint.CheckpointManagerOptions(
          create=True, **options_kwargs
      ),
  )


def restore_train_state(
    workdir: epath.PathLike,
    initial_train_state: Any | None = None,
    *,
    step: int | None = None,
) -> dict[str, Any]:
  """Loads params from checkpoint workdir."""
  checkpoint_manager = get_checkpoint_manager(workdir)
  step = step if step is not None else checkpoint_manager.latest_step()
  if step is None and initial_train_state is None:
    raise ValueError('No last step found but no initial state provided either.')
  if step is None and initial_train_state is not None:
    return initial_train_state

  restored = checkpoint_manager.restore(
      step, items={'train_state': initial_train_state}
  )
  logging.info('Restored checkpoint from step %d', step)
  return restored['train_state']
