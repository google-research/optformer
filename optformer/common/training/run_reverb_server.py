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

"""Launches a Reverb server for collecting training data."""

from absl import app
from absl import flags
import reverb
from reverb.platform.default import checkpointers
import tensorflow_datasets as tfds


_REVERB_BUFFER_SIZE = flags.DEFINE_integer(
    'reverb_buffer_size', int(10000), 'Reverb buffer size.'
)
_REVERB_PORT = flags.DEFINE_integer('reverb_port', None, 'Reverb server port.')
_REVERB_CHECKPOINTING = flags.DEFINE_bool('reverb_checkpointing', False, '')
_REVERB_TABLE_TYPE = flags.DEFINE_enum(
    'reverb_table_type',
    'uniform',
    ['uniform', 'queue'],
    'Reverb table type.  Uniform is non-deterministic insertion and removal. '
    ' Records may be evicted without being sampled, but are never sampled'
    ' twice.  Queue is deterministic, records are sampled exactly once and'
    ' inserts that would overflow the reverb_buffer_size are blocked.',
)


def main(_):
  """Creates a Reverb server with multiple tables for collecting data."""

  tables = []
  for table_name in (tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST):
    match _REVERB_TABLE_TYPE.value:
      case 'uniform':
        tables.append(
            reverb.Table(
                name=table_name,
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Uniform(),
                max_size=_REVERB_BUFFER_SIZE.value,
                rate_limiter=reverb.rate_limiters.MinSize(1),
                max_times_sampled=1,
            )
        )
      case 'queue':
        tables.append(
            reverb.Table.queue(
                name=table_name, max_size=_REVERB_BUFFER_SIZE.value
            )
        )
      case _:
        raise ValueError(
            f'Unknown reverb_table_type: {_REVERB_TABLE_TYPE.value}'
        )

  checkpointer = None
  if _REVERB_CHECKPOINTING.value:
    checkpointer = checkpointers.default_checkpointer()

  server = reverb.Server(
      tables, port=_REVERB_PORT.value, checkpointer=checkpointer
  )
  server.wait()


if __name__ == '__main__':
  app.run(main)
