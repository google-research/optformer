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


_BUFFER_SIZE = flags.DEFINE_integer(
    'buffer_size', int(10000), 'Reverb buffer size.'
)
_PORT = flags.DEFINE_integer('port', None, 'Reverb server port.')
_CHECKPOINTING = flags.DEFINE_bool('checkpointing', True, 'Checkpoint to disk.')
_TABLE_TYPE = flags.DEFINE_enum(
    'table_type',
    'uniform',
    ['uniform', 'queue'],
    'Uniform allows insertion w/ random replacement. Queue blocks when full.',
)


def main(_):
  """Creates a Reverb server with multiple tables for collecting data."""

  tables = []
  for table_name in (tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST):
    if _TABLE_TYPE.value == 'uniform':
      tables.append(
          reverb.Table(
              name=table_name,
              sampler=reverb.selectors.Uniform(),
              remover=reverb.selectors.Uniform(),
              max_size=_BUFFER_SIZE.value,
              rate_limiter=reverb.rate_limiters.MinSize(1),
              max_times_sampled=1,
          )
      )
    elif _TABLE_TYPE.value == 'queue':
      tables.append(
          reverb.Table.queue(name=table_name, max_size=_BUFFER_SIZE.value)
      )
    else:
      raise ValueError(f'Unknown table type: {_TABLE_TYPE.value}')

  checkpointer = None
  if _CHECKPOINTING.value:
    checkpointer = checkpointers.default_checkpointer()

  server = reverb.Server(tables, port=_PORT.value, checkpointer=checkpointer)
  server.wait()


if __name__ == '__main__':
  app.run(main)
