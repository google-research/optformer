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
import tensorflow_datasets as tfds


_REVERB_BUFFER_SIZE = flags.DEFINE_integer(
    'reverb_buffer_size', int(10000), 'Reverb buffer size.'
)
_REVERB_PORT = flags.DEFINE_integer('reverb_port', None, 'Reverb server port.')


def main(_):
  """Creates a Reverb server with multiple tables for collecting data."""

  tables = []
  for table_name in (tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST):
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

  server = reverb.Server(tables, port=_REVERB_PORT.value)
  server.wait()


if __name__ == '__main__':
  app.run(main)
