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

from absl.testing import flagsaver
from optformer.common.data import vocabs
from optformer.common.data.datasets import distributed
import reverb
import tensorflow as tf
from absl.testing import absltest


class DistributedDatasetFnTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Setup original dataset.
    self.vocab = vocabs.AsciiVocab()
    self.raw_data = [{"inputs": "hi", "targets": "bye"}]
    self.original_dataset = tf.data.Dataset.from_generator(
        lambda: self.raw_data,
        output_types={"inputs": tf.string, "targets": tf.string},
        output_shapes={"inputs": [], "targets": []},
    )

    # Setup distributed components.
    # Server on separate process/machine.

    self.table_name = "test_table"
    self.server = reverb.Server(
        tables=[
            reverb.Table(
                name=self.table_name,
                sampler=reverb.selectors.Fifo(),
                remover=reverb.selectors.Fifo(),
                max_size=100000,
                rate_limiter=reverb.rate_limiters.MinSize(1),
                max_times_sampled=1,
            ),
        ]
    )
    self.server_address = f"localhost:{self.server.port}"

    # Separate process/machine.
    self.client = reverb.Client(self.server_address)

  def test_client_server_interaction(self):
    # Separate process/machine (ideally same as model training process)
    with flagsaver.flagsaver(reverb_address=self.server_address):
      dataset_fn = distributed.DistributedDatasetFn(self.table_name)
      self.distributed_dataset = dataset_fn(self.original_dataset)

    data = next(self.original_dataset.as_numpy_iterator())
    self.client.insert(data, {self.table_name: 1.0})
    distributed_data = next(self.distributed_dataset.as_numpy_iterator())
    self.assertEqual(data, distributed_data)


if __name__ == "__main__":
  absltest.main()
