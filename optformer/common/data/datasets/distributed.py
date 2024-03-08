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

"""Distributed dataset for scaling data-loading to many CPUs."""

from typing import Optional

from absl import flags
import attrs
from optformer.common.data.datasets import base
import reverb
import seqio
import tensorflow as tf
import tree

# This flag's value will be passed from downstream binaries.
REVERB_ADDRESS = flags.DEFINE_string(
    'reverb_address',
    None,
    'Address of Reverb server, of form `host:port`.',
)

DISABLE_REVERB = flags.DEFINE_bool(
    'disable_reverb',
    False,
    'If true disables distributed Reverb logic (makes wrapper no-op).',
)


@attrs.define
class DistributedDatasetFn(base.DatasetFn[tf.data.Dataset]):
  """Creates a distributed version of a dataset using the Reverb API.

  The distributed dataset will request data from a specific table of a reverb
  server which collects units of data from multiple clients rather than a file
  source, but still needs to know the incoming dtypes/shapes from a template.
  """

  table_name: str = attrs.field(init=True)

  num_workers_per_iterator: int = attrs.field(default=1, kw_only=True)
  max_samples_per_stream: int = attrs.field(default=120, kw_only=True)
  max_in_flight_samples_per_worker: int = attrs.field(default=2, kw_only=True)
  prefetch: Optional[int] = attrs.field(default=8, kw_only=True)

  def __call__(
      self, element_spec: dict[str, tf.TensorSpec]
  ) -> reverb.TimestepDataset:
    """Creates the distributed Reverb dataset as a server.

    Args:
      element_spec: A dict of `tf.TensorSpec` specifying the original dataset
        dtypes and shapes.

    Returns:
      Reverb server dataset.
    """

    if REVERB_ADDRESS.value is None:
      raise ValueError('`reverb_address` flag is still unset!')

    ds = reverb.TimestepDataset(
        server_address=REVERB_ADDRESS.value,
        table=self.table_name,
        dtypes=tree.map_structure(lambda x: x.dtype, element_spec),
        shapes=tree.map_structure(lambda x: x.shape, element_spec),
        num_workers_per_iterator=self.num_workers_per_iterator,
        max_samples_per_stream=self.max_samples_per_stream,
        max_in_flight_samples_per_worker=self.max_in_flight_samples_per_worker,
    )

    # Change output from default `ReplaySample` struct to actual data.
    ds = ds.map(lambda rs: rs.data, num_parallel_calls=tf.data.AUTOTUNE)

    if self.prefetch is not None:
      ds = ds.prefetch(buffer_size=self.prefetch)

    options = tf.data.Options()
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.parallel_batch = True
    return ds.with_options(options)


@attrs.define
class DistributedSeqioDatasetFn(seqio.DatasetFnCallable):
  """Creates a distributed dataset whose table name is according to split."""

  seqio_dataset_fn: seqio.DatasetFnCallable = attrs.field(init=True)

  def __call__(
      self, split: str, shuffle_files: bool, seed: Optional[int] = None
  ) -> tf.data.Dataset:
    ds = self.seqio_dataset_fn(split, shuffle_files, seed)
    if DISABLE_REVERB.value:
      return ds
    return DistributedDatasetFn(table_name=split)(ds.element_spec)

  @property
  def __name__(self):
    # Seqio registry requires the __name__.
    return 'DistributedSeqioDatasetFn'
