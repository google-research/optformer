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

"""Wrapper classes which use our `DatasetFn` protocol to interact w/ other types."""

from typing import Optional, Sequence

import attrs
from optformer.common.data.datasets import base
import seqio
import tensorflow as tf


@attrs.define
class SeqioDatasetFnFunctor:
  """Applies our own `DatasetFn` protocols over SeqIO's `DatasetFnCallable`.

  Same as a 'functor' in functional programming terminology.
  """

  dataset_fns: Sequence[base.DatasetFn[tf.data.Dataset]] = attrs.field()

  def __call__(
      self, seqio_dataset_fn: seqio.DatasetFnCallable
  ) -> seqio.DatasetFnCallable:
    """Returns new SeqIO `DatasetFnCallable` after applying our dataset maps."""

    def new_dataset_fn(
        split: str,
        shuffle_files: bool,
        seed: Optional[int] = None,
    ) -> tf.data.Dataset:
      dataset = seqio_dataset_fn(split, shuffle_files, seed)
      for dataset_fn in self.dataset_fns:
        dataset = dataset_fn(dataset)
      return dataset

    return new_dataset_fn
