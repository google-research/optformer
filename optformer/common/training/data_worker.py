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

"""Library functions to allow CPU workers to collect data from a source."""

import time
from typing import Dict, Optional
from absl import logging
import numpy as np
import reverb
import seqio
import tensorflow_datasets as tfds


def produce_data(
    dataset_fn: seqio.DatasetFnCallable,
    reverb_address: str,
    sample_probs: Optional[Dict[tfds.Split, float]] = None,
    shuffle_files: bool = True,
    seed: Optional[int] = None,
) -> None:
  """Sends data items to a reverb client.

  This function is meant to be gin-configured if called in a binary.

  Args:
    dataset_fn: SeqIO-specific DatasetFn for specifying splits and shuffling.
    reverb_address: Address of the Reverb server to send data to. This should be
      set via FLAGS (from an XM launch).
    sample_probs: Probabilities for choosing a split at every iteration. The
      split also determines the table name to send data to. NOTE: This does NOT
      affect the actual train/validation/test partitioning.
    shuffle_files: `dataset_fn` argument. Can be overridden in gin.
    seed: `dataset_fn` argument. Should be set from FLAGS (replica ID from XM
      launch).
  """
  if sample_probs is None:
    sample_probs = {
        tfds.Split.TRAIN: 0.98,
        tfds.Split.VALIDATION: 0.01,
        tfds.Split.TEST: 0.01,
    }

  if seed is None:
    seed = int(time.time())

  logging.info('Attempting to connect to Reverb address: %s', reverb_address)
  reverb_client = reverb.Client(reverb_address)

  split_to_dataset = {}
  for split in sample_probs:
    dataset = dataset_fn(split=split, shuffle_files=shuffle_files, seed=seed)
    dataset = dataset.repeat()
    dataset = iter(dataset.as_numpy_iterator())
    split_to_dataset[split] = dataset

  count = 0
  while True:
    split = np.random.choice(
        list(sample_probs.keys()), p=list(sample_probs.values())
    )
    dataset = split_to_dataset[split]

    data = next(dataset)
    logging.log_first_n(logging.INFO, data, 10)
    logging.log_every_n(logging.INFO, 'Adding an item %d', 100, count)
    count += 1

    # Here we are sending one element at a time. Alternatively, we can send
    # the whole batch, but the training job will need to know batch size
    # used on the data producer side.
    try:
      reverb_client.insert(data, {split: 1.0})
    except RuntimeError as e:
      if 'Error when confirming that all items written to table.' in str(e):
        # This can happen when servers go down. More specifically when
        # "The writer has sent items that should be inserted and the server
        # may or may not have completed the request but was unable to send the
        # confirmation that the job has been completed before the connection
        # was lost", according to cassirer@.
        #
        # As a workaround, we just ignore the error and try again with the
        # next item. This may result in some data being lost. We don't know
        # whether or not the server received the data, so I'm not going to
        # retry as I'm judging that usually it'd be worse to risk introducing
        # duplicates than to drop some data.
        #
        # Longer term, the recommended approach is to switch to use reverb's
        # trajectory_writer, which doesn't suffer from this issue and should
        # perform better too, but this requires more work.
        logging.warning(
            'Reverb client insert failed (%s). This can happen if reverb '
            'servers go down. Ignoring and trying again with the next item.',
            e,
        )
      else:
        raise e
