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

"""Factories for creating objects across the code.

NOTE: Rules for this file when using gin:

1. The methods below should be called AFTER `gin_utils.parse_gin_flags()` is
called in a binary.

2. Methods below should NOT have default argument values, to fail and
detect quickly if gin was misconfigured.

3. Do NOT apply `gin.configurable` decorator explicitly on any methods in this
file. Importers can either apply `gin.configurable` on their own (e.g. inside
binary mains), or use gin's dynamic registration.
"""

import math
from optformer.common.data import datasets as datasets_lib
import seqio
import tensorflow_datasets as tfds


def num_embeddings_factory(vocab: seqio.Vocabulary, multiple: int = 128) -> int:
  """Round up to the nearest multiple, ideally 2^x."""
  return math.ceil(vocab.vocab_size / multiple) * multiple


def output_features_factory(
    name: str, vocab: seqio.Vocabulary
) -> dict[str, seqio.Feature]:
  """Tokenization scheme used in task registration."""

  if name == 'no_eos':
    return {
        'inputs': seqio.Feature(vocabulary=vocab, add_eos=False),
        'targets': seqio.Feature(vocabulary=vocab, add_eos=False),
    }
  elif name == 'eos':
    return {
        'inputs': seqio.Feature(vocabulary=vocab, add_eos=True),
        'targets': seqio.Feature(vocabulary=vocab, add_eos=True),
    }
  elif name == 'eos_targets_only':
    return {
        'inputs': seqio.Feature(vocabulary=vocab, add_eos=False),
        'targets': seqio.Feature(vocabulary=vocab, add_eos=True),
    }
  else:
    raise ValueError(f'Output feature name {name} not found.')


def register_task(
    task_name: str,
    seqio_dataset_fn: seqio.DatasetFnCallable,
    output_features: dict[str, seqio.Feature],
    distributed: bool = True,
) -> seqio.Task:
  """Register task from dataset function.

  Args:
    task_name: Assigned name to registry.
    seqio_dataset_fn: Template dataset for extracting shape information. Since
      we apply the distributed wrapper, this can be a "dummy".
    output_features: Specifies lengths and preprocessing.
    distributed: Whether the data should come from a reverb server instead.

  Returns:
    Registered task.
  """

  if distributed:
    seqio_dataset_fn = datasets_lib.DistributedSeqioDatasetFn(seqio_dataset_fn)

  source = seqio.FunctionDataSource(
      seqio_dataset_fn,
      splits=(tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST),
  )

  return seqio.TaskRegistry.add(
      task_name,
      source,
      output_features,
      preprocessors=[
          seqio.preprocessors.tokenize,
          seqio.preprocessors.append_eos_after_trim,
      ],
      shuffle_buffer_size=100000,
  )
