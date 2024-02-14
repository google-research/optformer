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

"""Ways to filter features."""

from typing import Dict

import attrs
from optformer.common.data.filters import base
import seqio
import tensorflow as tf


@attrs.define
class TokenLengthFilter(base.Filter[Dict[str, tf.Tensor]]):
  """Check if the string is below a certain token length given a vocabulary.

  If no vocabulary is provided, the raw character length is checked.
  """

  max_token_lengths: Dict[str, int] = attrs.field(
      factory=lambda: {'inputs': 4096, 'targets': 4096}
  )
  vocab: seqio.Vocabulary | None = attrs.field(default=None)

  def __call__(self, features: Dict[str, tf.Tensor]) -> bool:
    for k, v in self.max_token_lengths.items():
      if self.vocab:
        f_length = len(self.vocab.encode(features[k].numpy()))
      else:
        f_length = len(features[k].numpy())
      if f_length > v:
        raise ValueError(f'Feature {k} has length {f_length} > {v}')

    return True
