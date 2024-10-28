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

"""Omnipred-specific vocabulary."""

import functools
from optformer.common.data import vocabs
from optformer.common.serialization import numeric

DEFAULT_VOCAB_PATH = 'gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model'  # pylint: disable=line-too-long

DigitByDigitFloatTokenSerializer = numeric.DigitByDigitFloatTokenSerializer


class FloatMetricVocabulary(vocabs.HybridVocabulary[float]):
  """Vocabulary for specifically dealing with floats."""

  def __init__(
      self,
      sentencepiece_model_file: str,
      deserializer: DigitByDigitFloatTokenSerializer | None = None,
  ):

    self._deserializer = deserializer or DigitByDigitFloatTokenSerializer()

    super().__init__(
        sentencepiece_model_file,
        extra_tokens=list(self._deserializer.all_tokens_used()),
    )

  @property
  def deserializer(self) -> DigitByDigitFloatTokenSerializer:
    """To deal with pytypes."""
    return self._deserializer

  @property
  def decode_length(self) -> int:
    """Expected decode length, noting initial token ID is always used."""
    return self._deserializer.num_tokens_per_obj + 1

  @functools.cached_property
  def initial_token_id(self) -> int:
    s = ''.join(list(self._deserializer.all_tokens_used()))
    return self.encode(s)[0]
