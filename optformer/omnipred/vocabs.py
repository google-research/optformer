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

import attrs
from optformer.common.data import vocabs
from optformer.common.serialization import numeric


@attrs.define
class FloatMetricVocabulary(vocabs.HybridVocabulary[float]):
  """Vocabulary for specifically dealing with floats."""

  _deserializer: numeric.DigitByDigitFloatTokenSerializer = attrs.field(
      factory=numeric.DigitByDigitFloatTokenSerializer
  )

  @property
  def deserializer(self) -> numeric.DigitByDigitFloatTokenSerializer:
    """To deal with pytypes."""
    return self._deserializer

  @property
  def decode_length(self) -> int:
    """Expected decode length, noting initial 329 token ID is always used."""
    return self._deserializer.num_tokens_per_obj + 1
