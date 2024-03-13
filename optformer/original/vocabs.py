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

"""Vocabulary specific to Vizier using quantization."""

from optformer.common import serialization as s_lib
from optformer.common.data import vocabs
from optformer.original import serializers as os_lib
from optformer.vizier import serialization as vzs_lib


_VIZIER_SPECIAL_STRS = (
    os_lib.QuantizedMeasurementSerializer.PENDING,
    os_lib.QuantizedMeasurementSerializer.MISSING,
    os_lib.QuantizedMeasurementSerializer.INFEASIBLE,
    os_lib.QuantizedSuggestionSerializer.MISSING,
    os_lib.QuantizedSuggestionSerializer.INACTIVE_CHILD,
    os_lib.QuantizedTrialsSerializer.TRIAL_SEPARATOR,
    vzs_lib.TrialTokenSerializer.XY_SEPARATOR,
)


class QuantizedVocabulary(vocabs.HybridVocabulary[float]):
  """Vocabulary with additional quantized bins + special strings."""

  def __init__(
      self,
      sentencepiece_model_file: str,
      num_quantization_bins: int = 1000,
  ):

    self._integer_serializer = s_lib.IntegerTokenSerializer()
    self._string_serializer = s_lib.StringTokenSerializer()

    int_tokens = [
        self._integer_serializer.to_str(n) for n in range(num_quantization_bins)
    ]
    str_tokens = [
        self._string_serializer.to_str(s) for s in _VIZIER_SPECIAL_STRS
    ]

    super().__init__(
        sentencepiece_model_file=sentencepiece_model_file,
        extra_tokens=int_tokens + str_tokens,
    )
    self.num_quantization_bins = num_quantization_bins

  @property
  def deserializer(self) -> s_lib.IntegerTokenSerializer:
    """To deal with pytypes and only deserializes integer tokens."""
    return self._integer_serializer

  @property
  def quantization_vocab_index(self) -> int:
    return self.initial_extra_token_id
