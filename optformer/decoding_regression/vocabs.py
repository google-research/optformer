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

"""Vocabularies for encoding/decoding floats."""

import abc
import re

import attrs
import jaxtyping as jt
import numpy as np
from optformer.common.serialization import numeric

NEG_INF = -1e7


def extract_all_tokens(input_string):
  """Extracts all elements from a string of the form <...><...><...>..."""
  pattern = r'<[^>]+>'  # Matches anything between < and >
  matches = re.findall(pattern, input_string)
  return matches


class FloatVocab(abc.ABC):
  """Customized vocabulary API which allows constrained decoding via masking."""

  @property
  @abc.abstractmethod
  def size(self) -> int:  # V
    """Number of tokens in the vocabulary."""

  @property
  @abc.abstractmethod
  def token_length(self) -> int:
    """Length of token sequence needed to represent a single float."""

  @abc.abstractmethod
  def logit_mask(self, index: int) -> jt.Bool[jt.Array, 'V']:
    """Returns a mask of allowed tokens for the given index."""

  @abc.abstractmethod
  def to_int(self, f: float) -> list[int]:
    """Converts a float to a list of token ids."""

  @abc.abstractmethod
  def from_int(self, token_ids: list[int]) -> float:
    """Converts a list of token ids to a float."""


@attrs.define
class UnnormalizedVocab(FloatVocab):
  """Based on OmniPred's default float serializer."""

  serializer: numeric.IEEEFloatTokenSerializer = attrs.field(
      factory=numeric.IEEEFloatTokenSerializer
  )

  def __attrs_post_init__(self):
    self._vocab_toks = list(self.serializer.all_tokens_used())
    self._toks_to_ids = {token: i for i, token in enumerate(self._vocab_toks)}

  @property
  def size(self) -> int:
    return len(self._vocab_toks)

  @property
  def token_length(self) -> int:
    return self.serializer.num_tokens_per_obj

  def logit_mask(self, index: int) -> jt.Bool[jt.Array, 'V']:
    """Returns a mask of allowed tokens for the given index."""
    token_ids_allowed = np.array(
        [self._toks_to_ids[t] for t in self.serializer.tokens_used(index)]
    )
    mask = np.zeros(self.size, dtype=bool)
    mask[token_ids_allowed] = True
    return mask

  def to_int(self, f: float) -> list[int]:
    str_f = self.serializer.to_str(f)
    tok_strs = extract_all_tokens(str_f)
    return [self._toks_to_ids[t] for t in tok_strs]

  def from_int(self, token_ids: list[int]) -> float:
    str_f = ''.join([self._vocab_toks[tid] for tid in token_ids])
    return self.serializer.from_str(str_f)


# TODO: Add support for more bases.
@attrs.define
class NormalizedVocab(FloatVocab):
  """Vocab which supports only numbers within [0,1]."""

  num_bits: int = attrs.field(default=1)

  @property
  def size(self) -> int:
    return 2

  @property
  def token_length(self) -> int:
    return self.num_bits

  def logit_mask(self, index: int):
    return np.ones(self.size, dtype=bool)

  def to_int(self, f: float) -> list[int]:
    assert f >= 0 and f <= 1
    f_trunc = int(f * 2.0**self.num_bits)
    f_trunc = bin(f_trunc)[2:].zfill(self.num_bits)
    return [int(b) for b in f_trunc]

  def from_int(self, token_ids: list[int]) -> float:
    x = np.asarray(token_ids)
    return np.sum(x * 2.0 ** (-1 * np.arange(1, len(x) + 1)))
