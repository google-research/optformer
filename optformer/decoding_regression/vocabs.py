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
import itertools
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


@attrs.define
class NormalizedVocab(FloatVocab):
  """Vocab which supports only numbers within [0,1]."""

  base: int = attrs.field(default=2)
  length: int = attrs.field(default=1)

  @property
  def size(self) -> int:
    return self.base

  @property
  def token_length(self) -> int:
    return self.length

  def logit_mask(self, index: int):
    del index
    return np.ones(self.size, dtype=bool)

  def to_int(self, f: float) -> list[int]:
    if not 0 <= f <= 1:
      raise ValueError(f'f must be between 0 and 1, got {f}')

    f_trunc = int(f * self.base**self.length)
    if f_trunc == self.base**self.length:
      f_trunc -= 1  # Adjust for the edge case when f is exactly 1
    f_trunc = np.base_repr(f_trunc, base=self.base).zfill(self.length)
    return [int(b) for b in f_trunc]

  def from_int(self, token_ids: list[int]) -> float:
    if len(token_ids) != self.length:
      raise ValueError(f'Length {len(token_ids)} does not match {self.length}.')
    if not all(0 <= tid < self.base for tid in token_ids):
      raise ValueError(f'{token_ids} out of range(0, {self.base})')

    x = np.asarray(token_ids)
    coeff = np.power(self.base, -1 * np.arange(1, len(x) + 1), dtype=np.float32)
    return float(np.sum(x * coeff))


@attrs.define
class HammingDistanceVocab(FloatVocab):
  """Based on https://ieeexplore.ieee.org/document/8437644.

  Minimizes magnitude change on the float if the tokenization is corrupted. For
  now only works on base=2.
  """

  base: int = attrs.field(default=2)
  length: int = attrs.field(default=1)

  all_binary_sequences = attrs.field(init=False)

  def __attrs_post_init__(self):
    if self.base != 2:
      raise ValueError('Only base=2 is supported.')

    self.all_binary_sequences = []
    for i in range(self.length + 1):
      for comb in itertools.combinations(range(self.length), i):
        binary_seq = [0] * self.length
        for c in comb:
          binary_seq[c] = 1
        self.all_binary_sequences.append(tuple(binary_seq))
    self.all_binary_sequences.sort(key=lambda seq: (sum(seq), seq))

  @property
  def size(self) -> int:
    return self.base

  @property
  def token_length(self) -> int:
    return self.length

  def logit_mask(self, index: int):
    del index
    return np.ones(self.size, dtype=bool)

  def to_int(self, f: float) -> list[int]:
    if not 0 <= f <= 1:
      raise ValueError(f'f must be between 0 and 1, got {f}')

    f_int = int(f * self.base**self.length)
    if f_int == self.base**self.length:
      f_int -= 1  # Adjust for the edge case when f is exactly 1
    return list(self.all_binary_sequences[f_int])

  def from_int(self, token_ids: list[int]) -> float:
    if len(token_ids) != self.length:
      raise ValueError(f'Length {len(token_ids)} does not match {self.length}.')
    if not all(0 <= tid < self.base for tid in token_ids):
      raise ValueError(f'{token_ids} out of range(0, {self.base})')

    ind = self.all_binary_sequences.index(tuple(token_ids))
    return float(ind) / (self.base**self.length)


@attrs.define
class RepeatingVocab(FloatVocab):
  """Performs error correction by majority voting on decoded tokens."""

  base_vocab: FloatVocab = attrs.field()
  num_repeats: int = attrs.field(default=1)

  @property
  def size(self) -> int:
    return self.base_vocab.size

  @property
  def token_length(self) -> int:
    return self.base_vocab.token_length * self.num_repeats

  def logit_mask(self, index: int):
    true_index = index % self.base_vocab.token_length
    return self.base_vocab.logit_mask(true_index)

  def to_int(self, f: float) -> list[int]:
    return self.base_vocab.to_int(f) * self.num_repeats

  def from_int(self, token_ids: list[int]) -> float:
    if len(token_ids) != self.token_length:
      raise ValueError(
          f'Expected {self.token_length} tokens, got {len(token_ids)}'
      )

    # Reshape repeats into array.
    tokens = np.array(token_ids).reshape(
        self.num_repeats, self.base_vocab.token_length
    )

    # Perform majority voting on each column (token).
    voted_tokens = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), axis=0, arr=tokens
    )

    # Convert the voted tokens to a float using the base vocabulary.
    return self.base_vocab.from_int(voted_tokens.tolist())
