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

"""Ascii Vocabulary."""

from typing import Optional
import seqio
import tensorflow as tf


class AsciiVocab(seqio.Vocabulary):
  """Copied from seqio/vocabularies_test.py.

  Useful for char-by-char tokenization and testing.
  """

  def __init__(
      self, extra_ids: int = 0, use_eos: bool = True, use_unk: bool = True
  ):
    super().__init__(extra_ids=extra_ids)
    self._extra_ids = extra_ids
    self._use_eos = use_eos
    self._use_unk = use_unk

  @property
  def eos_id(self) -> Optional[int]:
    return 1 if self._use_eos else None

  @property
  def unk_id(self) -> Optional[int]:
    return 2 if self._use_unk else None

  @property
  def _base_vocab_size(self) -> int:
    return 128

  def _encode(self, s: str) -> list[int]:
    return [ord(c) for c in s]

  def _decode(self, ids: list[int]) -> str:
    return "".join("<eos>" if id == 1 else chr(id) for id in ids if id > 0)

  def _encode_tf(self, s: str) -> tf.Tensor:
    return tf.strings.unicode_decode(s, "UTF-8")

  def _decode_tf(self, ids: list[int]) -> tf.Tensor:
    s = tf.strings.unicode_encode(ids, "UTF-8")
    s = tf.strings.regex_replace(s, chr(0), "")
    s = tf.strings.regex_replace(s, chr(1), "<eos>")
    return s
