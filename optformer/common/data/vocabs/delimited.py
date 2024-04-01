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

"""A custom vocabulary whose tokens are all separator-delimited."""

from typing import Optional, Sequence, Tuple
import seqio
import tensorflow as tf
import tensorflow_text as tf_text


class DelimitedTokenVocabulary(seqio.Vocabulary):
  """A custom vocabulary whose tokens are all separator-delimited."""

  DELIMITERS: Tuple[str, str] = ('<', '>')

  def __init__(self, tokens: Sequence[str]):
    """Creates a lookup array to map integer indices to strings.

    Args:
      tokens: a sequence of tokens to each surrounded by a pair of separators.
    """
    self._tokens = list(tokens)

    left_d, right_d = self.DELIMITERS
    if not all(x.startswith(left_d) for x in self._tokens):
      raise ValueError(f'Tokens must start with {left_d=}')
    if not all(x.endswith(right_d) for x in self._tokens):
      raise ValueError(f'Tokens must end with {right_d=}')
    self._unk_id = len(self._tokens) + 1

    # Pad string, followed by regular token strings, followed_by
    # unknown tokens (which are mapped to empty string)
    self._id_to_tokens_tf = tf.constant(
        [''] + list(self._tokens) + [''], dtype=tf.string
    )

    # We map '' to the pad token which has value 0.
    token_to_id_initializer = tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant([''] + list(self._tokens), dtype=tf.string),
        values=tf.constant(range(len(self._tokens) + 1), dtype=tf.int64),
    )

    # Any unrecognizable string is automatically mapped to the oov bucket
    # with return value self._unk_id.
    self._token_to_id_tf = tf.lookup.StaticVocabularyTable(
        token_to_id_initializer, num_oov_buckets=1
    )

    super().__init__()

  @property
  def bos_id(self) -> Optional[int]:
    return None

  @property
  def eos_id(self) -> Optional[int]:
    return None

  @property
  def unk_id(self) -> Optional[int]:
    return self._unk_id

  @property
  def _base_vocab_size(self):
    return len(self._tokens) + 2  # add pad + unknown token

  def _encode(self, s: str) -> Sequence[int]:
    return list(self._encode_tf(tf.constant(s, dtype=tf.string)).numpy())

  def _decode(self, ids: Sequence[int]) -> str:
    return str(
        self._decode_tf(tf.constant(ids, dtype=tf.int32)).numpy(), 'utf-8'
    )

  def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
    _, right_d = self.DELIMITERS
    splitter = tf_text.RegexSplitter(split_regex=right_d)
    tokens_split = splitter.split(s)
    tokens_split = tf.reshape(tokens_split, (-1,))
    tokens_split = tokens_split + tf.constant(right_d, dtype=tf.string)
    ids_tf = self._token_to_id_tf.lookup(tokens_split)

    ids_tf = tf.dtypes.cast(ids_tf, tf.int32)
    return ids_tf

  def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
    return tf.strings.join(tf.gather(self._id_to_tokens_tf, ids), '')

  def __str__(self) -> str:
    return f'FloatTokenVocabulary({self._tokens}'
