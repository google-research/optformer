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

"""Custom SentencePiece vocabularies."""

import abc
import hashlib
import math
import os
import threading
from typing import ClassVar, Generic, Iterable, Sequence, TypeVar

from absl import logging
from optformer.common import serialization
import seqio
import tensorflow as tf
import tensorflow_text as tf_text

from sentencepiece import sentencepiece_model_pb2
import sentencepiece as sentencepiece_processor

TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_data"
)
# Saved example sentencepiece_model_pb2.ModelProto.
VOCAB_TEST_MODEL_FILE = os.path.join(
    TEST_DATA_DIR, "sentencepiece", "sentencepiece.model"
)


class SentencePieceVocabulary(seqio.Vocabulary):
  """SentencePiece vocabulary that supports extra tokens.

  Forked from seqio for stability + to avoid breakages.
  """

  # SentencePieceProcessor::LoadFromSerializedProto is not thread-safe.
  _load_model_lock: ClassVar[threading.Lock] = threading.Lock()

  def __init__(
      self,
      sentencepiece_model_file: str,
      extra_tokens: Sequence[str] = (),
  ):
    """Sentencepiece vocabulary which allows extra custom tokens.

    Custom tokens of the form <x> are guaranteed to be tokenzed atomically.

    Args:
      sentencepiece_model_file: Path to a sentencepiece_model_pb2.ModelProto.
      extra_tokens: Extra tokens.
    """
    self._sentencepiece_model_file = sentencepiece_model_file
    self._extra_tokens = extra_tokens

    with tf.io.gfile.GFile(sentencepiece_model_file, "rb") as f:
      model = sentencepiece_model_pb2.ModelProto.FromString(f.read())
    # Add extra string tokens.
    piece2idx = {piece.piece: i for i, piece in enumerate(model.pieces)}
    for extra_token in self._extra_tokens:
      # Add exponential length score for longest match.
      score = math.exp(len(extra_token))
      if extra_token in piece2idx:
        model.pieces[piece2idx[extra_token]].score = score
      else:
        model.pieces.add(
            piece=extra_token,
            score=score,
            type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED,
        )
    self._sp_model: bytes = model.SerializeToString()

    with self._load_model_lock:
      # Load Python tokenizer and ensure the EOS and PAD IDs are correct.
      self._tokenizer = sentencepiece_processor.SentencePieceProcessor()
      self._tokenizer.LoadFromSerializedProto(self._sp_model)
      if self._tokenizer.pad_id() != seqio.PAD_ID:
        logging.warning(
            "SeqIO PAD_ID=%s but SentencePiece pad_id=%s",
            seqio.PAD_ID,
            self._tokenizer.pad_id(),
        )
      self._tf_tokenizer = tf_text.SentencepieceTokenizer(model=self._sp_model)

  @property
  def bos_id(self) -> int | None:
    return self._tokenizer.bos_id()

  @property
  def eos_id(self) -> int | None:
    return self._tokenizer.eos_id()

  @property
  def unk_id(self) -> int | None:
    return self._tokenizer.unk_id()

  @property
  def vocab_size(self):
    return self._base_vocab_size

  @property
  def _base_vocab_size(self):
    return self._tokenizer.GetPieceSize()

  def _encode(self, s: str) -> Sequence[int]:
    return self._tokenizer.EncodeAsIds(s)

  def _decode(self, ids):
    # convert all the extra ids (sentinels) to UNK=2
    unk_id = self._tokenizer.unk_id()
    piece_size = self._tokenizer.GetPieceSize()
    ids = [unk_id if i >= piece_size else int(i) for i in ids]
    return self._tokenizer.DecodeIds(ids)

  def _encode_tf(self, s):
    return self._tf_tokenizer.tokenize(s)

  def _decode_tf(self, ids):
    return self._tf_tokenizer.detokenize(ids)

  def __eq__(self, other):
    if self._sp_model is None:
      return False
    if not isinstance(other, SentencePieceVocabulary):
      return False
    try:
      their_md5 = hashlib.md5(other._sp_model).hexdigest()
    except AttributeError:  # If other has no sp_model attribute
      return False
    our_md5 = hashlib.md5(self._sp_model).hexdigest()
    return our_md5 == their_md5

  def __str__(self) -> str:
    return (
        f"SentencePieceVocabulary(file={self._sentencepiece_model_file}, "
        f"extra_tokens={self._extra_tokens}, "
        f"spm_md5={hashlib.md5(self._sp_model).hexdigest()})"
    )

  @property
  def extra_tokens(self) -> Sequence[str]:
    return self._extra_tokens

  @property
  def initial_extra_token_id(self) -> int:
    """Beginning ID of the extra tokens."""
    return self.vocab_size - len(self._extra_tokens)

  def extra_token_id(self, extra_token: str) -> int:
    return self._extra_tokens.index(extra_token) + self.initial_extra_token_id


_T = TypeVar("_T")


class HybridVocabulary(SentencePieceVocabulary, Generic[_T], abc.ABC):
  """Normal vocabulary + deserializer to decode back into actual objects."""

  @property
  @abc.abstractmethod
  def deserializer(self) -> serialization.Deserializer[_T]:
    """Returns deserializer."""

  def decode_to_object(self, ids: Iterable[int]) -> _T:
    return self.deserializer.from_str(self.decode(ids))
