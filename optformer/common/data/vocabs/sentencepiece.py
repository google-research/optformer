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
from typing import Generic, Iterable, Sequence, TypeVar

from absl import logging
from optformer.common import serialization
import seqio
import tensorflow as tf

from sentencepiece import sentencepiece_model_pb2
import sentencepiece as sentencepiece_processor

TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_data"
)
# Saved example sentencepiece_model_pb2.ModelProto.
VOCAB_TEST_MODEL_FILE = os.path.join(
    TEST_DATA_DIR, "sentencepiece", "sentencepiece.model"
)


class SentencePieceVocabulary(seqio.SentencePieceVocabulary):
  """Sentence piece vocabulary that supports extra tokens."""

  def __init__(
      self,
      sentencepiece_model_file: str,
      extra_tokens: Sequence[str] = (),
  ):
    """Sentencepiece vocabulary which allows extra custom tokens.

    Custom tokens of the form <x> are guaranteed to be tokenzed atomically.

    Args:
      sentencepiece_model_file: File path to a
        sentencepiece_model_pb2.ModelProto.
      extra_tokens: Extra tokens.
    """
    super().__init__(sentencepiece_model_file=sentencepiece_model_file)
    self._extra_tokens = extra_tokens

  def _model_context(self) -> seqio.SentencePieceVocabulary._ModelContext:
    """Overrides parent function to only support extra tokens."""
    if self._model:
      return self._model

    # SentencePieceProcessor::LoadFromSerializedProto is not thread-safe.
    # Without a lock, users may randomly see SIGSEGV on
    # sentencepiece::ModelInterface::pad_piece when using the vocabulary in
    # SeqIO preprocessors.
    with self._load_model_lock:
      # Handle cases where SP can't load the file, but gfile can.
      with tf.io.gfile.GFile(self.sentencepiece_model_file, "rb") as f:
        sp_model = f.read()
        model = sentencepiece_model_pb2.ModelProto.FromString(sp_model)
        # Add extra string tokens.
        piece2idx = {piece.piece: i for i, piece in enumerate(model.pieces)}
        for extra_token in self._extra_tokens:
          # Add exponential length score for longest match.
          score = math.exp(len(extra_token))
          if extra_token in piece2idx:
            logging.info(
                "Overwriting piece: %s, score: %s -> %s",
                extra_token,
                model.pieces[piece2idx[extra_token]].score,
                score,
            )
            model.pieces[piece2idx[extra_token]].score = score
          else:
            logging.info("Adding piece: %s, score: %s", extra_token, score)
            model.pieces.add(
                piece=extra_token,
                score=score,
                type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED,
            )
        sp_model = model.SerializeToString()
      # Load Python tokenizer and ensure the EOS and PAD IDs are correct.
      tokenizer = sentencepiece_processor.SentencePieceProcessor()
      tokenizer.LoadFromSerializedProto(sp_model)
      if tokenizer.pad_id() != seqio.PAD_ID:
        logging.warning(
            (
                "T5 library uses PAD_ID=%s, which is different from the "
                "sentencepiece vocabulary, which defines pad_id=%s"
            ),
            seqio.PAD_ID,
            tokenizer.pad_id(),
        )

      self._model = self._ModelContext(tokenizer=tokenizer, sp_model=sp_model)
      return self._model

  def __str__(self) -> str:
    return (
        f"SentencePieceVocabulary(file={self.sentencepiece_model_file}, "
        f"extra_tokens={self._extra_tokens}, "
        f"spm_md5={hashlib.md5(self.sp_model).hexdigest()})"
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
