# Copyright 2022 Google LLC.
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

"""Vocabularies for the vizier project."""
import math
from typing import Optional

from absl import logging
import t5.data
import tensorflow as tf

from sentencepiece import sentencepiece_model_pb2
import sentencepiece as sentencepiece_processor

PAD_ID = t5.data.PAD_ID


class SentencePieceVocabularyWithCustomToken(t5.data.SentencePieceVocabulary):
  """Sentence piece vocabularity that supports custom tokens."""

  def __init__(
      self,
      sentencepiece_model_file,
      extra_ids=None,
      extra_tokens=None,
      normalizer_spec_overrides: Optional[
          sentencepiece_model_pb2.NormalizerSpec
      ] = None,
      reverse_extra_ids: bool = True,
  ):
    """Create a SentencePieceVocabulary.

    Optionally, specify a number of extra ids to add to the end of the
    vocabulary for use as sentinels, or provide a list of extra id strings that
    are guaranteed to be tokenized with a longest match. E.g., with extra tokens
    "12", "3", and "123", the string "123" will be tokenized as "123" instead of
    ("12", "3").

    Args:
      sentencepiece_model_file: a string
      extra_ids: an optional integer
      extra_tokens: an optional list of string.
      normalizer_spec_overrides: If not None, this proto will be merged into the
        model's normalizer and denormalizer specs. Thus, any options set on this
        object will override the values of those options in the loaded model.
      reverse_extra_ids: if True, extra_ids are numbered in descending order, so
        the first extra_id has the highest number. This is done for
        compatibility with span_corruption mask generation in T5.
    """
    self._extra_tokens = extra_tokens
    super().__init__(
        sentencepiece_model_file=sentencepiece_model_file,
        extra_ids=extra_ids,
        normalizer_spec_overrides=normalizer_spec_overrides,
        reverse_extra_ids=reverse_extra_ids,
    )

  def _load_model(self):
    """Load SPM and Python tokenizer."""
    # SentencePieceProcessor::LoadFromSerializedProto is not thread-safe.
    # Without a lock, users may randomly see SIGSEGV on
    # sentencepiece::ModelInterface::pad_piece when using the vocabulary in
    # SeqIO preprocessors.
    with self._load_model_lock:
      # Handle cases where SP can't load the file, but gfile can.
      with tf.io.gfile.GFile(self._sentencepiece_model_file, "rb") as f:
        self._sp_model = f.read()
        model = sentencepiece_model_pb2.ModelProto.FromString(self._sp_model)
        # Add placeholder strings for extra IDs.
        if self._extra_ids:
          # By default, we them in reverse order to match span corruption.
          if self._reverse_extra_ids:
            extra_id_tokens = reversed(range(self._extra_ids))
          else:
            extra_id_tokens = range(self._extra_ids)

          for i in extra_id_tokens:
            model.pieces.add(
                piece=f"▁<extra_id_{i}>",
                score=0.0,
                type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED,
            )

        # Add extra tokens.
        piece2idx = {piece.piece: i for i, piece in enumerate(model.pieces)}
        if self._extra_tokens:
          for extra_token in self._extra_tokens:
            # Add exponential length score for longest match.
            extra_token = f"{extra_token}"
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

        if self._normalizer_spec_overrides is not None:
          model.normalizer_spec.MergeFrom(self._normalizer_spec_overrides)
          model.denormalizer_spec.MergeFrom(self._normalizer_spec_overrides)
        self._sp_model = model.SerializeToString()
      # Load Python tokenizer and ensure the EOS and PAD IDs are correct.
      self._tokenizer = sentencepiece_processor.SentencePieceProcessor()
      self._tokenizer.LoadFromSerializedProto(self._sp_model)
      if self._tokenizer.pad_id() != PAD_ID:
        logging.warning(
            (
                "T5 library uses PAD_ID=%s, which is different from the "
                "sentencepiece vocabulary, which defines pad_id=%s"
            ),
            PAD_ID,
            self._tokenizer.pad_id(),
        )
