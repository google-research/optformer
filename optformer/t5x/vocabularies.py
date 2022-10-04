"""Vocabularies for the vizier project."""
import math

from absl import logging
import t5.data
import tensorflow as tf

from sentencepiece.src import sentencepiece_model_pb2
from sentencepiece.src.python import sentencepiece_processor

PAD_ID = t5.data.PAD_ID


class SentencePieceVocabularyWithCustomToken(t5.data.SentencePieceVocabulary):
  """Sentence piece vocabularity that supports custom tokens."""

  def __init__(self,
               sentencepiece_model_file,
               extra_ids=None,
               extra_tokens=None):
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
    """
    self._extra_tokens = extra_tokens
    super().__init__(
        sentencepiece_model_file=sentencepiece_model_file, extra_ids=extra_ids)

  def _load_model(self):
    """Load SPM and Python tokenizer."""
    # Handle cases where SP can't load the file, but gfile can.
    with tf.io.gfile.GFile(self._sentencepiece_model_file, "rb") as f:
      self._sp_model = f.read()
      # Add placeholder strings for extra IDs.
      model = sentencepiece_model_pb2.ModelProto.FromString(self._sp_model)
      if self._extra_ids:
        # We name them in reverse order to match their use in span corruption.
        for i in reversed(range(self._extra_ids)):
          model.pieces.add(
              piece=f"▁<extra_id_{i}>",
              score=0.0,
              type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED
          )

      # Add extra tokens.
      piece2idx = {piece.piece: i for i, piece in enumerate(model.pieces)}
      if self._extra_tokens:
        for extra_token in self._extra_tokens:
          # Add exponential length score for longest match.
          extra_token = f"{extra_token}"
          score = math.exp(len(extra_token))
          if extra_token in piece2idx:
            logging.info("Overwriting piece: %s, score: %s -> %s", extra_token,
                         model.pieces[piece2idx[extra_token]].score, score)
            model.pieces[piece2idx[extra_token]].score = score
          else:
            logging.info("Adding piece: %s, score: %s", extra_token, score)
            model.pieces.add(
                piece=extra_token,
                score=score,
                type=sentencepiece_model_pb2.ModelProto.SentencePiece
                .USER_DEFINED)
      self._sp_model = model.SerializeToString()

    # Load Python tokenizer and ensure the EOS and PAD IDs are correct.
    self._tokenizer = sentencepiece_processor.SentencePieceProcessor()
    self._tokenizer.LoadFromSerializedProto(self._sp_model)

    if self._tokenizer.pad_id() != PAD_ID:
      logging.warning(
          "T5 library uses PAD_ID=%s, which is different from the "
          "sentencepiece vocabulary, which defines pad_id=%s", PAD_ID,
          self._tokenizer.pad_id())
