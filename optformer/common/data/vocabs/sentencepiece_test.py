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

"""Tests for sentencepiece.py."""

import numpy as np
from optformer.common.data.vocabs import sentencepiece
import tensorflow as tf
from absl.testing import absltest


def _decode_tf(vocab, tokens):
  results = vocab.decode_tf(tf.constant(tokens, tf.int32)).numpy()

  def _apply(fun, sequence):
    if isinstance(sequence, (list, np.ndarray)):
      return [_apply(fun, x) for x in sequence]
    else:
      return fun(sequence)

  return _apply(lambda x: x.decode("UTF-8"), results)


class SentencePieceTest(absltest.TestCase):

  def test_extra_tokens(self):
    extra_tokens = ("hello", "goodbye")
    vocab = sentencepiece.SentencePieceVocabulary(
        sentencepiece.VOCAB_TEST_MODEL_FILE, extra_tokens=extra_tokens
    )
    # Original length is 26.
    self.assertEqual(26 + len(extra_tokens), vocab.vocab_size)

    self.assertEqual("", vocab.decode([3]))  # Empty string.
    self.assertEqual("v", vocab.decode([25]))

    test_string = "hellogoodbyev"
    test_tokens = [26, 27, 25]

    self.assertEqual(test_string, vocab.decode(test_tokens))
    self.assertEqual(test_string, _decode_tf(vocab, test_tokens))

    # Empty string is token 3, hence it appears at beginning for encoding.
    self.assertSequenceEqual([3] + test_tokens, vocab.encode(test_string))
    self.assertSequenceEqual(
        [3] + test_tokens, tuple(vocab.encode_tf(test_string).numpy())
    )


if __name__ == "__main__":
  absltest.main()
