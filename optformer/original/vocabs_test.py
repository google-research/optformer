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

from optformer.original import vocabs
from absl.testing import absltest
from absl.testing import parameterized


class QuantizedVocabTest(parameterized.TestCase):

  @parameterized.parameters(
      (26, vocabs.VOCAB_TEST_MODEL_FILE),
  )
  def test_quantized_indices(self, quantized_index: int, vocab_file: str):
    vocab = vocabs.QuantizedVocabulary(vocab_file, num_quantization_bins=10)
    self.assertEqual(quantized_index, vocab.quantization_vocab_index)

    init_token = vocab.encode("<0>")[0]
    self.assertEqual([init_token, quantized_index], vocab.encode("<0>"))
    self.assertEqual([init_token, quantized_index + 1], vocab.encode("<1>"))
    self.assertEqual([init_token, quantized_index + 9], vocab.encode("<9>"))
    self.assertEqual(
        [init_token, quantized_index + 3, quantized_index + 5],
        vocab.encode("<3><5>"),
    )

    self.assertEqual("<0>", vocab.decode([quantized_index]))
    self.assertEqual("<1>", vocab.decode([quantized_index + 1]))
    self.assertEqual("<9>", vocab.decode([quantized_index + 9]))

    self.assertEqual(0, vocab.decode_to_object([quantized_index]))
    self.assertEqual(1, vocab.decode_to_object([quantized_index + 1]))
    self.assertEqual(9, vocab.decode_to_object([quantized_index + 9]))

  @parameterized.parameters(
      (vocabs.VOCAB_TEST_MODEL_FILE,),
  )
  def test_equivalence(self, vocab_file: str):
    vocab1 = vocabs.QuantizedVocabulary(vocab_file, num_quantization_bins=10)
    vocab2 = vocabs.QuantizedVocabulary(vocab_file, num_quantization_bins=10)

    eq = vocab1 == vocab2
    self.assertTrue(eq)


if __name__ == "__main__":
  absltest.main()
