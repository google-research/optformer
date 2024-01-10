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

from typing import List

from optformer.common.data.vocabs import ascii as ascii_lib

from absl.testing import absltest
from absl.testing import parameterized


class AsciiTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.vocab = ascii_lib.AsciiVocab()

  @parameterized.parameters(
      ('', []),
      ('a', [97]),
      ('abc', [97, 98, 99]),
  )
  def test_encode(self, s: str, e: List[int]):
    self.assertEqual(self.vocab.encode(s), e)

  @parameterized.parameters(
      ([0], ''),
      ([1], '<eos>'),
      ([97], 'a'),
      ([0, 97], 'a'),
      ([1, 97], '<eos>'),  # NOTE: <eos>=1 terminates decoding.
      ([51, 52, 53], '345'),
  )
  def test_decode(self, e: List[int], s: str):
    self.assertEqual(self.vocab.decode(e), s)

  @parameterized.parameters(('a',), ('abc',), ('hello',))
  def test_reversibility(self, s: str):
    # Reversibility will only occur for non-special (PAD/EOS/UNK) tokens.
    encoded = self.vocab.encode(s)
    decoded = self.vocab.decode(encoded)
    self.assertEqual(decoded, s)


if __name__ == '__main__':
  absltest.main()
