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

"""Test the DelimitedTokenVocabulary class."""

from optformer.common.data.vocabs import delimited
from optformer.common.serialization import numeric
from absl.testing import absltest
from absl.testing import parameterized


class DelimitedTokenTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.float_serializer = numeric.DigitByDigitFloatTokenSerializer()
    self.vocab = delimited.DelimitedTokenVocabulary(
        tokens=list(self.float_serializer.all_tokens_used()),
    )
    self.assertEqual(self.vocab.vocab_size, 35)
    self.assertEqual(self.vocab.unk_id, 34)

  @parameterized.parameters(
      ('', []),
      ('<+>', [1]),
      ('<+><1><2><3><E-8>', [1, 4, 5, 6, 15]),  # 0 token is reserved for pad
      ('<10>', [34]),  # nonexistent token gets mapped to the unk id
  )
  def test_encode(self, s: str, e: list[int]):
    self.assertEqual(self.vocab.encode(s), e)

  @parameterized.parameters(
      ([], ''),
      ([1], '<+>'),
      ([1, 0, 0, 0], '<+>'),  # ignore the trailing pad tokens
      ([1, 4, 5, 6, 15], '<+><1><2><3><E-8>'),
      ([34], ''),  # unk token gets decoded to empty string.
  )
  def test_decode(self, e: list[int], s: str):
    self.assertEqual(self.vocab.decode(e), s)


if __name__ == '__main__':
  absltest.main()
