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

from optformer.common.data.filters import features
import tensorflow as tf
from absl.testing import absltest


class StringLengthFilterTest(absltest.TestCase):

  def test_e2e(self):
    filt = features.TokenLengthFilter(
        max_token_lengths={'inputs': 10, 'targets': 10}
    )
    good_features = {
        'inputs': tf.constant('hello', dtype=tf.string),
        'targets': tf.constant('world', dtype=tf.string),
    }
    self.assertTrue(filt(good_features))

    bad_features = {
        'inputs': tf.constant('tooooooooooooooooooooo', dtype=tf.string),
        'targets': tf.constant('looooooooooooooooooong', dtype=tf.string),
    }
    with self.assertRaises(ValueError):
      filt(bad_features)


if __name__ == '__main__':
  absltest.main()
