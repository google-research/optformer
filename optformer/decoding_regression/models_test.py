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

import functools
import numpy as np
from optformer.decoding_regression import models
from optformer.decoding_regression import vocabs
import tensorflow as tf
from absl.testing import absltest
from absl.testing import parameterized

keras = tf.keras


class ModelTest(parameterized.TestCase):

  @parameterized.parameters((None, None), (5, None), (None, 0.5), (3, 0.1))
  def test_e2e(self, top_k, top_p):
    # pylint: disable=invalid-name
    encoder = tf.keras.models.Sequential([])
    vocab = vocabs.UnnormalizedVocab()
    decoder = models.AttentionDecoder(encoder, vocab)

    num_data = 2000
    feature_dim = 10

    # Generate 10D linear data.
    X = np.random.uniform(size=(num_data, feature_dim))
    weights = np.random.uniform(size=(feature_dim,))
    Y = np.sum(X * weights, axis=-1)
    Y_token_ids = np.array([vocab.to_int(y) for y in Y])

    decoder.compile(
        keras.optimizers.Adam(learning_rate=1e-4),
        loss=functools.partial(
            models.weighted_sparse_categorical_crossentropy,
            weights=np.array([0.3, 0.3, 0.09, 0.01, 0.01, 0.3, 0.5]),
        ),
    )
    decoder.fit(
        [X, Y_token_ids[:, :-1]],
        Y_token_ids,
        batch_size=32,
        epochs=10,
        validation_split=0.2,
    )

    floats = decoder.decode(X[:10], top_k=top_k, top_p=top_p)
    self.assertLen(floats, 10)


if __name__ == "__main__":
  absltest.main()
