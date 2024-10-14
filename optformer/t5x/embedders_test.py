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

import jax
import jax.numpy as jnp
from optformer.t5x import embedders
from absl.testing import absltest
from absl.testing import parameterized


class FlaxEmbedderTest(parameterized.TestCase):

  @parameterized.parameters(
      (embedders.PoolingReduction,),
      (embedders.AttentionReduction,),
  )
  def test_different_reductions(self, reduction_factory):
    embedder = embedders.FlaxT5Embedder.from_small()
    embedder.reduction_factory = reduction_factory

    batch_size = 2
    length = 10

    tokens = jnp.ones((batch_size, length))
    params = embedder.init(jax.random.PRNGKey(0), tokens)
    result = embedder.apply(params, tokens)

    self.assertIsInstance(result, jnp.ndarray)
    self.assertEqual(result.shape, (batch_size, embedder.t5_config.emb_dim))
    self.assertEqual(result.dtype, jnp.float32)


class T5XTokensEmbedderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.embedder = embedders.T5XTokensEmbedder.from_small()

  def test_embed(self):
    batch_size = 2
    length = 10

    tokens = jnp.ones((batch_size, length))
    result = self.embedder.embed(tokens)

    self.assertIsInstance(result, jnp.ndarray)
    self.assertEqual(result.shape, (batch_size, self.embedder.dimension))
    self.assertEqual(result.dtype, jnp.float32)


class T5XTextEmbedderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    token_embedder = embedders.T5XTokensEmbedder.from_small()
    self.embedder = embedders.T5XTextEmbedder(token_embedder)

  def test_embed(self):
    texts = ['hi', 'hello']
    result = self.embedder.embed(texts)

    self.assertIsInstance(result, jnp.ndarray)
    self.assertEqual(result.shape, (2, self.embedder.dimension))
    self.assertEqual(result.dtype, jnp.float32)


if __name__ == '__main__':
  absltest.main()
