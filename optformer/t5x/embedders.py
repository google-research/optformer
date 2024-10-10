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

"""T5X-based embedders."""

import functools
from typing import Callable, Sequence, cast
import attrs
import flax
import gin
import jax
import jax.numpy as jnp
import jaxtyping as jt
from optformer.common.models import embedders
import seqio
from t5x import checkpoints as t5x_checkpoints
from t5x import models
from t5x.examples.t5 import network as t5_network
import tensorflow as tf

Tokens = jt.Int[jax.Array, 'B L']
SentencePieceVocabulary = seqio.SentencePieceVocabulary


# eq=False allows jitting but won't use jit-compilation cache. Not an issue if
# we only use one instance of the class.
@attrs.define(eq=False)
class T5XTokensEmbedder(embedders.Embedder[Tokens]):
  """T5 embedder with token inputs.

  The raw output of an encoder is [B, L, D]. A standard technique is to reduce
  this tensor to length-independent [B, D].

  According to t-SNE, avg-pool and max-pool work well over the length axis.
  """

  model: models.EncoderDecoderModel = attrs.field(init=True)
  checkpoint_path: str | None = attrs.field(init=True)

  reduction_fn: Callable[[jax.Array], jax.Array] = attrs.field(
      init=True, kw_only=True, default=functools.partial(jnp.average, axis=-2)
  )

  # Created downstream.
  network: t5_network.Transformer = attrs.field(init=False)
  variables: flax.core.scope.VariableDict = attrs.field(init=False)
  vocab: SentencePieceVocabulary = attrs.field(init=False)

  def __attrs_post_init__(self):
    self.network = cast(t5_network.Transformer, self.model.module)

    if self.checkpoint_path:
      full_params = t5x_checkpoints.load_t5x_checkpoint(self.checkpoint_path)
    else:
      x = jnp.empty((1, 16))
      full_params = self.network.init(jax.random.PRNGKey(0), x, x, x)

    params = full_params['target']
    params.pop('decoder')
    self.variables = {'params': params}
    self.vocab = cast(SentencePieceVocabulary, self.model.input_vocabulary)

  @functools.partial(jax.jit, static_argnames=['self'])
  def embed(self, tokens: Tokens) -> jt.Float[jax.Array, 'B D']:
    out = self.network.apply(
        self.variables, tokens, method=self.network.encode, enable_dropout=False
    )

    mask = tokens != self.vocab.pad_id
    embeddings = out * mask[..., None]
    embeddings: jt.Float[jax.Array, 'B L D'] = embeddings.astype(jnp.float32)

    # [B, L, D] -> [B, D] via reduction.
    return self.reduction_fn(embeddings)

  @property
  def dimension(self) -> int:
    return self.network.config.emb_dim

  @classmethod
  def from_pretrained(
      cls, gin_file: str, checkpoint_path: str
  ) -> 'T5XTokensEmbedder':
    # NOTE: gin parsing saves config globally. Could easily cause bugs if
    # switching between different configs.
    gin.parse_config_file(gin_file)
    model = gin.query_parameter('%MODEL').scoped_configurable_fn()
    return T5XTokensEmbedder(model, checkpoint_path)

  @classmethod
  def from_small(cls) -> 'T5XTokensEmbedder':
    gin_file = 't5x/examples/t5/t5_1_1/small.gin'
    ckpt = 'gs://t5-data/pretrained_models/t5x/t5_1_1_small/checkpoint_1000000'
    return T5XTokensEmbedder.from_pretrained(gin_file, ckpt)

  @classmethod
  def from_base(cls) -> 'T5XTokensEmbedder':
    gin_file = 't5x/examples/t5/t5_1_1/base.gin'
    ckpt = 'gs://pretrained_models/t5x/t5_1_1_base/checkpoint_1000000'
    return T5XTokensEmbedder.from_pretrained(gin_file, ckpt)

  @classmethod
  def from_large(cls) -> 'T5XTokensEmbedder':
    gin_file = 't5x/examples/t5/t5_1_1/large.gin'
    ckpt = 'gs://pretrained_models/t5x/t5_1_1_large/checkpoint_1000000'
    return T5XTokensEmbedder.from_pretrained(gin_file, ckpt)

  @classmethod
  def from_xl(cls) -> 'T5XTokensEmbedder':
    gin_file = 't5x/examples/t5/t5_1_1/xl.gin'
    ckpt = 'gs://pretrained_models/t5x/t5_1_1_xl/checkpoint_1000000'
    return T5XTokensEmbedder.from_pretrained(gin_file, ckpt)

  @classmethod
  def from_xxl(cls) -> 'T5XTokensEmbedder':
    gin_file = 't5x/examples/t5/t5_1_1/xxl.gin'
    ckpt = 'gs://pretrained_models/t5x/t5_1_1_xxl/checkpoint_1000000'
    return T5XTokensEmbedder.from_pretrained(gin_file, ckpt)


@attrs.define
class T5XTextEmbedder(embedders.Embedder[str]):
  """T5 embedder with batched text inputs."""

  token_embedder: T5XTokensEmbedder = attrs.field(init=True)

  add_eos: bool = attrs.field(kw_only=True, default=True)
  max_length: int = attrs.field(kw_only=True, default=1024)  # Abbreviated as L.

  def embed(self, texts: Sequence[str]) -> jt.Float[jax.Array, 'B D']:
    """Tokenizes texts and sends to T5 encoder."""
    batch_size = len(texts)

    ds = tf.data.Dataset.from_generator(
        lambda: [{'input': t} for t in texts],
        output_types={'input': tf.string},
        output_shapes={'input': []},
    )
    ds = seqio.preprocessors.tokenize(
        ds,
        output_features={
            'input': seqio.Feature(self.token_embedder.vocab, self.add_eos)
        },
        copy_pretokenized=False,
        with_eos=self.add_eos,
    )
    ds = seqio.trim_and_pad_dataset(
        ds, feature_lengths={'input': self.max_length}
    )
    ds = ds.batch(batch_size)

    tokens = next(ds.as_numpy_iterator())['input']  # [B, L]
    return self.token_embedder.embed(tokens)  # [B, D]

  @property
  def dimension(self) -> int:
    return self.token_embedder.dimension
