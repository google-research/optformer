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
from typing import Callable, Sequence
import attrs
import flax
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp
import jaxtyping as jt
from optformer.common.models import embedders
import seqio
from t5x import checkpoints as t5x_checkpoints
from t5x.examples.t5 import network as t5_network
import tensorflow as tf

Tokens = jt.Int[jax.Array, 'B T']
SentencePieceVocabulary = seqio.SentencePieceVocabulary
ReductionFn = Callable[
    [jt.Float[jax.Array, 'B T D']], jt.Float[jax.Array, 'B D']
]


class FlaxT5Embedder(nn.Module):
  """Low level flax embedder. Most logic forked from t5x.examples.t5.network.

  The raw output of an encoder is [B, T, D]. A standard technique is to reduce
  this tensor to length-independent [B, D].

  According to t-SNE, avg-pool and max-pool work well over the length axis.
  """

  t5_config: t5_network.T5Config
  vocab: SentencePieceVocabulary
  reduction_fn: ReductionFn = functools.partial(jnp.average, axis=-2)

  def setup(self):
    cfg = self.t5_config
    self.shared_embedding = t5_network.layers.Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        one_hot=False,  # One-hots blow up memory.
        name='token_embedder',
    )
    self.encoder = t5_network.Encoder(
        config=cfg, shared_embedding=self.shared_embedding
    )

  # NOTE: Consider using nn.remat if backward pass OOMs.
  def __call__(self, tokens: Tokens) -> jt.Float[jax.Array, 'B D']:
    encoder_mask = t5_network.layers.make_attention_mask(
        tokens > 0, tokens > 0, dtype=self.t5_config.dtype
    )
    out = self.encoder(tokens, encoder_mask, deterministic=True)
    mask = tokens != self.vocab.pad_id
    embeddings = out * mask[..., None]
    embeddings: jt.Float[jax.Array, 'B T D'] = embeddings.astype(jnp.float32)
    return self.reduction_fn(embeddings)

  @classmethod
  def from_gin_file(cls, gin_file: str) -> 'FlaxT5Embedder':
    # NOTE: gin parsing saves config globally. Could easily cause bugs if
    # switching between different configs in same process.
    gin.parse_config_file(gin_file)
    cfg_ref = gin.query_parameter('network.Transformer.config')
    cfg = cfg_ref.scoped_configurable_fn()
    vocab = gin.query_parameter('%VOCABULARY').scoped_configurable_fn()
    return FlaxT5Embedder(cfg, vocab)

  @classmethod
  def from_small(cls) -> 'FlaxT5Embedder':
    gin_file = 't5x/examples/t5/t5_1_1/small.gin'
    return cls.from_gin_file(gin_file)

  @classmethod
  def from_base(cls) -> 'FlaxT5Embedder':
    gin_file = 't5x/examples/t5/t5_1_1/base.gin'
    return cls.from_gin_file(gin_file)

  @classmethod
  def from_large(cls) -> 'FlaxT5Embedder':
    gin_file = 't5x/examples/t5/t5_1_1/large.gin'
    return cls.from_gin_file(gin_file)

  @classmethod
  def from_xl(cls) -> 'FlaxT5Embedder':
    gin_file = 't5x/examples/t5/t5_1_1/xl.gin'
    return cls.from_gin_file(gin_file)

  @classmethod
  def from_xxl(cls) -> 'FlaxT5Embedder':
    gin_file = 't5x/examples/t5/t5_1_1/xxl.gin'
    return cls.from_gin_file(gin_file)


SMALL_CKPT = 'gs://pretrained_models/t5x/t5_1_1_small/checkpoint_1000000'
BASE_CKPT = 'gs://pretrained_models/t5x/t5_1_1_base/checkpoint_1000000'
LARGE_CKPT = 'gs://pretrained_models/t5x/t5_1_1_large/checkpoint_1000000'
XL_CKPT = 'gs://pretrained_models/t5x/t5_1_1_xl/checkpoint_1000000'
XXL_CKPT = 'gs://pretrained_models/t5x/t5_1_1_xxl/checkpoint_1000000'


# eq=False allows jitting but won't use jit-compilation cache. Not an issue if
# we only use one instance of the class.
@attrs.define(eq=False)
class T5XTokensEmbedder(embedders.Embedder[Tokens]):
  """Wraps low-level Embedder with checkpoint and params."""

  flax_embedder: FlaxT5Embedder = attrs.field()
  variables: flax.core.scope.FrozenVariableDict = attrs.field()

  @functools.partial(jax.jit, static_argnames=['self'])
  def embed(self, tokens: Tokens) -> jt.Float[jax.Array, 'B D']:
    return self.flax_embedder.apply(self.variables, tokens)

  @property
  def dimension(self) -> int:
    return self.flax_embedder.t5_config.emb_dim

  @classmethod
  def from_pretrained(
      cls, flax_embedder: FlaxT5Embedder, checkpoint_path: str | None
  ) -> 'T5XTokensEmbedder':

    if checkpoint_path:
      train_state = t5x_checkpoints.load_t5x_checkpoint(checkpoint_path)
      params = train_state['target']  # Train state has optimizer state, remove.
      params.pop('decoder')
      variables = flax.core.freeze({'params': params})
    else:
      x = jnp.empty((1, 16))
      params = flax_embedder.init(jax.random.PRNGKey(0), x)
      variables = flax.core.freeze(params)

    return T5XTokensEmbedder(flax_embedder, variables)

  @classmethod
  def from_small(cls) -> 'T5XTokensEmbedder':
    flax_embedder = FlaxT5Embedder.from_small()
    return T5XTokensEmbedder.from_pretrained(flax_embedder, SMALL_CKPT)

  @classmethod
  def from_base(cls) -> 'T5XTokensEmbedder':
    flax_embedder = FlaxT5Embedder.from_base()
    return T5XTokensEmbedder.from_pretrained(flax_embedder, BASE_CKPT)

  @classmethod
  def from_large(cls) -> 'T5XTokensEmbedder':
    flax_embedder = FlaxT5Embedder.from_large()
    return T5XTokensEmbedder.from_pretrained(flax_embedder, LARGE_CKPT)

  @classmethod
  def from_xl(cls) -> 'T5XTokensEmbedder':
    flax_embedder = FlaxT5Embedder.from_xl()
    return T5XTokensEmbedder.from_pretrained(flax_embedder, XL_CKPT)

  @classmethod
  def from_xxl(cls) -> 'T5XTokensEmbedder':
    flax_embedder = FlaxT5Embedder.from_xxl()
    return T5XTokensEmbedder.from_pretrained(flax_embedder, XXL_CKPT)


@attrs.define
class T5XTextEmbedder(embedders.Embedder[str]):
  """T5 embedder with batched text inputs."""

  token_embedder: T5XTokensEmbedder = attrs.field(init=True)

  add_eos: bool = attrs.field(kw_only=True, default=True)
  max_length: int = attrs.field(kw_only=True, default=1024)  # Abbreviated as T.

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
            'input': seqio.Feature(
                self.token_embedder.flax_embedder.vocab, self.add_eos
            )
        },
        copy_pretokenized=False,
        with_eos=self.add_eos,
    )
    ds = seqio.trim_and_pad_dataset(
        ds, feature_lengths={'input': self.max_length}
    )
    ds = ds.batch(batch_size)

    tokens = next(ds.as_numpy_iterator())['input']  # [B, T]
    return self.token_embedder.embed(tokens)  # [B, D]

  @property
  def dimension(self) -> int:
    return self.token_embedder.dimension
