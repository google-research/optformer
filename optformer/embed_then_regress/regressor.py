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

"""String-level wrapper around Jax regressor."""

from typing import Any, Callable, Sequence
import attrs
import flax.typing as flax_typing
import jax
import jaxtyping as jt
import numpy as np
from optformer.embed_then_regress import checkpointing as ckpt_lib
from optformer.embed_then_regress import configs
from optformer.embed_then_regress import icl_transformer
from optformer.embed_then_regress import normalization
import seqio
import tensorflow as tf
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


# TODO: Maybe refactor omnipred2 regressor base class.
@attrs.define
class StatefulICLRegressor:
  """Stateful ICL regressor which operates on strings and floats."""

  model: icl_transformer.ICLTransformer = attrs.field()
  params: flax_typing.FrozenVariableDict = attrs.field()
  vocab: seqio.Vocabulary = attrs.field()

  max_trial_length: int = attrs.field(default=300, kw_only=True)  # L
  max_token_length: int = attrs.field(default=256, kw_only=True)  # T

  warper: normalization.StatefulWarper = attrs.field(
      factory=normalization.default_warper, kw_only=True
  )

  # Internal state containing tokens.
  _all_xt: jt.Int[np.ndarray, 'L T'] = attrs.field(init=False)
  _all_yt: jt.Float[np.ndarray, 'L'] = attrs.field(init=False)
  _mt: jt.Int[np.ndarray, 'T'] = attrs.field(init=False)
  _num_prev: int = attrs.field(init=False)
  _jit_apply: Callable[..., Any] = attrs.field(init=False)

  def __attrs_post_init__(self):
    self.reset()
    self._jit_apply = jax.jit(
        self.model.apply, static_argnames=('deterministic',)
    )

  def predict(self, xs: Sequence[str]) -> tfd.Distribution:
    """Returns prediction in normalized/warped space."""
    num_query = len(xs)

    temp_xt = np.copy(self._all_xt)
    temp_xt[self._num_prev : self._num_prev + num_query] = self._tokenize(xs)

    temp_yt = np.copy(self._all_yt)
    temp_yt = self.warper.warp(temp_yt)

    temp_mt = np.copy(self._mt)

    mask = np.ones(self.max_trial_length, dtype=bool)
    mask[self._num_prev :] = False

    # Need to add batch dimension to all inputs.
    mean, std = self._jit_apply(
        self.params,
        x=np.expand_dims(temp_xt, axis=0),  # [B=1, L, T],
        y=np.expand_dims(temp_yt, axis=0),  # [B=1, L],
        metadata=np.expand_dims(temp_mt, axis=0),  # [B=1, T],
        mask=np.expand_dims(mask, axis=0),  # [B=1, L],
        deterministic=True,
    )

    mean, std = np.squeeze(mean, axis=0), np.squeeze(std, axis=0)
    mean = mean[self._num_prev : self._num_prev + num_query]
    std = std[self._num_prev : self._num_prev + num_query]
    return tfd.Normal(mean, std)

  def absorb(self, xs: Sequence[str], ys: Sequence[float]):
    if len(xs) != len(ys):
      raise ValueError('xs and ys must have the same length.')
    num_pts = len(xs)
    self._all_xt[self._num_prev : self._num_prev + num_pts] = self._tokenize(xs)
    self._all_yt[self._num_prev : self._num_prev + num_pts] = np.array(ys)
    self._num_prev += num_pts

    self.warper.train(self._all_yt[: self._num_prev])

  def set_metadata(self, metadata: str) -> None:
    self._mt = self._tokenize([metadata])[0]

  def reset(self) -> None:
    self._all_xt = np.zeros(
        (self.max_trial_length, self.max_token_length), dtype=np.int32
    )
    self._all_yt = np.zeros(self.max_trial_length, dtype=np.float32)
    self._mt = np.zeros(self.max_token_length, dtype=np.int32)
    self._num_prev = 0

  def _tokenize(self, ss: Sequence[str]) -> jt.Int[np.ndarray, 'S T']:
    """Converts ss (strings) to tokens."""
    batch_size = len(ss)
    ds = tf.data.Dataset.from_generator(
        lambda: [{'input': s} for s in ss],
        output_types={'input': tf.string},
        output_shapes={'input': []},
    )
    ds = seqio.preprocessors.tokenize(
        ds, output_features={'input': seqio.Feature(self.vocab)}
    )
    ds = seqio.trim_and_pad_dataset(
        ds, feature_lengths={'input': self.max_token_length}
    )
    ds = ds.batch(batch_size)
    tokens = next(ds.as_numpy_iterator())['input']
    return tokens.astype(np.int32)

  @classmethod
  def from_checkpoint_and_configs(
      cls,
      ckpt_path: str,
      model_config: configs.ModelConfig,
      embedder_config: configs.T5EmbedderConfig,
      data_config: configs.DataConfig,
  ) -> 'StatefulICLRegressor':
    """Creates a regressor from configs."""

    return StatefulICLRegressor(
        model_config.create_model(embedder_config=embedder_config),
        params=ckpt_lib.restore_train_state(ckpt_path)['params'],
        vocab=data_config.create_vocab(),
        max_token_length=data_config.max_token_length,
    )
