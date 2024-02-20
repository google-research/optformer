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

"""Inference-related classes and functions."""

import dataclasses
import functools
from typing import Generic, Optional, TypeVar

import gin
import jax
from optformer.common.data import datasets
from optformer.common.data import featurizers
from t5x import checkpoints as checkpoints_lib
from t5x import gin_utils
from t5x import models
from t5x import train_state as train_state_lib


def initial_train_state(
    model: models.BaseTransformerModel,
) -> train_state_lib.FlaxOptimTrainState:
  """Gives an initial train state.

  This is mainly to re-generate hardware-dependent metadata such as
  "params_axes" (used by Adafactor) which can't be recovered from checkpoints.

  Args:
    model: T5X Model.

  Returns:
    Initial flax optimizer state.
  """

  model_variables = model.get_initial_variables(
      rng=jax.random.PRNGKey(42),
      input_shapes={
          'encoder_input_tokens': (1, 2),
          'decoder_input_tokens': (1, 2),
      },
  )
  return train_state_lib.FlaxOptimTrainState.create(
      model.optimizer_def, model_variables
  )


_T = TypeVar('_T')


@dataclasses.dataclass
class InferenceConfig(Generic[_T]):
  """Minimum set of objects required for end-to-end inference."""

  model: models.BaseTransformerModel
  featurizer: featurizers.Featurizer[_T]
  train_state: train_state_lib.FlaxOptimTrainState

  def get_dataset_fn(
      self, max_inputs_length: int, max_targets_length: int
  ) -> datasets.E2EInferenceDatasetFn[_T]:
    return datasets.E2EInferenceDatasetFn(
        featurizer=self.featurizer,
        input_vocabulary=self.model.input_vocabulary,
        output_vocabulary=self.model.output_vocabulary,
        feature_converter=self.model.FEATURE_CONVERTER_CLS(pack=False),
        max_inputs_length=max_inputs_length,
        max_targets_length=max_targets_length,
    )

  @classmethod
  @functools.cache
  def from_gin_file(
      cls, file: str, step: Optional[int] = None, restore_state: bool = True
  ) -> 'InferenceConfig[_T]':
    """Parses gin file with pre-defined variable names."""
    gin_utils.parse_gin_flags(
        gin_search_paths=['', '.', '/'],
        gin_files=[file],
        gin_bindings=[],
    )

    model = gin.query_parameter('%MODEL').scoped_configurable_fn()

    featurizer_ref = gin.query_parameter('%EVAL_FEATURIZER')
    featurizer = featurizer_ref.scoped_configurable_fn()

    checkpoint = checkpoints_lib.load_t5x_checkpoint(
        gin.query_parameter('%MODEL_DIR'),
        step=step,
    )
    train_state = initial_train_state(model)
    if restore_state:
      train_state = train_state.restore_state(checkpoint)

    return cls(model, featurizer, train_state)
