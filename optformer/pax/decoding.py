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

"""Text-to-text decoder."""

import functools
from typing import Sequence
from absl import logging
import attrs
import jax
from optformer.common.data import datasets
from optformer.pax import inference
from paxml import partitioning
from paxml import seqio_input
from praxis import base_input
from praxis import base_layer
from praxis import decoder_hparams as decoder_hparams_lib
from praxis import pytypes
import seqio
import tensorflow as tf

NestedJTensor = pytypes.NestedJTensor
NestedMap = pytypes.NestedMap
DecoderHparams = decoder_hparams_lib.DecoderHParams


# eq=False allows jitting but won't use jit-compilation cache. Not an issue if
# we only use one instance of the class.
@attrs.define(eq=False)
class TextToTextDecoder:
  """Text-to-text decoder."""

  infer_cfg: inference.InferenceConfig = attrs.field()

  batch_size: int = attrs.field(default=1, kw_only=True)
  inputs_length: int = attrs.field(default=4096, kw_only=True)
  targets_length: int = attrs.field(default=4096, kw_only=True)

  _infer_input: base_input.BaseInput = attrs.field(init=False)
  _input_sharding: jax.sharding.NamedSharding = attrs.field(init=False)

  def __attrs_post_init__(self):
    infer_inp_hps = self.infer_cfg.experiment.datasets()[0]
    self._infer_input = infer_inp_hps.Instantiate(batch_size=self.batch_size)
    mixture_name = self._infer_input.mixture_name
    try:
      provider = seqio.TaskRegistry.get(mixture_name)
    except ValueError:
      provider = seqio.MixtureRegistry.get(mixture_name)
    vocab = provider.output_features['targets'].vocabulary

    decoder_params: DecoderHparams = self.infer_cfg.task.model.decoder_tpl
    decoder_params.seqlen = self.inputs_length + self.targets_length
    decoder_params.eos_id = vocab.eos_id
    decoder_params.fprop_for_prefix = True
    decoder_params.max_decode_steps = self.targets_length

    output_features = {
        'inputs': seqio.Feature(vocab, add_eos=False),
        'targets': seqio.Feature(vocab, add_eos=False),
    }
    feature_converter = seqio_input.LanguageModelFeatures(
        pack=False, bos_id=vocab.bos_id, eos_id=vocab.eos_id
    )
    task_feature_lengths = {
        'inputs': self.inputs_length,
        'targets': self.targets_length,
    }
    self._inference_dataset_fn = datasets.SeqIOInferenceDatasetFn(
        output_features=output_features,
        feature_converter=feature_converter,
        task_feature_lengths=task_feature_lengths,
    )
    self._input_sharding = jax.sharding.NamedSharding(
        self.infer_cfg.partitioner.global_mesh,
        partitioning.PartitionSpec('data'),
    )

    logging.info('Warming up...')
    warmup = 'What is one plus five?'
    logging.info('Prompt: %s', warmup)
    logging.info('Model output: %s', self.decode(prompts=[warmup]))

  def decode(self, prompts: Sequence[str]) -> Sequence[str]:
    """Returns model decoding output given input prompts.

    Args:
      prompts: Batch of strings to be parallel-evaluated.
    """

    ##### DATA PROCESSING #####
    features = tf.data.Dataset.from_generator(
        lambda: ({'inputs': p, 'targets': ''} for p in prompts),
        output_types={'inputs': tf.string, 'targets': tf.string},
    )
    ds = self._inference_dataset_fn(features)
    ds = ds.batch(len(prompts))
    batch = next(ds.as_numpy_iterator())
    ##### END DATA PROCESSING #####

    partitioned_batch = jax.device_put(batch, self._input_sharding)
    decode_out = self._jax_decode(
        self.infer_cfg.train_state.mdl_vars, partitioned_batch
    )
    # TODO: add typing to outputs
    _, s_tuples, _ = self.infer_cfg.task.model.process_decode_out(
        self._infer_input, decode_out
    )
    s_dicts = [s_tuple[1] for s_tuple in s_tuples]
    return [s_dict['decoded_substr'] for s_dict in s_dicts]

  @functools.partial(jax.jit, static_argnames=['self'])
  def _jax_decode(
      self, variables: NestedJTensor, inputs: NestedMap
  ) -> NestedMap:
    """Jittable decoding function."""
    context_p = base_layer.JaxContext.HParams(do_eval=True)
    with base_layer.JaxContext(context_p):
      scalars_and_output, _ = self.infer_cfg.task.model.apply(
          variables,
          inputs,
          mutable=True,
          method=self.infer_cfg.task.model.decode,
      )
    return scalars_and_output[1]
