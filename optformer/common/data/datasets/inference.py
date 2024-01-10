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

"""T5 datasets."""

from typing import Any, Generic, Iterable, TypeVar

import attrs
from optformer.common.data import featurizers
from optformer.common.data.datasets import base
import seqio
import tensorflow as tf


def _inference_feature_converter_validator(
    instance: Any,
    attribute: attrs.Attribute,
    value: seqio.FeatureConverter,
) -> None:
  del instance, attribute
  if value.pack:
    raise ValueError(f"Inference should not use packing. Given: {value.pack}")


@attrs.define(kw_only=True)
class SeqIOInferenceDatasetFn(base.DatasetFn[tf.data.Dataset]):
  """Only meant to be used during inference.

  Performs tokenization + feature conversion on a featurized dataset via the
  SeqIO Task API, with all other kwargs optimized for inference.
  """

  # SeqIO's Feature contains the actual vocabulary. The string keys should match
  # the source dataset's keys.
  output_features: dict[str, seqio.Feature] = attrs.field(init=True)

  # For final conversion to e.g. T5X model inputs. Normally the T5X model
  # already contains a feature converter, but it's only used by training
  # pipelines and not at all in the model's inference API (i.e. `predict_batch`)
  # so we need to feature-convert the data ourselves.
  feature_converter: seqio.FeatureConverter = attrs.field(
      init=True, validator=_inference_feature_converter_validator
  )

  # Length of output tensors. The FeatureConverter should add '0' paddings if
  # the input data is too short. The string keys should match the source
  # dataset's keys.
  task_feature_lengths: dict[str, int] = attrs.field(init=True)

  def __call__(self, source: tf.data.Dataset) -> tf.data.Dataset:
    ds = source
    ds = seqio.preprocessors.tokenize(ds, self.output_features)
    ds = seqio.preprocessors.append_eos_after_trim(ds, self.output_features)
    ds = seqio.trim_dataset(ds, self.task_feature_lengths, self.output_features)
    ds = self.feature_converter(ds, self.task_feature_lengths)
    return ds


_S = TypeVar("_S")


# TODO: Should this just be merged w/ SeqIOInferenceDatasetFn?
@attrs.define(init=False)
class T5XInferenceDatasetFn(Generic[_S], base.DatasetFn[Iterable[_S]]):
  """Converts a batch of Python objects into a T5X Model input for inference.

  Python objects must be featurized first.
  """

  featurizer: featurizers.Featurizer[_S] = attrs.field(kw_only=True)
  tokenizer_and_converter: SeqIOInferenceDatasetFn = attrs.field(kw_only=True)

  def __init__(
      self,
      featurizer: featurizers.Featurizer[_S],
      input_vocabulary: seqio.Vocabulary,
      output_vocabulary: seqio.Vocabulary,
      feature_converter: seqio.FeatureConverter,
      max_encoder_sequence_length: int,
      max_decoder_sequence_length: int,
  ):
    """Custom init to reduce field ownership and align w/ T5X gin usage."""
    output_features = {
        # Input format is already rigorously defined, no need for EOS.
        "inputs": seqio.Feature(vocabulary=input_vocabulary, add_eos=False),
        # We control the decoding length ourselves, no need for EOS.
        "targets": seqio.Feature(vocabulary=output_vocabulary, add_eos=False),
    }
    tokenizer_and_converter = SeqIOInferenceDatasetFn(
        output_features=output_features,
        feature_converter=feature_converter,
        task_feature_lengths={
            "inputs": max_encoder_sequence_length,
            "targets": max_decoder_sequence_length,
        },
    )
    self.__attrs_init__(
        featurizer=featurizer, tokenizer_and_converter=tokenizer_and_converter
    )

  def __call__(self, source: Iterable[_S]) -> tf.data.Dataset:
    """Featurizes + tokenizes objects from a buffer.

    Args:
      source: Ideally a live buffer which should not be deleted.

    Returns:
      tf.Dataset holding a reference to the generator / buffer.
    """
    generator_dataset = tf.data.Dataset.from_generator(
        lambda: (self.featurizer.to_features(s) for s in source),
        output_types=self.featurizer.output_types,
        output_shapes=self.featurizer.output_shapes,
    )
    return self.tokenizer_and_converter(generator_dataset)
