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

from optformer.common.data import featurizers
from optformer.common.data import vocabs
from optformer.common.data.datasets import inference
import seqio
import tensorflow as tf

from absl.testing import absltest


class SeqIOInferenceDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self.vocab = vocabs.AsciiVocab()
    self.raw_data = [{"inputs": "hi", "targets": "bye"}]
    self.dataset = tf.data.Dataset.from_generator(
        lambda: self.raw_data,
        output_types={"inputs": tf.string, "targets": tf.string},
        output_shapes={"inputs": [], "targets": []},
    )

    self.output_features = {
        "inputs": seqio.Feature(vocabulary=self.vocab),
        "targets": seqio.Feature(vocabulary=self.vocab),
    }

    self.task_feature_lengths = {"inputs": 6, "targets": 6}

  def test_dataset_default(self):
    dataset_fn = inference.SeqIOInferenceDatasetFn(
        output_features=self.output_features,
        feature_converter=seqio.feature_converters.EncDecFeatureConverter(
            pack=False
        ),
        task_feature_lengths=self.task_feature_lengths,
    )

    dataset = dataset_fn(self.dataset)
    expected = [{
        "decoder_input_tokens": [0, 98, 121, 101, 1, 0],  # Ends w/ EOS=1
        "decoder_loss_weights": [1, 1, 1, 1, 0, 0],
        "decoder_target_tokens": [98, 121, 101, 1, 0, 0],
        "encoder_input_tokens": [104, 105, 1, 0, 0, 0],
    }]
    seqio.test_utils.assert_dataset(dataset, expected)

  @absltest.skip("Need to disable FeatureConverter validator to run.")
  def test_dataset_with_pack(self):
    # Normally `pack` shouldn't be used during inference. We only show this test
    # to warn the user about what happens if we do use packing.

    dataset_fn = inference.SeqIOInferenceDatasetFn(
        output_features=self.output_features,
        feature_converter=seqio.feature_converters.EncDecFeatureConverter(
            pack=True
        ),
        task_feature_lengths=self.task_feature_lengths,
    )

    dataset = dataset_fn(self.dataset)
    expected = [{
        "decoder_input_tokens": [0, 98, 121, 101, 0, 0],  # No EOS.
        "decoder_loss_weights": [1, 1, 1, 1, 0, 0],
        "decoder_positions": [0, 1, 2, 3, 0, 0],
        "decoder_segment_ids": [1, 1, 1, 1, 0, 0],
        "decoder_target_tokens": [98, 121, 101, 1, 0, 0],
        "encoder_input_tokens": [104, 105, 1, 0, 0, 0],
        "encoder_positions": [0, 1, 2, 0, 0, 0],
        "encoder_segment_ids": [1, 1, 1, 0, 0, 0],
    }]
    seqio.test_utils.assert_dataset(dataset, expected)


class T5XInferenceDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_vocab = vocabs.AsciiVocab()

  def test_e2e(self):
    inference_dataset_fn = inference.T5XInferenceDatasetFn(
        featurizer=featurizers.IdentityFeaturizer(),
        input_vocabulary=self.test_vocab,
        output_vocabulary=self.test_vocab,
        feature_converter=seqio.EncDecFeatureConverter(pack=False),
        max_encoder_sequence_length=1024,
        max_decoder_sequence_length=1024,
    )
    buffer = []
    buffer_dataset = inference_dataset_fn(buffer)

    with self.assertRaises(Exception):
      # Can't iterate an empty buffer.
      next(buffer_dataset.as_numpy_iterator())

    buffer.append("hello")
    np_iterator = buffer_dataset.as_numpy_iterator()
    next(np_iterator)


if __name__ == "__main__":
  absltest.main()
