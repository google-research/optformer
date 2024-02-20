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

"""Testing utilities to make life easier."""

from typing import Optional
import seqio
from t5x import adafactor
from t5x import decoding
from t5x import models
from t5x.examples.t5 import network


def small_encoder_decoder(
    vocab: seqio.Vocabulary,
    embedding_size: Optional[int] = None,
) -> models.EncoderDecoderModel:
  """Make small encoder-decoder model."""

  config = network.T5Config(
      vocab_size=embedding_size if embedding_size else vocab.vocab_size,
      emb_dim=2,
      num_heads=1,
      num_encoder_layers=1,
      num_decoder_layers=1,
      head_dim=2,
      mlp_dim=2,
  )
  return models.EncoderDecoderModel(
      module=network.Transformer(config),
      input_vocabulary=vocab,
      output_vocabulary=vocab,
      optimizer_def=adafactor.Adafactor(),
      decode_fn=decoding.temperature_sample,
  )
