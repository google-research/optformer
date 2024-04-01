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

"""Serializers for feedback (measurements) of *all* trials in a study."""

from typing import Tuple

import attrs
from optformer.common import serialization as s_lib
from optformer.original import numeric
from optformer.pyglove import types


@attrs.define
class QuantizedMeasurementSerializer(s_lib.Serializer[types.PyGloveStudy]):
  """Serializes quantized measurements of all trials in the study."""

  objective_target_interval: Tuple[float, float] = attrs.field(
      default=(0.0, 1.0)
  )
  quantizer: numeric.NormalizedQuantizer = attrs.field(
      factory=numeric.NormalizedQuantizer,
      kw_only=True,
  )
  token_serializer: s_lib.UnitSequenceTokenSerializer = attrs.field(
      factory=s_lib.UnitSequenceTokenSerializer,
      kw_only=True,
  )

  def to_str(self, study: types.PyGloveStudy, /) -> str:
    # First, normalizes objectives to 0 and 1.
    objectives = [trial.objective for trial in study.trials]
    min_obj, max_obj = min(objectives), max(objectives)
    scaler = numeric.LinearIntervalScaler(
        source_interval=(min_obj, max_obj),
        target_interval=self.objective_target_interval,
    )
    objectives = [scaler.map(x) for x in objectives]
    quantized = [self.quantizer.map(x) for x in objectives]

    return self.token_serializer.to_str(quantized)
