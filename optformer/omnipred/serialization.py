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

"""Omnipred-specific serializers."""

import attrs
import gin
from optformer.common import serialization as s_lib
from optformer.validation import runtime
from optformer.vizier import serialization as vs_lib
from vizier import pyvizier as vz


@gin.configurable
@attrs.define(frozen=True, kw_only=True)
class _SuggestionSerializer(s_lib.Serializer[vz.TrialSuggestion]):
  """Serialize a single trial suggestion (with metadata) using dicts."""

  primitive_serializer: s_lib.Serializer[s_lib.PrimitiveType] = attrs.field(
      factory=s_lib.PrimitiveSerializer
  )
  metadata_serializer: s_lib.Serializer[vz.Metadata] = attrs.field(
      factory=vs_lib.MetadataSerializer
  )
  include_metadata: bool = attrs.field(default=False)

  def to_str(self, suggestion: vz.TrialSuggestion, /) -> str:
    param_value_dict = suggestion.parameters.as_dict()
    for k, v in sorted(param_value_dict.items()):  # Sort by param names
      v = float(v) if isinstance(v, int) else v  # Continuify ints
      param_value_dict[k] = v

    out = {'suggestion': param_value_dict}
    if self.include_metadata:
      serialized_metadata = self.metadata_serializer.to_str(suggestion.metadata)
      out['x_metadata'] = serialized_metadata
    return self.primitive_serializer.to_str(out)


@gin.configurable
@attrs.define(frozen=True, kw_only=True)
class _ProblemSerializer(s_lib.Serializer[vz.ProblemStatement]):
  """Serializes problem."""

  primitive_serializer: s_lib.Serializer[s_lib.PrimitiveType] = attrs.field(
      factory=s_lib.PrimitiveSerializer
  )
  metadata_serializer: s_lib.Serializer[vz.Metadata] = attrs.field(
      factory=vs_lib.MetadataSerializer
  )
  include_metadata: bool = attrs.field(default=True)

  def to_str(self, problem: vz.ProblemStatement, /) -> str:
    out = {'objective': problem.metric_information.item().name}

    if self.include_metadata:
      serialized_metadata = self.metadata_serializer.to_str(problem.metadata)
      out['problem_metadata'] = serialized_metadata
    return self.primitive_serializer.to_str(out)


@gin.configurable
@attrs.define(frozen=True)
class OmniPredInputsSerializer(s_lib.Serializer[vz.ProblemAndTrials]):
  """Serializes single suggestion and study_config using raw text."""

  suggestion_serializer: s_lib.Serializer[vz.TrialSuggestion] = attrs.field(
      factory=_SuggestionSerializer
  )
  problem_serializer: s_lib.Serializer[vz.ProblemStatement] = attrs.field(
      factory=_ProblemSerializer
  )

  def to_str(self, study: vz.ProblemAndTrials, /) -> str:
    runtime.assert_length(study.trials, 1)
    trial_suggestion_str = self.suggestion_serializer.to_str(study.trials[0])
    study_config_str = self.problem_serializer.to_str(study.problem)
    return ','.join([trial_suggestion_str, study_config_str])


class OmniPredTargetsSerializer(s_lib.Serializer[vz.ProblemAndTrials]):
  """Uses `MeasurementTokenSerializer` for targets side.

  NOTE: `MeasurementTokenSerializer`s float serializer will be gin-configured.
  """

  def to_str(self, study: vz.ProblemAndTrials, /) -> str:
    runtime.assert_length(study.trials, 1)
    measurement_serializer = vs_lib.MeasurementTokenSerializer(
        study.problem.metric_information
    )
    return measurement_serializer.to_str(study.trials[0])
