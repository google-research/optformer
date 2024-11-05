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

"""Serializers for Vizier data."""

import attrs
from optformer.common import serialization as s_lib
from optformer.vizier import serialization as vs_lib
from vizier import pyvizier as vz

SuggestionSerializer = s_lib.Serializer[vz.TrialSuggestion]
ProblemSerializer = s_lib.Serializer[vz.ProblemAndTrials]


@attrs.define
class XSerializer(SuggestionSerializer):
  """Basic serializer for Vizier parameters and metadata."""

  # Needed for parameter ordering synchronization.
  search_space: vz.SearchSpace = attrs.field()

  primitive_serializer: s_lib.PrimitiveSerializer = attrs.field(
      factory=s_lib.PrimitiveSerializer, kw_only=True
  )

  # Use decimal or scientific notation for numeric param values.
  # Decimal works best only if search space is standardized.
  use_scientific: bool = attrs.field(default=False, kw_only=True)

  def to_str(self, t: vz.TrialSuggestion, /) -> str:
    param_dict = t.parameters.as_dict()

    new_param_dict = dict()
    for pc in self.search_space.parameters:
      value = param_dict[pc.name]
      if isinstance(value, (float, int)):
        float_format = '.2e' if self.use_scientific else '.2f'
        new_param_dict[pc.name] = format(value, float_format)
      else:
        new_param_dict[pc.name] = value

    metadata_str = vs_lib.MetadataSerializer().to_str(t.metadata)
    return self.primitive_serializer.to_str(
        {'params': new_param_dict, 'metadata': metadata_str}
    )


SearchSpaceSerializer = vs_lib.SearchSpaceSerializer
