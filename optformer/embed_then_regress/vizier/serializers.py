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

from optformer.common import serialization as s_lib
from vizier import pyvizier as vz

SuggestionSerializer = s_lib.Serializer[vz.TrialSuggestion]
ProblemSerializer = s_lib.Serializer[vz.ProblemAndTrials]


class XSerializer(SuggestionSerializer):
  """Basic serializer for Vizier parameters."""

  def to_str(self, t: vz.TrialSuggestion, /) -> str:
    out = dict()
    for key, value in t.parameters.as_dict().items():
      if isinstance(value, (float, int)):
        out[key] = format(value, '.2e')  # Scientific notation.
      else:
        out[key] = value
    return str(out)
