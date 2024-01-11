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

"""Serializers related to vz.Trial."""

import json

from optformer.common import serialization as s_lib
from vizier import pyvizier as vz


class JSONSuggestionSerializer(
    s_lib.Serializer[vz.ParameterDict], s_lib.Deserializer[vz.ParameterDict]
):

  def to_str(self, parameters: vz.ParameterDict, /) -> str:
    return json.dumps(parameters.as_dict())

  def from_str(self, s: str) -> vz.ParameterDict:
    return vz.ParameterDict(json.loads(s))
