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

"""Specific featurizers."""

from optformer.common import serialization as s_lib
from optformer.original import serializers as os_lib
from optformer.vizier.data import featurizers
from vizier import pyvizier as vz


# TODO: Replace with proper when moved.
class _DummyInputsSerializer(s_lib.Serializer[vz.ProblemAndTrials]):

  def to_str(self, obj: vz.ProblemAndTrials, /) -> str:
    return ""


def get_eval_featurizer():
  return featurizers.VizierStudyFeaturizer(
      _DummyInputsSerializer,
      lambda: os_lib.QuantizedTrialsSerializer((0.2, 0.8)),
  )
