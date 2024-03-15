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

from optformer.original import serializers as os_lib
from optformer.vizier.data import featurizers


def get_train_featurizer() -> featurizers.VizierStudyFeaturizer:
  return featurizers.VizierStudyFeaturizer(
      os_lib.ProblemStudySerializer,
      os_lib.RandomizedQuantizedTrialsSerializerFactory(),
  )


def get_eval_featurizer() -> featurizers.VizierStudyFeaturizer:
  return featurizers.VizierStudyFeaturizer(
      os_lib.ProblemStudySerializer,
      lambda: os_lib.QuantizedTrialsSerializer((0.2, 0.8)),
  )
