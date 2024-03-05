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

"""Wrappers for OptFormer algorithms into other APIs."""

import attrs
from optformer.vizier.algorithms import base
from vizier import pythia


@attrs.define
class OptFormerPolicy(pythia.Policy):
  """OptFormer Algorithm wrapped as a Vizier Pythia Policy.

  Since Transformers are stateless and ideally the OptFormer model should handle
  service trial cases (PENDING / STOPPING, etc.) on its own, having the
  OptFormer as a `pythia.Policy` provides the most flexibility compared to a
  `vza.Designer`.
  """

  algorithm: base.Algorithm = attrs.field(init=True)
  _policy_supporter: pythia.PolicySupporter = attrs.field(init=True)

  def suggest(self, request: pythia.SuggestRequest) -> pythia.SuggestDecision:
    history = self._policy_supporter.GetTrials()
    suggestions = self.algorithm.stateless_suggest(request.count, history)
    return pythia.SuggestDecision(suggestions)

  def early_stop(
      self, request: pythia.EarlyStopRequest
  ) -> pythia.EarlyStopDecisions:
    raise NotImplementedError
