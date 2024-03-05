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

"""Base classes for optimization primitives (ex: algorithms)."""

import abc
from typing import Sequence
from vizier import pyvizier as vz


class Algorithm(abc.ABC):

  @abc.abstractmethod
  def stateless_suggest(
      self, count: int, history: Sequence[vz.Trial]
  ) -> Sequence[vz.TrialSuggestion]:
    """Given an optimization history, output a count of suggestions.

    In-line with Transformers being stateless, we do not keep track of
    historical trajectories in this class.

    Args:
      count: Makes best effort to generate this many suggestions.
      history: Historical optimization trajectory of trials.

    Returns:
      New suggestions.
    """
