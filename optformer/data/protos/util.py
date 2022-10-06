# Copyright 2022 Google LLC.
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

"""Methods to convert protos to PyVizier objects."""
from optformer.data.protos import study_pb2
from vizier.service import pyvizier as oss_vz


class ProblemAndTrialsConverter:
  """A wrapper for study_pb2.ProblemAndTrials."""

  @classmethod
  def from_proto(cls,
                 proto: study_pb2.ProblemAndTrials) -> oss_vz.ProblemAndTrials:
    """Converts proto to Python object."""
    problem = oss_vz.ProblemStatementConverter.from_proto(proto.problem)
    trials = oss_vz.TrialConverter.from_protos(proto.trials)
    return oss_vz.ProblemAndTrials(problem, trials)

  @classmethod
  def to_proto(
      cls, problem_and_trials: oss_vz.ProblemAndTrials
  ) -> study_pb2.ProblemAndTrials:
    """Converts Python object to proto."""
    problem_proto = oss_vz.ProblemStatementConverter.to_proto(
        problem_and_trials.problem)
    trial_protos = oss_vz.TrialConverter.to_protos(problem_and_trials.trials)
    return study_pb2.ProblemAndTrials(
        problem=problem_proto, trials=trial_protos)
