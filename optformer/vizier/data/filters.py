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

"""Vizier-specific filters."""

import attrs
import numpy as np
from optformer.common.data import filters
from vizier import pyvizier as vz
from vizier._src.pyglove import constants
from vizier.pyvizier import converters


def _validate_parameters(
    parameters: dict[str, vz.ParameterValueTypes], space: vz.SearchSpace
) -> bool:
  """Call with parameters=trial.parameters.as_dict()."""
  builder = vz.SequentialParameterBuilder(space, traverse_order='bfs')
  try:
    for pc in builder:
      builder.choose_value(parameters.pop(pc.name))
  except KeyError:
    return False

  if parameters:
    return False
  return True


@attrs.define
class ConstantMetricFilter(filters.Filter[vz.ProblemAndTrials]):
  """Checks if all trials share exactly the same y-values (normally a bug)."""

  # If true, pretends NaN trials don't exist.
  ignore_infeasible: bool = attrs.field(default=False)

  def __call__(self, study: vz.ProblemAndTrials) -> bool:
    convtr = converters.TrialToArrayConverter.from_study_config(study.problem)
    ys = convtr.to_labels(study.trials)
    if self.ignore_infeasible:
      ys[np.isnan(ys)] = 0.0

    if np.isclose(ys - ys[0, :], 0.0).all():
      raise ValueError('All trial metrics are equal.')
    return True


class CompletedStudyFilter(filters.Filter[vz.ProblemAndTrials]):
  """Checks if the study only has completed trials with correct params."""

  def __call__(self, study: vz.ProblemAndTrials) -> bool:
    for trial in study.trials:
      if trial.status != vz.TrialStatus.COMPLETED:
        raise ValueError(f'Found a non-completed trial: {trial}')

      param_dict = trial.parameters.as_dict()
      if not _validate_parameters(param_dict, study.problem.search_space):
        raise ValueError(f'Trial {trial} params not contained in search space.')
    return True


class RemovePygloveStudies(filters.Filter[vz.ProblemAndTrials]):
  """Remove pyglove studies.

  Pyglove studies typically store the true search spec inside metadata, and
  fill the Vizier search space with a placeholder.
  """

  def __call__(self, study: vz.ProblemAndTrials) -> bool:
    # Study has a single objective and has no safety constraints.
    sc = study.problem

    constants_ns = sc.metadata.ns(constants.METADATA_NAMESPACE)
    if constants.STUDY_METADATA_KEY_DNA_SPEC in constants_ns:
      raise ValueError('Study has pyglove metadata.')
    return True


class SingleMetricOnly(filters.Filter[vz.ProblemAndTrials]):
  """Checks if the study has a single objective."""

  def __call__(self, study: vz.ProblemAndTrials) -> bool:
    # Study has a single objective and has no safety constraints.
    sc = study.problem
    if not sc.is_single_objective:
      raise ValueError(f'Is not single objective: {sc.metric_information}')
    if len(sc.metric_information) != 1:
      raise ValueError(f'Has {len(sc.metric_information)} metrics.')
    return True


@attrs.define
class MaxNumMetricOnly(filters.Filter[vz.ProblemAndTrials]):
  """Checks if the study has at most `max_num_metrics` objective."""

  _max_num_metrics: int = attrs.field(default=1)

  def __call__(self, study: vz.ProblemAndTrials) -> bool:
    metric_info = study.problem.metric_information
    if len(metric_info) > 1:
      raise ValueError(
          f'Has {len(metric_info)} > {self._max_num_metrics} metrics.'
      )
    return True


class FlatSpaceOnly(filters.Filter[vz.ProblemAndTrials]):
  """Checks if the study is flat space."""

  def __call__(self, study: vz.ProblemAndTrials) -> bool:
    if study.problem.search_space.is_conditional:
      raise ValueError('Search space is conditional.')
    return True


@attrs.define
class TrialCountFilter(filters.Filter[vz.ProblemAndTrials]):
  """Checks if the Study has desired number of trials."""

  _min_trials: int = attrs.field(default=1)
  _max_trials: int = attrs.field(default=int(1e15))

  def __attrs_post_init__(self):
    if self._min_trials > self._max_trials or self._max_trials <= 0:
      raise ValueError(f'Invalid filter: {self}')

  def __call__(self, study: vz.ProblemAndTrials) -> bool:
    # The number of trials is more than a minimum threshold.
    num_trials = len(study.trials)
    if num_trials < self._min_trials or (num_trials > self._max_trials):
      raise ValueError(f'Has {num_trials} trials. Filter: {self}')
    return True
