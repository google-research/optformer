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

"""Vizier-specific augmenters."""

import abc
import hashlib
import json
import random
from typing import MutableSequence, Optional, Sequence, TypeVar

import attrs
import numpy as np
from optformer.common.data import augmenters
from vizier import pyvizier as vz
from vizier.pyvizier import converters
from vizier.pyvizier.multimetric import xla_pareto
from vizier.utils import json_utils


_T = TypeVar('_T')


class VizierAugmenter(augmenters.Augmenter[_T]):

  @abc.abstractmethod
  def augment_study(self, study: vz.ProblemAndTrials, /) -> vz.ProblemAndTrials:
    """Unified API to call over `vz.ProblemAndTrials`.

    Args:
      study: A Vizier study.

    Returns:
      Augmented study. Can also be a reference to original study if `augment`
      modifies the sub-object in-place.
    """


class VizierIdempotentAugmenter(VizierAugmenter[_T]):
  """VizierAugmenters that are idempotent.

  Other components may use this typing for validation purposes.

  NOTE: An alternative design is to add a property `is_idempotent` to
  `VizierAugmenter` API so that different configurations of the same class
  can behave differently. We chose to require a class to be always idempotent
  or not, for simplicity.
  """


@attrs.define
class SearchSpacePermuter(VizierAugmenter[vz.SearchSpace]):
  """Permutes the search space's parameters."""

  seed: Optional[int] = attrs.field(init=True, kw_only=True, default=None)

  def augment(self, search_space: vz.SearchSpace, /) -> vz.SearchSpace:
    """Logic below reduces expensive object-copying as much as possible."""
    rng = random.Random(self.seed)

    # Pop out all parameter configs, shuffle, then put back in.
    parameter_names = list(search_space.parameter_names)
    p_configs = [search_space.pop(name) for name in parameter_names]
    rng.shuffle(p_configs)
    for p_config in p_configs:
      search_space.add(p_config)

    return search_space

  def augment_study(self, obj: vz.ProblemAndTrials) -> vz.ProblemAndTrials:
    self.augment(obj.problem.search_space)  # In-place.
    return obj


@attrs.define
class MetricsConfigPermuter(VizierAugmenter[vz.MetricsConfig]):
  """Permutes the metrics in a config."""

  seed: Optional[int] = attrs.field(init=True, kw_only=True, default=None)

  def augment(self, metrics_config: vz.MetricsConfig, /) -> vz.MetricsConfig:
    """Logic below reduces expensive object-copying as much as possible."""
    metrics = list(metrics_config)

    rng = random.Random(self.seed)
    rng.shuffle(metrics)
    return vz.MetricsConfig(metrics)

  def augment_study(self, study: vz.ProblemAndTrials, /) -> vz.ProblemAndTrials:
    new_mc = self.augment(study.problem.metric_information)
    study.problem.metric_information = new_mc
    return study


@attrs.define
class TrialsPermuter(VizierAugmenter[MutableSequence[vz.Trial]]):
  """Permutes a list of trials."""

  seed: Optional[int] = attrs.field(init=True, kw_only=True, default=None)

  def augment(
      self, trials: MutableSequence[vz.Trial], /
  ) -> MutableSequence[vz.Trial]:
    """Logic below reduces expensive object-copying as much as possible."""
    rng = random.Random(self.seed)
    rng.shuffle(trials)
    return trials

  def augment_study(self, study: vz.ProblemAndTrials, /) -> vz.ProblemAndTrials:
    self.augment(study.trials)
    return study


def _pareto_argsort(
    metrics: vz.MetricsConfig,
    trials: Sequence[vz.Trial],
    *,
    seed: Optional[int] = None,
    reverse: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
  """Argsort the trials by pareto rank. Does not support safe metrics yet.

  Args:
    metrics:
    trials:
    seed: Seed for breaking ties. (Does not yet work for single objectives)
    reverse: If True, sorted list goes from worst (highest pareto rank) to best.

  Returns:
    Sorted indices for `trials` and the pareto rank of the corresponding trials.
    More precisely, if
      sorted_idx, ranks = _pareto_argsort(metrics, trials)
    then
      trials[sorted_idx[i]] has pareto rank ranks[i].
  """
  if not metrics.of_type(vz.MetricType.OBJECTIVE):
    raise ValueError('Requires at least one objective metric.')
  if metrics.of_type(vz.MetricType.SAFETY):
    raise ValueError('Cannot work with safe metrics.')

  problem = vz.ProblemStatement(metric_information=metrics)
  converter = converters.TrialToArrayConverter.from_study_config(
      problem, flip_sign_for_minimization_metrics=True, dtype=np.float32
  )
  labels = converter.to_labels(trials)
  if reverse:
    labels = -labels

  if len(metrics) == 1:
    # For single objectives, rank is simply the order.
    ranks = np.arange(len(trials))
    sorted_idx = np.argsort(-labels.squeeze())
    return sorted_idx, ranks
  else:
    rng = np.random.RandomState(seed)
    ranks = xla_pareto.pareto_rank(labels)
    # Add uniform noise to break ties.
    sorted_idx = np.argsort(ranks + rng.uniform(0, 1, ranks.shape))
    return sorted_idx, ranks[sorted_idx]


@attrs.define(kw_only=True)
class ParetoRankReverseSortAndSubsample(VizierAugmenter[vz.ProblemAndTrials]):
  """Sorts the trials by reverse pareto rank and subsample them.

  This augmenter subsamples `num_trials` trials such that their pareto ranks
  form a non-increasing sequence (i.e. ordered by worst to best).

  It also populates study's `metadata['N']` to be `num_trials`.

  A model trained with this augmenter can generate suggestions by following
  these steps:
    1. Given a study, sort the trials from worst to best
    2. Set the problem statement's `metadata['N']` to be the smallest
      value among `self.num_trials` that exceeds the current trial count.
    3. Prompt for suggestions.

  Attributes:
    num_trials:
    seed: If not None, this augmenter is idempotent.
  """

  num_trials: Sequence[int] = attrs.field(
      default=(1, 50, 100, 150, 200, 250, 300),
  )
  seed: Optional[int] = attrs.field(default=None)

  def augment(self, study: vz.ProblemAndTrials, /) -> vz.ProblemAndTrials:
    return self.augment_study(study)

  def augment_study(self, study: vz.ProblemAndTrials, /) -> vz.ProblemAndTrials:
    study = TrialsSorter(reverse=True).augment_study(study)
    sampler = TrialsSubsampler(num_trials=self.num_trials, seed=self.seed)
    return sampler.augment_study(study)


@attrs.define(kw_only=True)
class TrialsSubsampler(VizierAugmenter[vz.ProblemAndTrials]):
  """Subsample the Trials.

  Attributes:
    num_trials:
    seed: If not None, this augmenter is idempotent. TODO: Reject None.
  """

  num_trials: Sequence[int] = attrs.field()
  metadata_name: str = attrs.field(default='N')
  seed: Optional[int] = attrs.field(default=None)

  def augment(self, study: vz.ProblemAndTrials, /) -> vz.ProblemAndTrials:
    return self.augment_study(study)

  def augment_study(self, study: vz.ProblemAndTrials, /) -> vz.ProblemAndTrials:
    """Subsample the trials."""
    num_trials = [x for x in self.num_trials if x < len(study.trials)]
    if not num_trials:
      raise ValueError('Not enough trials to subsample.')

    n: int = np.random.RandomState(self.seed).choice(num_trials)
    if n < len(study.trials):
      # TODO: For multi-objective, we should take all rank 0 trials.
      indices = np.linspace(0, len(study.trials) - 1, n).astype(np.int_)
      study.trials[:] = np.asarray(study.trials)[indices].tolist()

    # TODO: Need to make sure this does not get wiped out.
    if self.metadata_name:
      if self.metadata_name in study.problem.metadata:
        raise ValueError(
            f'Duplicate metadata name as subsampling {self.metadata_name}'
        )
      study.problem.metadata[self.metadata_name] = str(n)
    return study


@attrs.define
class RemoveStudyMetadata(VizierIdempotentAugmenter[vz.ProblemStatement]):
  """Remove all study-level metadata from the ProblemStatement."""

  def augment(self, problem: vz.ProblemStatement, /) -> vz.ProblemStatement:
    problem.metadata.clear()
    return problem

  def augment_study(self, study: vz.ProblemAndTrials, /) -> vz.ProblemAndTrials:
    self.augment(study.problem)
    return study


def _has_missing_metrics(t: vz.Trial, metrics: vz.MetricsConfig) -> bool:
  for m in metrics:
    if m.name not in t.final_measurement_or_die.metrics:
      return True
  return False


def _has_nan(t: vz.Trial, metrics: vz.MetricsConfig) -> bool:
  metric_vals = [
      t.final_measurement_or_die.metrics[m.name].value
      for m in metrics
      if m.name in t.final_measurement_or_die.metrics
  ]
  return np.isnan(metric_vals).any()


@attrs.define(init=True, kw_only=True)
class IncompleteTrialRemover(VizierIdempotentAugmenter[vz.ProblemAndTrials]):
  """Removes incomplete trials."""

  remove_infeasible: bool = attrs.field(default=False)
  remove_missing_metrics: bool = attrs.field(default=True)
  remove_nan: bool = attrs.field(default=True)

  # NOTE: ProblemAndTrials `trials` field is frozen, so we still edit in-place.
  def augment(self, study: vz.ProblemAndTrials, /) -> vz.ProblemAndTrials:
    metrics = study.problem.metric_information
    trials = study.trials

    trials[:] = [t for t in trials if t.status == vz.TrialStatus.COMPLETED]
    if self.remove_infeasible:
      trials[:] = [t for t in trials if not t.infeasible]
    if self.remove_missing_metrics:
      trials[:] = [t for t in trials if not _has_missing_metrics(t, metrics)]
    if self.remove_nan:
      trials[:] = [t for t in trials if not _has_nan(t, metrics)]
    return study

  def augment_study(self, study: vz.ProblemAndTrials, /) -> vz.ProblemAndTrials:
    return self.augment(study)


@attrs.define
class TrialsSorter(VizierIdempotentAugmenter[vz.ProblemAndTrials]):
  """Sort a study's trials from best to worst (based on metric goal).

  If reverse is True, sorts from worst to best.
  """
  reverse: bool = attrs.field(default=False)

  def augment(self, study: vz.ProblemAndTrials, /) -> vz.ProblemAndTrials:
    sorted_idx, _ = _pareto_argsort(
        study.problem.metric_information, study.trials, reverse=self.reverse
    )
    study.trials[:] = np.asarray(study.trials)[sorted_idx]
    return study

  def augment_study(self, study: vz.ProblemAndTrials, /) -> vz.ProblemAndTrials:
    return self.augment(study)


def _flip_metric(metric_name: str, measurement: vz.Measurement) -> None:
  val = measurement.metrics[metric_name].value
  measurement.metrics[metric_name] = vz.Metric(value=-val)


class ConvertToMaximizationProblem(
    VizierIdempotentAugmenter[vz.ProblemAndTrials]
):
  """Make the study a maximization study.

  This augmenter changes all minimization metric goals to maximize and flip
  signs of such metric values.

  NOTE: This should only be used if absolute values of metrics do not matter
  (e.g. in quantized case).
  """

  def augment(self, study: vz.ProblemAndTrials, /) -> vz.ProblemAndTrials:
    for mc in study.problem.metric_information:
      if mc.goal.is_maximize:
        continue

      mc.flip_goal()
      for trial in study.trials:
        for measurement in trial.measurements:
          _flip_metric(mc.name, measurement)
        if trial.final_measurement:
          _flip_metric(mc.name, trial.final_measurement)

    return study

  def augment_study(self, study: vz.ProblemAndTrials, /) -> vz.ProblemAndTrials:
    return self.augment(study)


@attrs.define(init=True, kw_only=True)
class RandomMetricFlipper(VizierAugmenter[vz.ProblemAndTrials]):
  """Flips a random subset of metrics across all measurements.

  Useful during training to check regression bimodality.
  """

  seed: Optional[int] = attrs.field(default=None)

  def augment(self, study: vz.ProblemAndTrials, /) -> vz.ProblemAndTrials:
    rng = random.Random(self.seed)

    for trial in study.trials:
      for mc in study.problem.metric_information:
        if trial.final_measurement and rng.choice([True, False]):
          _flip_metric(mc.name, trial.final_measurement)

    return study

  def augment_study(self, study: vz.ProblemAndTrials, /) -> vz.ProblemAndTrials:
    return self.augment(study)


class HashProblemMetadata(VizierAugmenter):
  """Converts the problem metadata tree into a hash."""

  def augment(self, study: vz.ProblemAndTrials, /) -> vz.ProblemAndTrials:
    dump = json.dumps(study.problem.metadata, cls=json_utils.MetadataEncoder)
    hashed_metadata = hashlib.sha256(dump.encode('utf-8')).hexdigest()

    # There's no easy way to clear the metadata due to how it handles namespaces
    # versus keys.
    study.problem.metadata._stores.clear()  # pylint:disable=protected-access
    study.problem.metadata['H'] = hashed_metadata
    return study

  def augment_study(self, study: vz.ProblemAndTrials, /) -> vz.ProblemAndTrials:
    return self.augment(study)
