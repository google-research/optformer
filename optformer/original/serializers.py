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

"""Decoder-specific serializations using quantization."""

import functools
import itertools
import math
from typing import Dict, Optional, Sequence, Tuple

import attrs
import numpy as np
from optformer.common import serialization as s_lib
from optformer.original import numeric
from optformer.validation import runtime
from optformer.vizier import serialization as vs_lib
from vizier import pyvizier as vz
from vizier._src.pyvizier.shared import parameter_iterators as pi
from vizier.pyvizier import converters


_vizier_converter_factory = functools.partial(
    converters.DefaultModelInputConverter,
    scale=True,
    float_dtype=np.float64,
    max_discrete_indices=0,
)


@attrs.define(auto_attribs=False, frozen=True)
class QuantizedSuggestionSerializer(
    s_lib.Serializer[vz.TrialSuggestion], s_lib.Deserializer[vz.TrialSuggestion]
):
  """Serializes a TrialSuggestion via quantization.

  Important corner cases for conditional search space:
    1. If an active node (leaf or not) is not assigned a value in
      TrialSuggestion, then it is marked MISSING.
    2. If a non-active leaf node is assigned a value in TrialSuggestion, then
      it is still serialized and deserialized.
    3. If a non-active leaf node is missing a value in TrialSuggestion, then
      it is marked INACTIVE_CHILD.

  Understanding this behavior _does_ require mental gymnastics.
  """

  search_space: vz.SearchSpace = attrs.field(init=True)

  quantizer: numeric.NormalizedQuantizer = attrs.field(
      factory=numeric.NormalizedQuantizer,
      kw_only=True,
  )
  token_serializer: s_lib.UnitSequenceTokenSerializer = attrs.field(
      factory=s_lib.UnitSequenceTokenSerializer,
      kw_only=True,
  )

  # ---------------------------------------------------------------------------
  # Special token string values for parameter statuses below
  # ---------------------------------------------------------------------------

  # Parameter value is missing, for any reason (intentional or unintentional).
  MISSING: str = 'MISSING_PARAM'

  # Parameter is a child of a parent whose value did not activate said child.
  # TODO: Use for conditional cases for more precision over
  # missing params.
  INACTIVE_CHILD: str = 'INACTIVE_CHILD'

  @functools.cached_property
  def _merged_parameter_configs(self) -> Dict[str, vz.ParameterConfig]:
    # Traverse through the conditional tree and collect all ParameterConfig
    # objects.
    all_parameter_configs = itertools.chain.from_iterable([
        top_level_config.traverse()
        for top_level_config in self.search_space.parameters
    ])

    # Merge the parameter configs of the same name.
    # This is only for range.
    merged_parameter_configs: Dict[str, vz.ParameterConfig] = dict()
    for parameter_config in all_parameter_configs:
      name = parameter_config.name  # Alias
      if name not in merged_parameter_configs:
        merged_parameter_configs[name] = parameter_config
      else:
        # Merge with the existing config.
        merged_parameter_configs[name] = vz.ParameterConfig.merge(
            merged_parameter_configs[name], parameter_config
        )
    return merged_parameter_configs

  def _quantize_one_parameter(
      self, config: vz.ParameterConfig, suggestion: vz.TrialSuggestion
  ) -> int:
    value = suggestion.parameters[config.name].value

    if config.type == vz.ParameterType.CATEGORICAL:
      return config.feasible_values.index(value)

    elif config.type == vz.ParameterType.DISCRETE:
      # Convert to np.float32 to avoid errors due to small differences.
      feasible_values = [np.float32(v) for v in config.feasible_values]
      value = np.float32(value)
      return feasible_values.index(value)

    elif config.type in [vz.ParameterType.DOUBLE, vz.ParameterType.INTEGER]:
      # Normalize to [0,1] and then quantize.
      converter = _vizier_converter_factory(config)
      value = float(converter.convert([suggestion])[0, 0])
      value = int(self.quantizer.map(value))
      return value
    raise ValueError('Parameter type {config.type} is not supported.')

  def to_str(self, suggestion: vz.TrialSuggestion, /) -> str:
    """Converts each suggestion parameter to an int.

    For CATEGORICAL/DISCRETE, this corresponds to the index with respect
    to the feasible values.

    For DOUBLE/INTEGER this corresponds to the normalized + quantized int value.

    Args:
      suggestion: TrialSuggestion. Assumed to match with the given problem
        statement.

    Returns:
      String-ified array of ints.
    """
    quantizations = []

    builder = pi.SequentialParameterBuilder(self.search_space)
    missing_parameters: set[str] = set()

    for pc in builder:
      if pc.name not in suggestion.parameters:
        missing_parameters.add(pc.name)
        builder.skip()
      else:
        builder.choose_value(suggestion.parameters[pc.name].value)

    for p_config in self._merged_parameter_configs.values():
      name = p_config.name
      if name not in suggestion.parameters:
        if name in missing_parameters:
          quantizations.append(self.MISSING)
        else:
          quantizations.append(self.INACTIVE_CHILD)
      else:
        quantizations.append(self._quantize_one_parameter(p_config, suggestion))
    return self.token_serializer.to_str(quantizations)

  def from_str(self, s: str) -> vz.TrialSuggestion:
    """Deserializes P quantized parameters (as string) to P `vz.Parameter`s.

    Args:
      s: String of form `<q_1><q_2>...<q_P>` where each `q_i` is a quantized
        int. Ordering determined by `self.search_space`.

    Returns:
      Corresponding suggestion.
    """
    params = vz.ParameterDict()

    quantized: Sequence[str | int] = self.token_serializer.from_str(s)
    runtime.assert_length(quantized, len(self._merged_parameter_configs))

    for q, pc in zip(quantized, self._merged_parameter_configs.values()):
      if q in (self.MISSING, self.INACTIVE_CHILD):
        pass
      elif pc.type in [vz.ParameterType.DOUBLE, vz.ParameterType.INTEGER]:
        vz_converter = _vizier_converter_factory(pc)
        q = self.quantizer.unmap(q)
        params[pc.name] = vz_converter.to_parameter_values(np.array(q))[0]
      elif pc.type in [vz.ParameterType.CATEGORICAL, vz.ParameterType.DISCRETE]:
        params[pc.name] = vz.ParameterValue(pc.feasible_values[int(q)])
      else:
        raise ValueError(f'Parameter type {pc.type} is not supported.')

    return vz.TrialSuggestion(parameters=params)


@attrs.define(auto_attribs=False, frozen=True, kw_only=True)
class QuantizedMeasurementSerializer(
    s_lib.Serializer[vz.Trial], s_lib.Deserializer[vz.Measurement]
):
  """Converts a Trial's measurement via quantization."""

  metrics_config: vz.MetricsConfig = attrs.field(init=True)
  metrics_scalers: Dict[str, numeric.LinearIntervalScaler] = attrs.field(
      init=True
  )
  quantizer: numeric.NormalizedQuantizer = attrs.field(
      factory=numeric.NormalizedQuantizer
  )
  token_serializer: s_lib.UnitSequenceTokenSerializer = attrs.field(
      init=True, factory=s_lib.UnitSequenceTokenSerializer
  )
  # ---------------------------------------------------------------------------
  # Special token string values below for notifying trial statuses to model.
  # ---------------------------------------------------------------------------

  # User is currently evaluating the pending trial. Ideally, the suggestion
  # should not be duplicated. Useful signal for batch suggestion algorithms.
  PENDING: str = 'PENDING'

  # User failed to provide the metric for whatever reason. We can resend the
  # same suggestion again to confirm the missing metric.
  MISSING: str = 'MISSING_METRIC'

  # The objective function cannot be evaluated at the suggestion. Ideally this
  # area of the search space should therefore be avoided.
  INFEASIBLE: str = 'INFEASIBLE'

  def to_str(self, trial: vz.Trial, /) -> str:
    """Converts a trial's measurement to str.

    We need `vz.Trial` instead of `vz.Measurement` to obtain enough
    distinguishing information about ACTIVE/PENDING vs INFEASIBILE.

    We take the unbiased objective, and will linearly map it to an interval
    specified by the scaler, and then quantize to collect an int.

    Args:
      trial: Single-objective trial. We expect this matches with the provided
        metrics_config.

    Returns:
      Metrics tokens. If our config has M metrics, we will **always** output
      M tokens, each of which is either a quantized value (ex: <42>) or
      special measurement status (ex: <pending>).
    """
    num_metrics = len(self.metrics_config)

    if trial.status == vz.TrialStatus.ACTIVE:
      all_pending = num_metrics * [self.PENDING]
      return self.token_serializer.to_str(all_pending)

    missing_or_infeasible_str = (
        self.INFEASIBLE if trial.infeasible else self.MISSING
    )

    # Not pending but still no measurements: default to missing or infeasible.
    if trial.final_measurement is None:
      all_miss_or_infeas = num_metrics * [missing_or_infeasible_str]
      return self.token_serializer.to_str(all_miss_or_infeas)

    # Final measurement exists. Now look at contents.
    metrics_vals = []
    measurement = trial.final_measurement
    for metric_config in self.metrics_config:
      m_name = metric_config.name
      if m_name not in measurement.metrics:
        metrics_vals.append(missing_or_infeasible_str)
      else:
        objective = measurement.metrics[m_name].value
        if math.isnan(objective):
          metrics_vals.append(self.INFEASIBLE)
        else:
          normalized_objective = self.metrics_scalers[m_name].map(objective)
          quantized_objective = self.quantizer.map(normalized_objective)
          metrics_vals.append(quantized_objective)
    return self.token_serializer.to_str(metrics_vals)

  def from_str(self, s: str, /) -> vz.Measurement:
    """Returns measurement.

    NOTE: It's impossible to guarantee bijectivity on deserialization.
    Also, the caller should be responsible for dealing w/ trial statuses (e.g.
    PENDING) instead of this class.

    Args:
      s: String of form `<q_1><q_2>...<q_M>` where <q_i> is either a quantized
        int or special status token. `self.metrics_config` determines ordering.

    Returns:
      vz.Measurement with each metric as an actual float, NaN, or simply
        nonexistent.
    """
    metric_vals = self.token_serializer.from_str(s)
    runtime.assert_length(metric_vals, len(self.metrics_config))

    measurement = vz.Measurement()
    for metric_config, metric_val in zip(self.metrics_config, metric_vals):
      if isinstance(metric_val, int):
        normalized_objective = self.quantizer.unmap(metric_val)
        raw_objective = self.metrics_scalers[metric_config.name].unmap(
            normalized_objective
        )
        measurement.metrics[metric_config.name] = raw_objective
      elif metric_val == self.INFEASIBLE:
        measurement.metrics[metric_config.name] = math.nan
      elif metric_val == self.MISSING:
        pass
      else:
        raise ValueError(f'Cannot handle special status token {metric_val}.')

    return measurement


# TODO: This is actually implicitly calling factory methods.
# Consider de-constructing down into a `QuantizedMeasurementSerializerFactory`.


@attrs.define(auto_attribs=False, frozen=True)
class QuantizedTrialsSerializer(s_lib.Serializer[vz.ProblemAndTrials]):
  """Serializes trajectories of trials.

  Parameters and metrics that exist in Trial but are not configured in the
  problem statement are silently dropped.

  Attributes:
    objective_target_interval: Metrics are linearly scaled to this range.
    str_tokenizer:
    serialize_measurements: If true (default), serialization includes the
      measurements.
    TRIAL_SEPARATOR: String that marks the end of a trial.
  """

  objective_target_interval: Tuple[float, float] = attrs.field(
      default=(0.0, 1.0)
  )

  str_tokenizer: s_lib.TokenSerializer[str] = attrs.field(
      factory=s_lib.StringTokenSerializer, kw_only=True
  )

  serialize_measurements: bool = attrs.field(default=True, kw_only=True)

  # ---------------------------------------------------------------------------
  # Special token string values below.
  # ---------------------------------------------------------------------------

  TRIAL_SEPARATOR: str = '|'

  def _create_scaler_for_metric(
      self, study: vz.ProblemAndTrials, metric_name: str
  ) -> numeric.LinearIntervalScaler:
    objectives = []
    for trial in study.trials:
      if trial.is_completed and not trial.infeasible:
        if trial.final_measurement:  # Required to safisfy pytype.
          objectives.append(trial.final_measurement.metrics[metric_name].value)

    # Compute objective bounds.
    if objectives:
      objective_source_interval = (min(objectives), max(objectives))
    else:
      # NOTE: Measurement serializer won't get used anyways. This is a dummy.
      objective_source_interval = (0.0, 1.0)

    return numeric.LinearIntervalScaler(
        source_interval=objective_source_interval,
        target_interval=self.objective_target_interval,
    )

  def to_str(self, study: vz.ProblemAndTrials, /) -> str:
    """Serializes a trajectory.

    Will output the following format: [|, p1, p2, p3, *, m1, m2, |, ...]

    when a trial has parameters [p1, p2, p3] and metrics [m1, m2].
    `...` corresponds to the next trial's serialization.

    Args:
      study: Trajectory (problem statement and list of trials).

    Returns:
      Serialized trajectory. See above on examples outputs.
    """
    suggestion_serializer = QuantizedSuggestionSerializer(
        search_space=study.problem.search_space
    )

    if self.serialize_measurements:
      measurement_serializer = QuantizedMeasurementSerializer(
          metrics_config=study.problem.metric_information,
          metrics_scalers={
              metric.name: self._create_scaler_for_metric(study, metric.name)
              for metric in study.problem.metric_information
          },
      )
      trial_serializer = vs_lib.TrialTokenSerializer(
          suggestion_serializer=suggestion_serializer,
          measurement_serializer=measurement_serializer,
      )
    else:
      trial_serializer = suggestion_serializer
    trial_strs = [trial_serializer.to_str(t) for t in study.trials]

    # NOTE: To avoid empty string outputs (which lead to weird tokenizations),
    # we always add the separator to the beginning.
    trial_sep_token = self.str_tokenizer.to_str(self.TRIAL_SEPARATOR)
    return trial_sep_token.join([''] + trial_strs + [''])


@attrs.define
class RandomizedQuantizedTrialsSerializerFactory(
    s_lib.SerializerFactory[vz.ProblemAndTrials]
):
  """Randomizes target interval for data augmentation during training."""

  interval_sampler: numeric.UniformIntervalSampler = attrs.field(
      factory=lambda: numeric.UniformIntervalSampler(length_bounds=(0.0, 1.0))
  )

  def __call__(
      self, *, seed: Optional[int] = None
  ) -> s_lib.Serializer[vz.ProblemAndTrials]:
    interval = self.interval_sampler(seed=seed)
    return QuantizedTrialsSerializer(objective_target_interval=interval)


@attrs.define
class ProblemStudySerializer(s_lib.Serializer[vz.ProblemAndTrials]):
  problem_serializer: s_lib.Serializer[vz.ProblemStatement] = attrs.field(
      factory=vs_lib.ProblemSerializer
  )

  def to_str(self, study: vz.ProblemAndTrials, /) -> str:
    return self.problem_serializer.to_str(study.problem)
