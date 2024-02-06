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

"""Custom-token based serializers for Vizier objects."""

import math
import re
from typing import Literal

import attrs
import gin
from optformer.common import serialization as s_lib
from optformer.common.serialization import numeric
from optformer.validation import runtime
import ordered_set
from vizier import pyvizier as vz


@gin.configurable
@attrs.define(auto_attribs=False, frozen=True, kw_only=True)
class TrialTokenSerializer(s_lib.Serializer[vz.Trial]):
  """Converts Trial (Suggestion X and Measurement Y) w/ XY or YX ordering.

  XY ordering is appropriate for cloning algorithms (predict X_{t} given
  history_{t-1}) and regressing objectives (predict Y_{t} given history_{t-1}).

  YX ordering is appropriate for sampling level-sets (predict X_{t} given
  Y_{t}).
  """

  suggestion_serializer: s_lib.Serializer[vz.TrialSuggestion] = attrs.field()
  measurement_serializer: s_lib.Serializer[vz.Trial] = attrs.field()
  str_serializer: s_lib.TokenSerializer[str] = attrs.field(
      factory=s_lib.StringTokenSerializer
  )

  order: Literal['xy', 'yx'] = attrs.field(default='xy')  # pytype:disable=annotation-type-mismatch

  # ---------------------------------------------------------------------------
  # Special token string values below.
  # ---------------------------------------------------------------------------

  XY_SEPARATOR: str = '*'

  def to_str(self, trial: vz.Trial, /) -> str:
    """Returns Trial serialization.

    Ex: For quantized case, if X = [p1, p2, p3] and Y = [m1, m2], then output is
    either `X + [*] + Y` or `Y + [*] + X` depending on order.

    Args:
      trial: Trial with any status and final_measurement.

    Returns:
      Serialized trial.
    """
    suggest_str = self.suggestion_serializer.to_str(trial)
    measurement_str = self.measurement_serializer.to_str(trial)

    xy_sep_tok = self.str_serializer.to_str(self.XY_SEPARATOR)

    if self.order == 'xy':
      return suggest_str + xy_sep_tok + measurement_str
    elif self.order == 'yx':
      return measurement_str + xy_sep_tok + suggest_str
    else:
      raise ValueError(f'Invalid order: {self.order}')


@attrs.define(auto_attribs=False)
class MeasurementTokenSerializer(
    s_lib.Serializer[vz.Trial], s_lib.Deserializer[vz.Measurement]
):
  """Converts a measurement into custom tokens."""

  metrics_config: vz.MetricsConfig = attrs.field(init=True)
  float_serializer: s_lib.CartesianProductTokenSerializer[float] = attrs.field(
      init=True,
      kw_only=True,
      factory=numeric.DigitByDigitFloatTokenSerializer,
  )

  # Created after initialization.
  status_serializer: s_lib.RepeatedUnitTokenSerializer[str] = attrs.field(
      init=False
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

  def __attrs_post_init__(self):
    self.status_serializer = s_lib.RepeatedUnitTokenSerializer(
        s_lib.StringTokenSerializer(),
        self.float_serializer.num_tokens_per_obj,
    )

  def all_tokens_used(self) -> ordered_set.OrderedSet[str]:
    statuses = [self.PENDING, self.MISSING, self.INFEASIBLE]
    status_tokens = [self.status_serializer.to_str(st) for st in statuses]
    return self.float_serializer.all_tokens_used().union(status_tokens)

  def to_str(self, trial: vz.Trial, /) -> str:
    """Converts a trial's measurement to str using float tokenization.

    We need `vz.Trial` instead of `vz.Measurement` to obtain enough
    distinguishing information about ACTIVE/PENDING vs INFEASIBILE.

    Applies a floating point serialization to each measurement. Every final
    measurement metric would be converted to "f" number of tokens
    (f = float_serializer.num_tokens_per_float).

    Args:
      trial: Any trial.

    Returns:
      Metrics tokens. If our config has M metrics, we will **always** output
      M * f tokens, with every f tokens corresponding to a single float or the
      same special status token (e.g. <PENDING>.)
    """
    num_metrics = len(self.metrics_config)

    infeas_toks = self.status_serializer.to_str(self.INFEASIBLE)
    missing_toks = self.status_serializer.to_str(self.MISSING)
    if trial.infeasible:
      miss_or_infeas = infeas_toks
    else:
      miss_or_infeas = missing_toks

    if trial.status == vz.TrialStatus.ACTIVE:
      return self.status_serializer.to_str(self.PENDING) * num_metrics

    # Not pending but still no measurements: default to missing or infeasible.
    if trial.final_measurement is None:
      return miss_or_infeas * num_metrics

    # Final measurement exists. Now look at contents.
    metrics_vals = []
    measurement = trial.final_measurement
    for metric_config in self.metrics_config:
      m_name = metric_config.name
      if m_name not in measurement.metrics:
        metrics_vals.append(miss_or_infeas)
      else:
        objective = measurement.metrics[m_name].value
        if math.isnan(objective):
          metrics_vals.append(infeas_toks)
        else:
          metrics_vals.append(self.float_serializer.to_str(objective))
    return ''.join(metrics_vals)

  def from_str(self, s: str, /) -> vz.Measurement:
    """Returns measurement given a serialized string according to this tokenization.

    NOTE: Due to rounding-off errors, deserialization is not bijective with
    serialization.

    Args:
      s: String of form `<m^1_1>...<m^1_f>...<m^M_1>...<m^M_f>` where the
        superscript represents one of M metrics and the subscript represents one
        of the f tokens per metric. Each token should be either a
        self.float_serializer recognizable token or a special status token.

    Returns:
      vz.Measurement with each metric as an actual float, NaN, or simply
        nonexistent.
    """
    left_d, right_d = s_lib.TokenSerializer.DELIMITERS
    pattern = f'{left_d}[^{left_d}{right_d}]+{right_d}'
    n_tokens_per_metric = self.float_serializer.num_tokens_per_obj
    token_list = re.findall(pattern, s)
    runtime.assert_length(
        token_list, len(self.metrics_config) * n_tokens_per_metric
    )

    measurement = vz.Measurement()
    for i, metric_config in enumerate(self.metrics_config):
      start, end = i * n_tokens_per_metric, (i + 1) * n_tokens_per_metric
      metric_tokens = ''.join(token_list[start:end])
      if metric_tokens == self.status_serializer.to_str(self.INFEASIBLE):
        measurement.metrics[metric_config.name] = math.nan
      elif metric_tokens == self.status_serializer.to_str(self.MISSING):
        continue
      else:
        # Special tokens (e.g. PENDING) will error out from float serializer.
        try:
          f = self.float_serializer.from_str(metric_tokens)
          measurement.metrics[metric_config.name] = f
        except Exception as e:
          raise ValueError(f'Cannot float-deserialize {metric_tokens}') from e

    return measurement
