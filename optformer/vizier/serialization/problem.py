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

"""Converts Vizier study configs to strings."""

import dataclasses
from typing import Dict

import attrs
from optformer.common import serialization as s_lib
from optformer.vizier.serialization import metadata
from vizier import pyvizier as vz


@dataclasses.dataclass
class FieldMapping:
  """Keys used to shortcut StudyConfig field names."""

  name: str = 'N'
  algorithm_name: str = 'A'

  metadata: str = 'X'

  objective_name: str = 'O'
  goal_name: str = 'G'

  param_type: str = 'P'
  scale_type: str = 'S'

  discrete_feasible_values: str = 'F'
  categorical_feasible_values: str = 'C'
  feasible_length: str = 'L'
  bound_min: str = 'm'
  bound_max: str = 'M'

  subspace: str = 's'


# TODO: Find good way to make customizable + gin-configurable.
_ScaleTypeMapping = {
    None: 'None',
    vz.ScaleType.LINEAR: vz.ScaleType.LINEAR.name,
    vz.ScaleType.LOG: vz.ScaleType.LOG.name,
    vz.ScaleType.REVERSE_LOG: vz.ScaleType.REVERSE_LOG.name,
    vz.ScaleType.UNIFORM_DISCRETE: vz.ScaleType.UNIFORM_DISCRETE.name,
}

_ParameterTypeMapping = {
    vz.ParameterType.DOUBLE: vz.ParameterType.DOUBLE.name,
    vz.ParameterType.INTEGER: vz.ParameterType.INTEGER.name,
    vz.ParameterType.CATEGORICAL: vz.ParameterType.CATEGORICAL.name,
    vz.ParameterType.DISCRETE: vz.ParameterType.DISCRETE.name,
}


@attrs.define(auto_attribs=False, frozen=True, kw_only=True)
class SearchSpaceSerializer(s_lib.Serializer[vz.SearchSpace]):
  """Serializes Vizier search space to string."""

  primitive_serializer: s_lib.Serializer[s_lib.PrimitiveType] = attrs.field(
      factory=s_lib.PrimitiveSerializer
  )
  field_mapping: FieldMapping = attrs.field(factory=FieldMapping)

  include_param_name: bool = attrs.field(default=True)
  include_scale_type: bool = attrs.field(default=True)
  include_bounds_and_feasibles: bool = attrs.field(default=True)

  PARAM_CONFIG_SEPARATOR: str = '*'

  def _pc_to_str(self, pc: vz.ParameterConfig) -> str:
    pc_info = {}
    pc_info[self.field_mapping.param_type] = _ParameterTypeMapping[pc.type]

    # Optional information.
    if self.include_param_name:
      pc_info[self.field_mapping.name] = pc.name
    if vz.ParameterType.is_numeric(pc.type) and self.include_scale_type:
      pc_info[self.field_mapping.scale_type] = _ScaleTypeMapping[pc.scale_type]

    # Possible parameter infos.
    if pc.type in [vz.ParameterType.DOUBLE, vz.ParameterType.INTEGER]:
      if self.include_bounds_and_feasibles:
        min_val, max_val = pc.bounds
        pc_info[self.field_mapping.bound_min] = min_val
        pc_info[self.field_mapping.bound_max] = max_val

    elif pc.type == vz.ParameterType.CATEGORICAL:
      pc_info[self.field_mapping.feasible_length] = len(pc.feasible_values)
      if self.include_bounds_and_feasibles:
        pc_info[self.field_mapping.categorical_feasible_values] = (
            pc.feasible_values
        )
    elif pc.type == vz.ParameterType.DISCRETE:
      pc_info[self.field_mapping.feasible_length] = len(pc.feasible_values)
      if self.include_bounds_and_feasibles:
        # Sometimes `feasible_values` can be all ints. Force them to be floats.
        float_feasible_values = [float(v) for v in pc.feasible_values]
        pc_info[self.field_mapping.discrete_feasible_values] = (
            float_feasible_values
        )
    else:
      raise ValueError(f'{pc.type} not supported yet.')

    for value, subspace in pc.subspaces():
      pc_info[self.field_mapping.subspace] = {value: self.to_str(subspace)}
    return self.primitive_serializer.to_str(pc_info)

  def to_str(self, obj: vz.SearchSpace) -> str:
    """Converts search space.

    Loops over all parameter configs to collect bounds/feasible values
    to be collected into an info dict. Info dict will then be
    primitive-serialized.

    Args:
      obj: SearchSpace.

    Returns:
      String-ified list of parameter configs.
    """
    search_space = obj
    param_info_strs = []
    for pc in search_space.parameters:
      param_info_strs.append(self._pc_to_str(pc))

    return self.PARAM_CONFIG_SEPARATOR.join(param_info_strs)


@attrs.define(auto_attribs=False, frozen=True, kw_only=True)
class MetricsConfigSerializer(s_lib.Serializer[vz.MetricsConfig]):
  """Serializes Vizier metric configuration to string."""

  primitive_serializer: s_lib.Serializer[s_lib.PrimitiveType] = attrs.field(
      factory=s_lib.PrimitiveSerializer
  )
  field_mapping: FieldMapping = attrs.field(factory=FieldMapping)

  include_metric_name: bool = attrs.field(default=True)
  include_goal: bool = attrs.field(default=True)

  METRIC_INFO_SEPARATOR: str = '@'

  def _metric_info_to_str(self, metric_info: vz.MetricInformation) -> str:
    metric_dict = {}
    if self.include_metric_name:
      metric_dict[self.field_mapping.objective_name] = metric_info.name
    if self.include_goal:
      metric_dict[self.field_mapping.goal_name] = metric_info.goal.name

    return self.primitive_serializer.to_str(metric_dict)

  def to_str(self, obj: vz.MetricsConfig) -> str:
    metrics_config = obj

    mc_strs = []
    for mc in metrics_config:
      mc_str = self._metric_info_to_str(mc)
      mc_strs.append(mc_str)

    return self.METRIC_INFO_SEPARATOR.join(mc_strs)


@attrs.define(auto_attribs=False, frozen=True, kw_only=True)
class ProblemSerializer(s_lib.Serializer[vz.ProblemStatement]):
  """Serializes Vizier ProblemStatements."""

  primitive_serializer: s_lib.Serializer[s_lib.PrimitiveType] = attrs.field(
      factory=s_lib.PrimitiveSerializer
  )
  search_space_serializer: s_lib.Serializer[vz.SearchSpace] = attrs.field(
      factory=SearchSpaceSerializer
  )
  metrics_config_serializer: s_lib.Serializer[vz.MetricsConfig] = attrs.field(
      factory=MetricsConfigSerializer
  )
  metadata_serializer: s_lib.Serializer[vz.Metadata] = attrs.field(
      factory=metadata.MetadataSerializer
  )

  field_mapping: FieldMapping = attrs.field(factory=FieldMapping)

  include_metadata: bool = attrs.field(default=True)

  # Class variables.
  INFO_SEPARATOR: str = '&&'
  ALGORITHM_KEY: str = 'algorithm'

  def to_str(self, problem: vz.ProblemStatement, /) -> str:
    alg_info: Dict[str, s_lib.PrimitiveType] = {}
    alg_info[self.field_mapping.algorithm_name] = problem.metadata.get(
        self.ALGORITHM_KEY,
    )

    extra_info: Dict[str, s_lib.PrimitiveType] = {}

    if self.include_metadata:
      extra_info[self.field_mapping.metadata] = self.metadata_serializer.to_str(
          problem.metadata
      )

    alg_str = self.primitive_serializer.to_str(alg_info)
    metrics_str = self.metrics_config_serializer.to_str(
        problem.metric_information
    )
    ss_str = self.search_space_serializer.to_str(problem.search_space)
    extra_str = self.primitive_serializer.to_str(extra_info)

    return self.INFO_SEPARATOR.join([alg_str, metrics_str, ss_str, extra_str])
