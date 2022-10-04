"""Converts Study protos into various text formats."""
import abc
import enum
import json
import re
import typing
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

from absl import logging
import ml_collections
import numpy as np

from optformer.t5x import utils
from vizier import pyvizier as vz
from vizier.pyvizier import converters

Study = vz.ProblemAndTrials  # Alias
Scalar = Union[int, float, str]
Value = Union[Scalar, Sequence[Scalar]]
StudyInfo = Dict[str, Value]
ParameterInfo = Dict[str, Value]
ParameterValueDict = Dict[str, Scalar]
ParameterValueList = List[Scalar]
ParameterValues = Union[ParameterValueDict, ParameterValueList]
ParameterIntValues = List[int]
MetricValueDict = Dict[str, Scalar]
MetricValueList = List[Scalar]
MetricValues = Union[MetricValueDict, MetricValueList]
AlgorithmInfo = Union[str, int]  # int corresponds to Algorithm ENUM

Aux = Dict[str, Any]  # Auxiliary information to be returned from converter.

# Token to denote the end of study information and the start of parameters'
# information in the config.
STUDY_PARAMETER_SEPARATION_TOKEN = '&'
# Token to denote separation between two parameters' information string in the
# study config.
PARAMETER_SEPARATION_TOKEN = '*'
# Token to denote end of old trial and start of new trial.
TRIAL_SEPARATION_TOKEN = '|'
# parameter_metric_separation_token: Token to denote separation between a
# trial's parameters and metrics.
PARAMETER_METRIC_SEPARATION_TOKEN = '*'

_MINIMUM_CONFIG_PROBABILITY = {
    'bbob': 1.0,
    'default': 0.1,
}


class ConvertedStudy(NamedTuple):
  inputs: str  # Conditioning inputs.
  target_inputs: str  # Inputs aligned with the targets.
  targets: str  # Targets.
  # The number of parameters with a non-fixed value in the search space. For a
  # conditional search space, we need to count all possible parameters.
  num_parameters: int
  aux: Aux
  num_permuted_trials: int = 0


# TODO: Validate the metadata fields below and make them official.
def get_algorithm(problem: vz.ProblemStatement,
                  default_algorithm: str = '') -> AlgorithmInfo:
  """Searches problem statement metadata for 'algorithm' key."""
  return problem.metadata.get('algorithm', default=default_algorithm)


def bbob_study_name_filter(study: Study, name_list: List[str]) -> bool:
  """Skip BBOB studies if the function name is not in the list."""
  description = study.problem.metadata.get('bbob', default=None)
  if not description:
    return True
  return description.get('name') in name_list


def study_name_filter(study: Study, name_list: List[str]) -> bool:
  return study.problem.metadata.get('name') in name_list


def algorithm_name_filter(study: Study,
                          algorithm_list: Sequence[AlgorithmInfo]) -> bool:
  """Skip studies if the algorithm is not in the given list."""
  return get_algorithm(study.problem) in algorithm_list


_HPOB_TEST_SEARCH_SPACES = [
    '4796', '5527', '5636', '5859', '5860', '5891', '5906', '5965', '5970',
    '5971', '6766', '6767', '6794', '7607', '7609', '5889'
]


def hpob_augmented_study_filter(study: Study) -> bool:
  description = study.problem.metadata.get('hpob')
  if description:
    return description.get('search_space_id') in _HPOB_TEST_SEARCH_SPACES
  return True


def study_filter_combiner(
    study: Study, filter_fns: Sequence[Callable[[Study], bool]]) -> bool:
  for filter_fn in filter_fns:
    if not filter_fn(study):
      return False
  return True


class Converter(abc.ABC):
  """Base class for converting Studies into text representations for language models.

  These strings will later be fed into a tokenizer, which is dependent on
  Transformer pipeline.
  """

  @abc.abstractmethod
  def study_to_texts(self, study: Study) -> ConvertedStudy:
    raise NotImplementedError('Abstract method')

  @abc.abstractmethod
  def set_config(self, config: Dict[str, Any]):
    """Override configurations."""
    raise NotImplementedError('Abstract method')

  @abc.abstractmethod
  def pytrial_to_parameter_values(
      self, aux: Aux, trial: vz.Trial) -> Optional[ParameterIntValues]:
    """Convert a trial to parameter values represented in integers."""
    raise NotImplementedError('Abstract method')

  @abc.abstractmethod
  def parameter_texts_to_trial(
      self, aux: Aux, parameter_texts: Sequence[str]) -> vz.TrialSuggestion:
    """Reverse the mapping from parameter value to text."""
    raise NotImplementedError('Abstract method')

  @property
  @abc.abstractmethod
  def trial_token_scheme(self) -> utils.TrialTokenScheme:
    raise NotImplementedError('Abstract property')

  @property
  @abc.abstractmethod
  def num_quantized_values(self) -> int:
    """Number of quantized objective values."""
    raise NotImplementedError('Abstract property')

  @abc.abstractmethod
  def objective_value_to_int(
      self,
      value: Union[float, np.ndarray],
      aux: Aux,
      metric_name: Optional[str] = None) -> Union[int, np.ndarray]:
    """Convert a real objective value to a quantized integer."""
    raise NotImplementedError('Abstract method')


_MINIMUM_CONFIG_PROBABILITY = {
    'bbob': 1.0,
    'default': 0.1,
}


# TODO: Remove TargetType since we only support 'objective'.
@enum.unique
class TargetType(str, enum.Enum):
  OBJECTIVE = 'objective'
  IMPROVEMENT = 'improvement'


# For legacy reasons.
_scale_type_to_int_enum: Dict[vz.ScaleType, int] = {
    vz.ScaleType.LINEAR: 1,
    vz.ScaleType.LOG: 2,
    vz.ScaleType.REVERSE_LOG: 3,
    vz.ScaleType.UNIFORM_DISCRETE: 4
}

_parameter_type_to_int_enum: Dict[vz.ParameterType, int] = {
    vz.ParameterType.DOUBLE: 1,
    vz.ParameterType.INTEGER: 2,
    vz.ParameterType.CATEGORICAL: 3,
    vz.ParameterType.DISCRETE: 4
}


class OptFormerConverter(Converter):
  """OptFormer Converter.

  Default configuration is described at https://arxiv.org/abs/2205.13320v1.

  Output of study will look like the following below:

  --------------------------------------------------------------------------
  Optional_study_config, Param_1_config, Param_2_config, …, Param_D_config,
  TRIAL_SEP_TOKEN,
  Trial_1_param_1_value, ... , Trial_1_param_D_value, PARAM_SEP_TOKEN,
  Trial_1_metrics,
  TRIAL_SEP_TOKEN,
  Trial_2_param_1_value, ... , Trial_2_param_D_value, PARAM_SEP_TOKEN,
  Trial_2_metrics,
  TRIAL_SEP_TOKEN,
  …
  Trial_N_param_1_value, ... , Trial_N_param_D_value, PARAM_SEP_TOKEN,
  Trial_N_metrics,
  --------------------------------------------------------------------------

  where rows are separated by the TRIAL_SEP_TOKEN, and for each row, the param
  values and metrics are separated by the PARAM_SEP_TOKEN.
  """

  def __init__(self,
               *,
               objective_range_after_norm: Tuple[float, float] = (0., 1.0),
               rand_objective_scale_range: Optional[Sequence[float]] = None,
               num_quantized_values: int = 1000,
               study_filter: Optional[Callable[[Study], bool]] = None,
               randomize_parameter_order: bool = True,
               min_trials: int = 10,
               max_trials: Optional[int] = 200,
               float_precision_in_config: int = 3,
               discard_const_objective: bool = True,
               num_initial_tokens: int = 1,
               minimum_config: bool = False,
               minimum_config_per_study: bool = False,
               int_as_double: bool = False,
               infeasible_trial_penalty_multiplier: float = 0.0,
               target_type: TargetType = TargetType.OBJECTIVE,
               improvement_steps: Optional[int] = None,
               improvement_discount: Optional[float] = None):
    r"""Initializes the converter.

    Note that value normalization is applied before quantization.

    With the following setting:
      target_type = 'objective'
    the converted study has:
      target_inputs == targets.

    With the following setting:
      target_type = 'improvement'
      improvement_steps = H  >= 1
      improvement_discount = None
    the target value at step t is
      target_t = max y_{0:min(T-1, t+H-1)} - max y_{0:t-1}
    where T is the sequence length and target_0 is defined as 0.

    With the following setting:
      target_type = 'improvement'
      improvement_steps = None
      improvement_discount = d \in [0, 1]
    the target value at step t is
      target_t = \sum_{h=0}^{T-1-t} d^h OneStepImp_{t+h}
    with one step improvement:
      OneStepImp_{t+h} = max y_{0:t+h} - max y_{0:t+h-1}

    Args:
      objective_range_after_norm: target interval after objective normalization
        (min, max). This option is ignored if rand_objective_scale_range is not
        None.
      rand_objective_scale_range: The length of the objective value range after
        normalization if not None. After normalize the objective value to [0,
        1], the converter first samples a target range, r in
        `rand_objective_scale_range`, and sample a min value, m in [0, 1-r],
        then apply an affine transformation to the range of [m, m+r].
      num_quantized_values: number of discrete values after quantization.
      study_filter: additional study filter.
      randomize_parameter_order: randomize the nonfixed parameter order if True.
      min_trials: required minimum number of valid trials. Empty texts will be
        returned if it is not satisfied.
      max_trials: maximum number of valid trials to return. The list of trials
        will be truncated beyond this number of trials. Objective value
        normalization is applied for the first `max_trials` trials.
      float_precision_in_config: floats in the config string will be formatted
        with the given precision as, e.g., '{:.10g}'.
      discard_const_objective: study with a constant objective function will be
        discard.
      num_initial_tokens: number of initial tokens in a targets sequence before
        the first trial starts.
      minimum_config: always keep minimum config information in the inputs
        string.
      minimum_config_per_study: randomly setting minimum config per study. This
        will override minimum_config when True.
      int_as_double: treat integer as double in config string.
      infeasible_trial_penalty_multiplier: penalty multiplier to assign an
        objective value to infeasible trials. When maximizing, the objective
        value will be min - multiplier * (max - min) if min != max else min - 1.
      target_type: 'objective' or 'improvement'. The target is the objective
        value or objective improvement in future steps.
      improvement_steps: non-negative integer, the improvement is defined with a
        given number of future steps. Only one of the improvement_steps and
        improvement_discount can be not a None.
      improvement_discount: float in [0, 1], the improvement is defined with a
        discounting factor. Only one of the improvement_steps and
        improvement_discount can be not a None.
    """
    # Float formatting string, e.g. '{:.10g}' if the precision is 10.
    float_fmt = '{{:.{}g}}'.format(float_precision_in_config)
    trial_token_scheme = utils.TrialTokenScheme(num_initial_tokens)
    config = ml_collections.ConfigDict(
        dict(
            objective_range_after_norm=objective_range_after_norm,
            rand_objective_scale_range=rand_objective_scale_range,
            num_quantized_values=num_quantized_values,
            study_filter=study_filter,
            randomize_parameter_order=randomize_parameter_order,
            min_trials=min_trials,
            max_trials=max_trials,
            float_precision_in_config=float_precision_in_config,
            float_fmt=float_fmt,
            discard_const_objective=discard_const_objective,
            num_initial_tokens=num_initial_tokens,
            trial_token_scheme=trial_token_scheme,
            minimum_config=minimum_config,
            minimum_config_per_study=minimum_config_per_study,
            int_as_double=int_as_double,
            infeasible_trial_penalty_multiplier=infeasible_trial_penalty_multiplier,
            target_type=TargetType(target_type),
            improvement_steps=improvement_steps,
            improvement_discount=improvement_discount,
        ))
    self._verify_config(config)
    self._config = config

  def _verify_config(self, config: ml_collections.ConfigDict):
    self._verify_objective_range_after_norm(config.objective_range_after_norm)
    self._verify_rand_objective_scale_range(config.rand_objective_scale_range)
    self._verify_min_and_max_trials(config.min_trials, config.max_trials)
    self._verify_target_type(config.target_type, config.improvement_steps,
                             config.improvement_discount)

  def _verify_objective_range_after_norm(self, objective_range: Tuple[float,
                                                                      float]):
    """Set the min and max objective value if normalization is applied."""
    min_obj, max_obj = objective_range
    if not 0.0 <= min_obj < max_obj <= 1.0:
      raise ValueError(
          'min_obj and max_obj must satisfy '
          f'0.0 <= min_obj ({min_obj}) < max_obj ({max_obj}) <= 1.0.')

  def _verify_rand_objective_scale_range(
      self, scale_range: Optional[Sequence[float]]):
    if scale_range is not None:
      if (len(scale_range) != 2 or
          not 0. <= scale_range[0] <= scale_range[1] <= 1.0):
        raise ValueError('rand_objective_scale_range must satisfy the format of'
                         ' (m, M) with 0 <= m <= M <= 1.')

  def _verify_min_and_max_trials(self, min_trials: int,
                                 max_trials: Optional[int]):
    if max_trials is not None and min_trials > max_trials:
      raise ValueError(f'max_trials ({max_trials}) must be equal to or greater '
                       f'than min_trials ({min_trials})')

  def _verify_target_type(self,
                          target_type: TargetType,
                          improvement_steps: Optional[int] = None,
                          improvement_discount: Optional[float] = None):
    """Verify Acquisition Optimization settings."""
    if target_type == TargetType.IMPROVEMENT:
      if improvement_steps is not None and improvement_discount is not None:
        raise ValueError(
            'Only one of improvement_steps and improvement_discount can be '
            f'specified but both ({improvement_steps}, {improvement_discount}) '
            'are given non-None values.')
      if improvement_steps is not None and improvement_steps <= 0:
        raise ValueError(f'improvement_steps ({improvement_steps}) must be '
                         'greater than 0.')
      if (improvement_discount is not None and
          not 0. <= improvement_discount <= 1.0):
        raise ValueError(f'improvement_discount ({improvement_discount}) must '
                         'be in the range of [0, 1].')

  def can_be_converted(self, study: Study) -> bool:
    """Check if the study is eligible to be converted."""
    problem = study.problem

    # Study has a single objective and has no safety constraints.
    if not (problem.metric_information.is_single_objective and
            len(problem.metric_information) == 1):
      logging.warning('Study has multiple objective or contains constraints.')
      return False

    # The number of trials is more than a minimum threshold.
    if len(study.trials) < self._config.min_trials:
      logging.warning(
          'Study has only %d trials, fewer than the required '
          'min_trials value of %d.', len(study.trials), self._config.min_trials)
      return False

    # The study should not have conditional parameters.
    # Note: the current class does not support a conditional search space.
    if problem.search_space.is_conditional:
      logging.warning('Study search space is not flat.')
      return False

    if self._config.study_filter and not self._config.study_filter(study):
      logging.warning('Study does not pass the study_filter.')
      return False

    return True

  def _int_to_str(self, x: int) -> str:
    return f'<{x}>'

  def _str_to_int(self, s: str) -> int:
    m = re.fullmatch('<(\\d+)>', s)
    if not m:
      raise ValueError(f'Input string {s} is not a valid int string.')
    return int(m.group(1))

  def _value_to_str(self, v: Value) -> str:
    """Convert a value or a sequence of value to a string."""
    if isinstance(v, float):
      return self._config.float_fmt.format(v)
    elif isinstance(v, int):
      return self._int_to_str(v)
    elif isinstance(v, str):
      return json.dumps(v)
    else:
      # Sequence[Scalar]
      #
      # Note isinstance('abc', Sequence) = True. So we cannot use this
      # condition to determine if v is str or a list of scalars.
      if all([isinstance(vi, float) for vi in v]):
        # Format floats with a precision.
        return '[' + ','.join(map(self._config.float_fmt.format, v)) + ']'
      else:
        # Omit spaces after ',' comapred to json.dumps(v).
        return '[' + ','.join(map(json.dumps, v)) + ']'

  def _dict_to_str(self, d: Dict[str, Any]) -> str:
    """Convert a dict to a text string, assuming keys contain letters only."""
    return ','.join([f'{k}:{self._value_to_str(v)}' for k, v in d.items()])

  def _int_array_to_str(self, x: np.ndarray) -> str:
    """Convert a numpy int array to a text string."""
    return ''.join([self._int_to_str(int(x_i)) for x_i in x])

  @property
  def num_quantized_values(self) -> int:
    return self._config.num_quantized_values

  @property
  def trial_token_scheme(self) -> utils.TrialTokenScheme:
    return self._config.trial_token_scheme

  def quantize(self, x: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
    # Convert a fraction number to an integer. E.g. 0.12345 --> 123
    # E.g. with bins = 1000, range [0.0, 1.0] --> [0, 990]
    bins = self.num_quantized_values
    if isinstance(x, np.ndarray):
      x = np.clip(x, 0, 1)
      return np.minimum((x * bins).astype(np.int32), bins - 1)
    else:
      x = min(max(x, 0.0), 1.0)
      return min(int(x * bins), bins - 1)

  def dequantize(self, x: Union[int, np.ndarray]) -> float:
    # Reverse the _quanize method.
    bins = self.num_quantized_values
    return (x + 0.5) / bins

  def _config_primitives_to_text(
      self, study_info: StudyInfo, fixed_parameters: ParameterValueDict,
      nonfixed_parameter_infos: Dict[str, ParameterInfo]) -> str:
    """Convert primitives data to a text string."""
    study_info_string = self._dict_to_str(study_info)
    fixed_parameters_string = self._dict_to_str(fixed_parameters)
    parameter_info_strings = [
        self._dict_to_str(pi) for pi in nonfixed_parameter_infos.values()
    ]

    config_string = study_info_string + STUDY_PARAMETER_SEPARATION_TOKEN
    config_string += (
        fixed_parameters_string + STUDY_PARAMETER_SEPARATION_TOKEN)
    config_string += (PARAMETER_SEPARATION_TOKEN.join(parameter_info_strings))
    return config_string

  def _has_fixed_value(self, pc: vz.ParameterConfig) -> Tuple[bool, Any]:
    """Check whether the parameter has a fixed value."""
    if pc.type in [vz.ParameterType.DOUBLE, vz.ParameterType.INTEGER]:
      min_val, max_val = pc.bounds
      if min_val == max_val:
        return True, min_val
    else:
      # vz.ParameterType.CATEGORICAL, vz.ParameterType.DISCRETE
      if len(pc.feasible_values) == 1:
        return True, pc.feasible_values[0]
    return False, None

  def _format_possible_float_list(self, xs):
    """Format the list if all values are float or int, or return the input."""

    def maybe_float(x):
      """Return float if possible, otherwise None."""
      try:
        f = float(x)
      except ValueError:
        return None
      return f

    fs = list(map(maybe_float, xs))
    if any([f is None for f in fs]):
      return xs
    else:
      return list(map(self._config.float_fmt.format, fs))

  def _single_metric_information(
      self, sc: vz.ProblemStatement) -> Optional[vz.MetricInformation]:
    """Obtains single metric info from problem. Returns None if invalid."""
    if not sc.metric_information.is_single_objective:
      logging.warning(
          'Study %s contains multiple metrics '
          'and is not supported by this converter. Please use the '
          'is_valid method to check the validity.', sc)
      return None
    if sc.metric_information.is_safety_metric:
      logging.warning('Study contains a safety metric and is not supported: %s',
                      sc)
      return None
    return sc.metric_information.item()

  def study_config_to_primitives(
      self, sc: vz.ProblemStatement, aux: Aux
  ) -> Tuple[StudyInfo, ParameterValueDict, Dict[str, ParameterInfo]]:
    """Convert study config to intermediate dicts of primitives."""
    null_return = {}, {}, {}

    minimum_config = self._use_minimum_config(sc)

    metric_information = self._single_metric_information(sc)
    if metric_information is None:
      return null_return

    study_info = {}
    study_info['N'] = sc.metadata.get(
        'name', default='') if not minimum_config else ''
    study_info['A'] = get_algorithm(sc)
    study_info['O'] = metric_information.name if not minimum_config else ''
    study_info['G'] = metric_information.goal.value

    fixed_parameters = {}

    nonfixed_parameter_infos = {}
    # parameter_configs = list(study.study_config.parameter_configs)
    # parameter_configs.sort(key=lambda x: x.name)
    for pc in sc.search_space.parameters:
      # Skip parameters of a fixed value.
      is_fixed, fixed_value = self._has_fixed_value(pc)
      if is_fixed:
        if not minimum_config:
          fixed_parameters[pc.name] = fixed_value
        continue

      parameter_info = {}
      # Parameter name.
      parameter_info['N'] = pc.name if not minimum_config else ''
      # Parameter scaling.
      if vz.ParameterType.is_numeric(pc.type):
        parameter_info['S'] = 0
        if pc.scale_type is not None and not minimum_config:
          parameter_info['S'] = _scale_type_to_int_enum[pc.scale_type]

      # Parameter type.
      parameter_type = pc.type
      if (pc.type == vz.ParameterType.INTEGER and self._config.int_as_double):
        parameter_type = vz.ParameterType.DOUBLE
      parameter_info['P'] = _parameter_type_to_int_enum[parameter_type]
      # Parameter value range or set depending on the type.
      # TODO: Consider showing original bounds/feasible values in
      # metadata as opposed to log-scale. Unlikely to change performance though,
      # since correlation strength will roughly be the same b/w "CIFAR10" and
      # the scaled/unscaled bounds/feasible values.
      if pc.type in [vz.ParameterType.DOUBLE, vz.ParameterType.INTEGER]:
        min_val, max_val = pc.bounds
        if pc.scale_type is not None and pc.scale_type.is_nonlinear():
          min_val, max_val = np.log(min_val), np.log(max_val)
        if not minimum_config:
          parameter_info['m'] = min_val
          parameter_info['M'] = max_val
        elif (pc.type == vz.ParameterType.INTEGER and
              not self._config.int_as_double):
          # If using minimum_config and an integer parameter is not used as
          # double, provided the number of valid integer values. This is
          # necessary for the model to infer the set of valid values in the
          # minimum configuration setting.
          parameter_info['L'] = max_val - min_val + 1
      elif pc.type == vz.ParameterType.CATEGORICAL:
        parameter_info['L'] = len(pc.feasible_values)
        if not minimum_config:
          parameter_info['C'] = self._format_possible_float_list(
              pc.feasible_values)
      elif pc.type == vz.ParameterType.DISCRETE:
        # TODO: Apply log/reverse-log scaling.
        parameter_info['L'] = len(pc.feasible_values)
        if not minimum_config:
          parameter_info['F'] = pc.feasible_values

      if (pc.type in [vz.ParameterType.CATEGORICAL, vz.ParameterType.DISCRETE]
          and parameter_info['L'] > self.num_quantized_values):
        # When applying parameter normalization, we quantize double and
        # integer parameters. Then we will not consider discrete or
        # categorical parameters with too many feasible values. Otherwise,
        # the integer token vocabulary will not be bounded.
        return null_return

      nonfixed_parameter_infos[pc.name] = parameter_info

    nonfixed_parameter_infos = self._update_aux_and_maybe_permute_parameters(
        aux, sc, minimum_config, fixed_parameters, nonfixed_parameter_infos)

    return study_info, fixed_parameters, nonfixed_parameter_infos

  def _normalize_objectives(self, aux: Aux, metric_name: str,
                            objectives: np.ndarray) -> Optional[np.ndarray]:
    """Normalize and quantize study objectives, return None if failed.

    Linear mapping from [obj_min, obj_max] to [tgt_min, tgt_max]:
    (x - obj_min) / (obj_max - obj_min) * (tgt_max - tgt_min) + tgt_min
    = (tgt_max - tgt_min) / (obj_max - obj_min) * x
      + tgt_min - (tgt_max - tgt_min) / (obj_max - obj_min) * obj_min

    Update aux with:
      scale = (tgt_max - tgt_min) / (obj_max - obj_min)
      offset = tgt_min - scale * obj_min

    The target range [tgt_min, tgt_max] is sampled if
    rand_objective_scale_range is not None, or specified by
    objective_range_after_norm.

    TODO: apply for better normalization function.

    Args:
      aux: Aux object
      metric_name: metric name
      objectives: raw objective values.

    Returns:
      normalized and quantized objective values or None if failed.
    """
    obj_min = min(objectives)
    obj_max = max(objectives)

    if obj_min == obj_max:
      if self._config.discard_const_objective:
        # Discard the study if an objective is constant. Currently we only
        # consider studies with a single objective function. If the
        # objective is constant, it is not a good training example.
        logging.warning('Discard the study because the objective is '
                        'constant.')
        return None
      else:
        # Set the value range to [const_value - 1., const_value + 1.].
        obj_min = obj_min - 1.
        obj_max = obj_max + 1.
    # Compute / sample the target objective range after normalization.
    if self._config.rand_objective_scale_range is None:
      # Normalize to [target_min, target_max].
      tgt_min, tgt_max = self._config.objective_range_after_norm
    else:
      # Sample a target range.
      min_range, max_range = self._config.rand_objective_scale_range
      tgt_range = (min_range + (max_range - min_range) * np.random.uniform())
      assert 0 < tgt_range <= 1
      # Sample the lower bound.
      tgt_min = (1. - tgt_range) * np.random.uniform()
      tgt_max = tgt_min + tgt_range

    scale = (tgt_max - tgt_min) / (obj_max - obj_min)
    offset = tgt_min - scale * obj_min
    # Update aux.
    aux['objective_mapping'] = {metric_name: {'scale': scale, 'offset': offset}}

    # Apply the transformation and quantization.
    objectives = typing.cast(
        np.ndarray, self.objective_value_to_int(objectives, aux, metric_name))

    # Make sure the quantized objective value is in [0, Q).
    if not (objectives.min() >= 0 and
            objectives.max() < self._config.num_quantized_values):
      logging.warning('Quantized objectives are out of boundary.')
      return None
    return objectives

  def _trial_primitives_to_text(
      self, parameters: np.ndarray, objectives: np.ndarray,
      targets_or_none: Optional[np.ndarray]) -> Tuple[str, str]:
    """Convert all trial primitives to target inputs and targets string."""
    param_strings = [self._int_array_to_str(p) for p in parameters]
    obj_strings = [self._int_to_str(int(o)) for o in objectives]
    param_obj_strings = [
        p + PARAMETER_METRIC_SEPARATION_TOKEN + o
        for p, o in zip(param_strings, obj_strings)
    ]
    target_inputs_string = TRIAL_SEPARATION_TOKEN.join(param_obj_strings)

    if targets_or_none is None:
      targets_string = target_inputs_string
    else:
      tgt_strings = [self._int_to_str(int(t)) for t in targets_or_none]
      param_tgt_strings = [
          p + PARAMETER_METRIC_SEPARATION_TOKEN + t
          for p, t in zip(param_strings, tgt_strings)
      ]
      targets_string = TRIAL_SEPARATION_TOKEN.join(param_tgt_strings)
    return target_inputs_string, targets_string

  def trials_to_primitives(
      self, sc: vz.ProblemStatement, trials: Sequence[vz.Trial], aux: Aux
  ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Convert trials to parameters, objectives and targets, or None if fail."""
    null_return = None, None, None

    metric_information = self._single_metric_information(sc)
    if metric_information is None:
      return null_return
    metric_name = metric_information.name

    # Find a list of valid trials and extract parameter and objective values.
    parameters = []
    objectives = []
    trial_ids = []
    for pytrial in trials:
      trial_parameters, trial_objective = self._pytrial_to_primitives(
          aux, metric_name, pytrial)
      if trial_parameters is None:
        # Skip the trial if it fails to be converted to primitives.
        continue

      parameters.append(trial_parameters)
      objectives.append(trial_objective)
      trial_ids.append(pytrial.id)

      if (self._config.max_trials and
          len(parameters) >= self._config.max_trials):
        # Ignore trials after `max_trials` trials. Note that this truncation
        # will affect the study_objectives normalization.
        break

    if (not objectives or len(objectives) < self._config.min_trials):
      # Skip the study if there are not enough number of valid trials.
      logging.warning(
          'There are no valid trials or the number (%d) is less '
          'than %d.', len(objectives), self._config.min_trials)
      return null_return

    parameters = np.array(parameters, dtype=np.int32)
    objectives = np.array(objectives, dtype=np.float32)
    aux['trial_ids'] = trial_ids

    if np.any(np.isinf(objectives)):
      # Skip the study if there are inf after converted to float32.
      logging.warning('Some trials have inf objective value in float32.')
      return null_return

    # Impute NaN values from infeasible trials.
    objectives = self._impute_nan_objectives(metric_information, objectives)
    if objectives is None:
      logging.warning('Failed to impute objectives with NaN values.')
      return null_return

    # Normalize study objectives.
    objectives = self._normalize_objectives(aux, metric_name, objectives)
    if objectives is None:
      return null_return

    targets_or_none = self._targets(metric_information, objectives)
    if targets_or_none is not None:
      num_targets = targets_or_none.shape[0]
      if num_targets < self._config.min_trials:
        # Skip the study if there are not enough number of valid targets.
        logging.warning('The number of valid targets (%d) is less than %d.',
                        num_targets, self._config.min_trials)
        return null_return
      parameters = parameters[:num_targets]
      objectives = objectives[:num_targets]

    return parameters, objectives, targets_or_none

  def _impute_nan_objectives(self, metric_information: vz.MetricInformation,
                             objectives: np.ndarray) -> Optional[np.ndarray]:
    """Assign a bad value to infeasible trials (nan), return None if failed."""
    if np.any(np.isnan(objectives)):
      nonnan_values = objectives[~np.isnan(objectives)]
      if nonnan_values.size == 0:
        # All values are nan.
        return None
      obj_min = nonnan_values.min()
      obj_max = nonnan_values.max()
      multiplier = self._config.infeasible_trial_penalty_multiplier
      # We applying the multiplier only when obj_max > obj_min. When they are
      # equal, it does not make any difference to scale the constant 1.0 because
      # after normalization, there are always two unique values, 1 for
      # regular observations and 0 for infeasible trials if maximization.
      penalty = (multiplier * (obj_max - obj_min)) if obj_max > obj_min else 1.0
      if metric_information.goal.is_maximize:
        nan_value = obj_min - penalty
      else:
        nan_value = obj_max + penalty
      objectives[np.isnan(objectives)] = nan_value
    return objectives

  def _use_minimum_config(self, sc: vz.ProblemStatement) -> bool:
    """Determine whether to keep minimum information in a converted study."""
    if self._config.minimum_config_per_study:
      # Sample the bernoulli variable according to a probability per source.
      description = sc.metadata.get('bbob')
      study_source = 'bbob' if description else 'default'
      prob = _MINIMUM_CONFIG_PROBABILITY[study_source]
      minimum_config = np.random.uniform() < prob
      return minimum_config
    else:
      return self._config.minimum_config

  def study_to_texts(self, study: Study) -> ConvertedStudy:
    """Convert the study to a dict of texts. Returns empty strings if fails."""
    null_return = ConvertedStudy(
        inputs='', target_inputs='', targets='', num_parameters=0, aux=dict())

    if not self.can_be_converted(study):
      logging.warning('Study is not eligible to be converted.')
      return null_return

    # Prepare the dict to store auxiliary information for study conversion.
    aux = dict()

    # Extract information from study config.
    sc = study.problem
    study_info, fixed_parameters, nonfixed_parameter_infos = (
        self.study_config_to_primitives(sc, aux))
    if not nonfixed_parameter_infos:
      # No parameters with non-fixed values.
      logging.warning('All parameters are fixed.')
      return null_return

    # Convert study config to a string.
    config_string = self._config_primitives_to_text(study_info,
                                                    fixed_parameters,
                                                    nonfixed_parameter_infos)

    # Extract information from trials.
    parameters, objectives, targets_or_none = self.trials_to_primitives(
        sc, study.trials, aux)
    if parameters is None:
      # Skip the study if None is returned.
      return null_return

    # Convert trials to a string.
    target_inputs_string, targets_string = self._trial_primitives_to_text(
        parameters, objectives, targets_or_none)

    return ConvertedStudy(
        inputs=config_string,
        target_inputs=target_inputs_string,
        targets=targets_string,
        num_parameters=len(nonfixed_parameter_infos),
        aux=aux)

  def _update_aux_and_maybe_permute_parameters(self, aux, sc, minimum_config,
                                               fixed_parameters,
                                               nonfixed_parameter_infos):
    """Update aux with parameter configs and maybe permute parameter order."""
    aux['minimum_config'] = minimum_config

    # Determine the parameter order and reorder nonfixed_parameter_infos.
    parameter_names = list(nonfixed_parameter_infos.keys())
    if self._config.randomize_parameter_order:
      parameter_names = np.random.permutation(parameter_names)
    else:
      parameter_names = sorted(parameter_names)
    nonfixed_parameter_infos = {
        name: nonfixed_parameter_infos[name] for name in parameter_names
    }

    # Update aux with parameter_name_to_configs in the order of
    # nonfixed_parameter_names.
    all_parameter_name_to_configs = {
        p.name: p for p in sc.search_space.parameters
    }
    aux['parameter_name_to_configs'] = {
        name: all_parameter_name_to_configs[name] for name in parameter_names
    }

    # Update aux with fixed_parameter_name_to_config_and_values.
    aux['fixed_parameter_name_to_config_and_values'] = {
        name: (all_parameter_name_to_configs[name], value)
        for name, value in fixed_parameters.items()
    }
    return nonfixed_parameter_infos

  def pytrial_to_parameter_values(
      self, aux: Aux, trial: vz.Trial) -> Optional[ParameterIntValues]:
    """Given a trial, return a dict of parameter values or None."""
    parameter_name_to_configs = aux['parameter_name_to_configs']
    trial_parameters = []
    for p_name, p_config in parameter_name_to_configs.items():
      if p_name not in trial.parameters:
        return None

      value = trial.parameters[p_name].value
      if p_config.type in [
          vz.ParameterType.CATEGORICAL, vz.ParameterType.DISCRETE
      ]:
        # Replaces categorical strings or discrete values with index from study
        # config.
        # TODO: Shall we treat a discrete parameter as a numeric value
        # and normalize it?
        if p_config.type == vz.ParameterType.DISCRETE:
          # Convert to np.float32 to avoid errors due to small differences.
          feasible_values = [np.float32(v) for v in p_config.feasible_values]
          value = np.float32(value)
        else:
          feasible_values = p_config.feasible_values
        try:
          trial_parameters.append(feasible_values.index(value))
        except ValueError as e:
          logging.warning(
              'Failed to find the value, %s, of parameter, %s, in '
              'the list of feasible values. Error message: %s', p_name, value,
              str(e))
          return None
      else:
        min_val, max_val = p_config.bounds
        if min_val == max_val:
          raise ValueError(
              f'The value range ({min_val}, {max_val}) of nonfixed parameter '
              f'({p_name}) should not be empty.')

        if not min_val <= value <= max_val:
          # Skip a trial if the parameter value is out of bounds.
          logging.warning('Parameter value %s is out of the bounds (%s, %s)',
                          value, min_val, max_val)
          return None

        # Normalize parameter value.
        converter = converters.DefaultModelInputConverter(
            p_config, scale=True, float_dtype=np.float64)
        value = converter.convert([trial])[0, 0]
        value = float(value)

        # Quantize to an integer.
        value = typing.cast(int, self.quantize(value))

        trial_parameters.append(value)

    return trial_parameters

  def parameter_texts_to_trial(
      self, aux: Aux, parameter_texts: Sequence[str]) -> vz.TrialSuggestion:
    """Reverse the mapping from parameter value to text."""
    parameter_name_to_configs = aux['parameter_name_to_configs']

    if len(parameter_name_to_configs) != len(parameter_texts):
      raise ValueError(f'The number of parameters ({len(parameter_texts)}) '
                       'does not match the number of parameters '
                       f'({len(parameter_name_to_configs)}).')
    p_dict = {}
    for (p_name, p_config), p_text in zip(parameter_name_to_configs.items(),
                                          parameter_texts):
      quantized_value = self._str_to_int(p_text)

      if p_config.type in [
          vz.ParameterType.CATEGORICAL, vz.ParameterType.DISCRETE
      ]:
        # Quantized value is the index.
        value = p_config.feasible_values[quantized_value]
      else:
        normalized_value = self.dequantize(quantized_value)

        # Undo the normalization.
        converter = converters.DefaultModelInputConverter(
            p_config, scale=True, float_dtype=np.float64)
        value = converter.to_parameter_values(
            np.array(normalized_value))[0].value  # pytype:disable=attribute-error
        value = float(value)

        if p_config.type == vz.ParameterType.INTEGER:
          value = round(value)

      p_dict[p_name] = value

    # Add fixed parameter values if any.
    for p_name, (
        _, value) in aux['fixed_parameter_name_to_config_and_values'].items():
      p_dict[p_name] = value

    suggestion = vz.TrialSuggestion()
    # Sort the parameters according to its names.
    for p_name in sorted(p_dict):
      suggestion.parameters[p_name] = p_dict[p_name]
    return suggestion

  def objective_value_to_int(
      self,
      value: Union[float, np.ndarray],
      aux: Aux,
      metric_name: Optional[str] = None) -> Union[int, np.ndarray]:
    """Convert the objective value to a quantized integer."""
    if metric_name is None:
      if len(aux['objective_mapping']) > 1:
        raise ValueError('Cannot identify metric name')
      map_params = list(aux['objective_mapping'].values())[0]
    else:
      map_params = aux['objective_mapping'][metric_name]

    scale = map_params['scale']
    offset = map_params['offset']
    if isinstance(value, np.ndarray):
      # Use a high precision to avoid numerical errors that cause the normalized
      # value to be out of the range of [0, 1].
      value = value.astype(np.float64)
    normalized_value = value * scale + offset
    return self.quantize(normalized_value)

  def _pytrial_to_primitives(
      self, aux: Aux, metric_name: str,
      trial: vz.Trial) -> Tuple[Optional[ParameterIntValues], Optional[float]]:
    """Convert a trial to primitives. Returns Nones if fails."""
    null_value = None, None

    parameters = self.pytrial_to_parameter_values(aux, trial)
    if parameters is None:
      return null_value

    if trial.status == vz.TrialStatus.COMPLETED:
      if trial.infeasible:
        # The trial is infeasible because of job failures, diverging gradients,
        # etc, which can be due to parameter choices (e.g. too high learning
        # rate) or due to user errors (e.g. code error, resource contention).
        # Here we assign np.nan to infeasible trials.
        objective = np.nan
      else:
        if not trial.final_measurement:
          logging.warning(
              'A feasible completed trial must have final_measurement.')
          return null_value
        objective = trial.final_measurement.metrics.get_value(metric_name, None)
        if objective is None:
          # Skip the trial if the metric is missing.
          return null_value
    elif trial.infeasible:
      # Final measurement does not exist if
      # 1) the trial is infeasible due to job failures, diverging gradients,
      # etc. or
      # 2) the trial was never completed.
      # Infeasibility can be due to parameter choices (e.g. too high learning
      # rate) or due to user errors (e.g. code error, resource contention).
      # Here we assign np.nan to infeasible trials and discard other incompleted
      # trials.
      objective = np.nan
    else:
      return null_value

    return parameters, objective

  def set_config(self,
                 config: Optional[Dict[str, Any]] = None,
                 **kwargs) -> None:
    """Override configurations."""
    config = config or {}
    config.update(kwargs)
    for k in config:
      if k not in config:
        raise ValueError(f'Unknown config name: {k}.')
    self._config.update(config)
    self._verify_config(self._config)

  def _targets(self, metric_information: vz.MetricInformation,
               objectives: np.ndarray) -> Optional[np.ndarray]:
    """Prediction targets, None if the targets are the objectives.

    The improvement values are always within [0, num_quantized_values). So after
    quantization and tokenization, they will be converted to single int tokens.

    Args:
      metric_information: metric information.
      objectives: 1D array of objective values.

    Returns:
      1D array of target values or None if the target_type == objective.
    """
    if self._config['target_type'] == TargetType.OBJECTIVE:
      return None
    elif self._config['target_type'] == TargetType.IMPROVEMENT:
      if metric_information.goal.is_minimize:
        # Flip objectives.
        objectives = -objectives

      if self._config['improvement_steps'] is not None:
        # Return in the next `improvement_steps` steps.

        steps = self._config['improvement_steps']
        max_obj = np.maximum.accumulate(objectives)
        # imp[t+1] = m[t+s] - m[t], t = [0, N-s)
        imp = max_obj[steps:] - max_obj[:-steps]
        # Append improvement at step 0, always assumed to be 0.
        imp = np.concatenate([[0.], imp])  # Shape of [N - steps + 1].
      else:
        # Cumulative return with a discount of `improvement_discount`.
        discount = self._config['improvement_discount']
        max_obj = np.maximum.accumulate(objectives)
        # 1 step improvement[t+1] = m[t+1] - m[t], t = [0, N-1)
        one_step_imp = max_obj[1:] - max_obj[:-1]
        # Append improvement at step 0, always assumed to be 0.
        one_step_imp = np.concatenate([[0.], one_step_imp])  # Shape of [N].

        # Working from the last step backward.
        imp = np.zeros_like(one_step_imp, dtype=np.float32)
        imp[-1] = one_step_imp[-1]
        for t in range(len(imp) - 2, 0, -1):
          imp[t] = one_step_imp[t] + discount * imp[t + 1]

      imp = imp.astype(objectives.dtype)

      # Make sure the improvement int value is in [0, Q).
      if not (imp.min() >= 0 and imp.max() < self.num_quantized_values):
        logging.warning('Computed improvements are out of boundary.')
        return None
      return imp
    else:
      raise ValueError('Unsupported target_type value: '
                       f'{self._config["target_type"]}')
