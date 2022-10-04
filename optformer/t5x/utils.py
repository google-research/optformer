"""Utility functions and types."""
import enum
from typing import Tuple, Union

import tensorflow as tf

Int = Union[int, tf.Tensor]


class TargetType(enum.Enum):
  SEPARATOR = 1
  PARAMETER = 2
  FUNCTION = 3
  OTHER = 4


class TrialTokenScheme(object):
  """Calculate trial token allocations.

  Every trial is in the format of "[parameters] * fun_value |".

  The entire token sequence is in the format of
    "INIT_TOKENS TRIAL_0 TRIAL_1 ... TRIAL_LAST_EXCEPT_|"
  """

  def __init__(self, num_initial_tokens: Int):
    self.num_initial_tokens = num_initial_tokens

  def trial_length(self, num_parameters: Int) -> Int:
    return num_parameters + 3

  def max_trials(self, num_parameters: Int, sequence_length: Int) -> Int:
    """Compute the maximum number of trials in a sequence."""
    num_trials = (sequence_length - self.num_initial_tokens
                  + 1  # The "|" token of the last trial is not needed.
                  ) // self.trial_length(num_parameters)
    return num_trials

  def fun_index_in_trial(self, num_parameters: Int, trial_index: Int) -> Int:
    """Index of the function token in a trial."""
    trial_len = self.trial_length(num_parameters)
    return (self.num_initial_tokens + (trial_index + 1) * trial_len - 1) - 1

  def param_index_range_in_trial(self, num_parameters: Int,
                                 trial_index: Int) -> Tuple[Int, Int]:
    """Start and end (excluded) indices of the parameter tokens in a trial."""
    trial_len = self.trial_length(num_parameters)
    start_index = self.num_initial_tokens + trial_index * trial_len
    end_index = start_index + num_parameters
    return (start_index, end_index)
