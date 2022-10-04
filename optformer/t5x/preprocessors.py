"""T5 preprocessors."""

from typing import Dict, Mapping, Sequence

from absl import logging
from optformer.t5x import utils
import seqio
import tensorflow as tf

TargetType = utils.TargetType


def target_type_selector(features, num_initial_tokens: int = 1
                         ) -> Dict[TargetType, tf.Tensor]:
  """Return a dict of boolean sequences corresponding to each target type."""
  targets = features['targets']
  pos = tf.range(tf.size(targets), dtype=tf.int32)

  # Every trial is in the format of "[parameters] * fun_value |".
  num_parameters = features['num_parameters']
  period = num_parameters + 3
  rem = tf.math.mod(pos - num_initial_tokens, period)

  # Boolean sequence to select separator tokens.
  is_sep = tf.fill(tf.shape(targets), False)
  # Initial tokens: pos < num_initial_tokens.
  is_sep = tf.logical_or(is_sep, tf.less(pos, num_initial_tokens))
  # Token that separates parameters with metrics, rem = D.
  is_sep = tf.logical_or(is_sep, tf.equal(rem, num_parameters))
  # Token that separates trials, rem = D + 2
  is_sep = tf.logical_or(is_sep, tf.equal(rem, num_parameters + 2))

  # rem \in [0, D)
  is_param = tf.logical_and(tf.greater_equal(pos, num_initial_tokens),
                            tf.less(rem, num_parameters))

  # rem = D + 1
  is_fun = tf.logical_and(tf.greater_equal(pos, num_initial_tokens),
                          tf.equal(rem, num_parameters + 1))

  return {
      TargetType.SEPARATOR: is_sep,
      TargetType.PARAMETER: is_param,
      TargetType.FUNCTION: is_fun
  }


@seqio.map_over_dataset
def add_targets_types(features: Mapping[str, tf.Tensor],
                      num_initial_tokens: int = 1) -> Mapping[str, tf.Tensor]:
  """Add a sequence of target types."""
  target_type_dict = target_type_selector(features, num_initial_tokens)
  targets_types = tf.fill(tf.shape(features['targets']), 0)
  for t, sel_seq in target_type_dict.items():
    targets_types += tf.cast(sel_seq, tf.int32) * t.value
  tf.debugging.assert_equal(tf.reduce_any(tf.equal(targets_types, 0)), False,
                            message='Some target tokens have unknown types.')
  features = dict(features)
  features['targets_types'] = targets_types
  return features


@seqio.map_over_dataset
def add_targets_masks(features: Mapping[str, tf.Tensor],
                      masked_types: Sequence[str]) -> Mapping[str, tf.Tensor]:
  """Add a mask sequence to the targets.

  A token is masked if it belongs to one of the masked types.

  Args:
    features: feature sequence dict.
    masked_types: a list of target types to mask.

  Returns:
    Feature sequence dict with an additional binary sequence 'targets_masks' of
    type int32.
  """
  if 'targets_types' not in features:
    raise ValueError('"targets_types" does not exist in the input features.'
                     'Apply preprocessor add_targets_types first.')
  targets_types = features['targets_types']
  mask = tf.fill(tf.shape(targets_types), True)

  # Map lowercase target type name to value.
  type_name_to_value = {v.name.lower(): v.value for v in TargetType}
  for type_name in masked_types:
    type_value = type_name_to_value.get(type_name.lower())
    if type_value is not None:
      mask = tf.logical_and(mask, tf.not_equal(targets_types, type_value))
      logging.info('Preprocessor add_targets_masks: mask %s tokens.', type_name)
    else:
      raise ValueError(f'Unsupported TargetType name: {type_name}')

  features = dict(features)
  features['targets_masks'] = tf.cast(mask, tf.int32)
  return features
