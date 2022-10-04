"""Utility class and methods to run inference from a trained model."""

import copy
import dataclasses
import math
import os
import re
import tempfile
import typing
from typing import Any, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

from absl import logging
from flax.core.frozen_dict import FrozenDict
import gin
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from optformer.data import converters
from optformer.data import tasks as t5_tasks
from optformer.t5x import models as vizier_models
import seqio
from t5x import decoding
from t5x import gin_utils
from t5x import models
from t5x import partitioning
from t5x import train_state as train_state_lib
from t5x import utils
import tensorflow as tf
from tensorflow.io import gfile
from vizier import pyvizier as vz

Aux = converters.Aux
Batch = Mapping[str, jnp.ndarray]

OPTFORMER_STUDY_CONVERTER_KWARGS = {
    'objective_range_after_norm': (0.2, 0.8),
    'rand_objective_scale_range': None,
    # Do not filter studies.
    'study_filter': None,
    'min_trials': 0,
    'max_trials': 1000,
    'discard_const_objective': False,
    # Disable random data augmentation.
    'minimum_config_per_study': False,
}

DEFAULT_GIN_SEARCH_PATHS = (
    'optformer/t5x/configs',
)

_DEFAULT_GIN_PATTERNS_TO_SKIP = [
]


def update_sequence(sequence: jnp.ndarray,
                    index: jnp.ndarray,
                    value: jnp.ndarray) -> jnp.ndarray:
  """Insert value to sequence at index."""
  index = jnp.asarray(index, dtype=jnp.int32)
  value = jnp.expand_dims(jnp.asarray(value, dtype=sequence.dtype), axis=1)
  one_hot = jax.nn.one_hot(index, sequence.shape[1], dtype=sequence.dtype)
  new_sequence = sequence * (1 - one_hot) + value * one_hot
  return new_sequence


def logits_to_log_probs(logits: jnp.ndarray) -> jnp.ndarray:
  logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
  log_probs = logits - logits_sum
  return log_probs


def config_from_model_dir(model_dir: str,
                          gin_patterns_to_skip: Sequence[str]) -> str:
  """Modify the config file in a model dir and write it to a local copy."""
  config_file = os.path.join(model_dir, 'config.gin')

  with gfile.Open(config_file, 'r') as f:
    config_str = f.read()

  # Remove ending "\\" in commented lines.
  modified = []
  for line in config_str.split('\n'):
    cur_line = line + ''  # Make a copy.
    if cur_line.strip().startswith('#'):
      while cur_line.strip().endswith('\\'):
        cur_line = cur_line.rstrip()[:-1]
    modified.append(cur_line)
  config_str = '\n'.join(modified)

  # Remove line continuation.
  config_str = config_str.replace('\\\n', ' ')

  modified = []
  for line in config_str.split('\n'):
    # Comment lines matching any of the given list of patterns.
    for pattern in gin_patterns_to_skip:
      if re.fullmatch(pattern, line):
        modified.append('# ' + line)
        break
    else:
      modified.append(line)
  config_str = '\n'.join(modified)

  local_config_file = tempfile.NamedTemporaryFile(
      mode='wt', prefix='config_', suffix='.gin', dir='/tmp', delete=False)
  local_config_file.write(config_str)
  local_config_file.close()

  return local_config_file.name


def restore_train_state(
    model: models.BaseTransformerModel,
    batch_size: int,
    sequence_length: Mapping[str, int],
    partitioner: partitioning.BasePartitioner,
    restore_checkpoint_cfg: Optional[utils.RestoreCheckpointConfig] = None,
    from_scratch: bool = False,
    init_rng: Optional[jnp.ndarray] = None,
    restore_from_model_dir: Optional[str] = None
) -> Tuple[train_state_lib.TrainState, train_state_lib.TrainState]:
  """Restores optimizer given the appropriate arguments.

  Args:
    model: t5x model to restore, can be extracted via gin.
    batch_size: Used to determine input shape.
    sequence_length: Used to determine input shape.
    partitioner: Configuration for model parallelism, can be extracted via gin.
    restore_checkpoint_cfg: Configuration for checkpoint loading.
    from_scratch: Whether to force checkpoint to start from initialization.
    init_rng: rng for initializing parameters.
    restore_from_model_dir: model_dir to restore the model from.

  Returns:
    train_state: the restored train state.
    train_state_axes: a TrainState object containing a PartitionSpec (or
      None) for each parameter, in place of the parameter itself.
  """
  if init_rng is None:
    init_rng = jax.random.PRNGKey(17)
  input_shapes = _get_input_shapes(batch_size, sequence_length, model)
  train_state_initializer = utils.TrainStateInitializer(
      optimizer_def=model.optimizer_def,
      init_fn=model.get_initial_variables,
      input_shapes=input_shapes,
      partitioner=partitioner)

  train_state_axes = train_state_initializer.train_state_axes

  if (from_scratch or not restore_checkpoint_cfg or
      not restore_checkpoint_cfg.path):
    train_state = train_state_initializer.from_scratch(init_rng=init_rng)
  else:
    restore_checkpoint_cfgs = []
    if restore_from_model_dir:
      restore_checkpoint_cfgs.append(
          dataclasses.replace(
              restore_checkpoint_cfg,
              path=restore_from_model_dir,
              mode='latest'))
    if restore_checkpoint_cfg.path:
      restore_checkpoint_cfgs += [restore_checkpoint_cfg]
    train_state = train_state_initializer.from_checkpoint(
        restore_checkpoint_cfgs, init_rng=init_rng)
  if not train_state:
    raise ValueError('error initializing train state')

  return train_state, train_state_axes


def _get_input_shapes(
    batch_size: int, sequence_length: Mapping[str, int],
    model: models.BaseTransformerModel) -> Mapping[str, Tuple[int, int]]:
  """Get input shapes for models.

  Args:
    batch_size: int
    sequence_length: dictionary with the length of the inputs and targets.
    model: Model class to determine the class of model that we have: Encoder,
      Decoder, or Encoder+Decoder.

  Returns:
    Dictionary which encodes the input shapes.
  """
  model_feats = model.FEATURE_CONVERTER_CLS().MODEL_FEATURES
  context_length = sequence_length['targets']
  if 'inputs' in sequence_length:
    context_length += sequence_length['inputs']
  if ('encoder_input_tokens' in model_feats and
      'decoder_input_tokens' in model_feats):
    return {
        'encoder_input_tokens': (batch_size, sequence_length['inputs']),
        'decoder_input_tokens': (batch_size, sequence_length['targets'])
    }
  elif ('encoder_input_tokens' not in model_feats and
        'decoder_input_tokens' in model_feats):
    return {'decoder_input_tokens': (batch_size, context_length)}
  elif ('encoder_input_tokens' in model_feats and
        'decoder_input_tokens' not in model_feats):
    return {'decoder_input_tokens': (batch_size, sequence_length['inputs'])}
  else:
    raise ValueError('Input type not supported')


def _verify_indices_shapes(batch, indices):
  batch_size = batch['decoder_input_tokens'].shape[0]
  if batch_size != len(indices) or indices.ndim != 1:
    raise ValueError('Indices array must be 1-dimensional and match the '
                     'length of decoder_input_tokens.')


def default_study_converter_kwargs(study_converter):
  """Default study converter kwargs for the inference model."""
  if isinstance(study_converter, converters.OptFormerConverter):
    study_converter_kwargs = OPTFORMER_STUDY_CONVERTER_KWARGS
  else:
    raise ValueError(
        f'Unsupported study converter type {type(study_converter)}')
  return study_converter_kwargs


class InferenceModel(object):
  """Simple wrapper to load pretrained model and run prediction."""
  infer_task_count: int = 0

  def __init__(
      self,
      model: models.BaseTransformerModel,
      checkpoint_path: str,
      model_dir: str,
      params: train_state_lib.FrozenVariableDict,
      train_state_axes: train_state_lib.TrainState,
      partitioner: partitioning.BasePartitioner,
      batch_size: int,
      vocab: seqio.Vocabulary,
      vocab_index_from: int,  # First vocabulary index in prediction.
      num_embeddings: int,  # Embedding dimension, same as the output dimension.
      task_feature_lengths: Mapping[str, int],
      num_initial_tokens: int = 1,  # Number of initial tokens in the target
      # string before first trial starts.
      dataset_builder: Optional[t5_tasks.DatasetFromStudy] = None):
    self._model = model
    self._checkpoint_path = checkpoint_path
    self._model_dir = model_dir
    self._params = params
    self._train_state_axes = train_state_axes
    self._partitioner = partitioner
    device_count = jax.device_count()
    batch_size = (batch_size + device_count - 1) // device_count * device_count
    self._batch_size = batch_size  # A multiple of device_count.
    self._vocab = vocab
    self._task_feature_lengths = task_feature_lengths
    self._num_initial_tokens = num_initial_tokens
    self._dataset_builder = (dataset_builder if dataset_builder
                             else self._make_dataset_builder())
    self._vocab_index_from = vocab_index_from
    self._vocab_index_to = (
        vocab_index_from + self.study_converter.num_quantized_values)
    self._num_embeddings = num_embeddings
    self._rng = random.PRNGKey(0)

    self._partitioned_compute_logits_from_batch = self._partitioner.partition(
        self._compute_logits_from_batch,
        in_axis_resources=(train_state_axes.params,
                           partitioning.PartitionSpec('data',)),
        out_axis_resources=partitioning.PartitionSpec('data',),
        static_argnums=[2])

    self._partitioned_model_predict_fn = self._partitioner.partition(
        self._model_predict_fn,
        in_axis_resources=(self._train_state_axes.params,
                           partitioning.PartitionSpec('data',), None, None),
        out_axis_resources=partitioning.PartitionSpec('data',),
        static_argnums=[4, 5])

  @classmethod
  def from_checkpoint(
      cls,
      checkpoint_path_or_model_dir: str,
      batch_size: int = 1,
      model_gin_file: Optional[str] = None,
      gin_patterns_to_skip: Optional[Sequence[str]] = None,
      gin_search_paths: Sequence[str] = DEFAULT_GIN_SEARCH_PATHS,
      overwrite_gin_files: Optional[Sequence[str]] = None,
      overwrite_gin_bindings: Optional[Sequence[str]] = None,
  ) -> 'InferenceModel':
    """Create an inference model from a checkpoint path or model directory.

    The model_gin_file, or if None, the config file from the model directory
    will be applied first except the training script related configs. Then if
    overwrite_gin_files or overwrite_gin_bindings are provided, they will be
    applied to overwrite the configurations.

    Args:
      checkpoint_path_or_model_dir: checkpoint path or model directory.
      batch_size: batch size.
      model_gin_file: model gin file if not None, otherwise use the default file
        in the model directory.
      gin_patterns_to_skip: sequence of gin string patterns with which lines in
        the model gin file will be skipped.
      gin_search_paths: paths that will be searched for gin files.
      overwrite_gin_files: paths to gin config files to be parsed. Files will be
        parsed in order with conflicting settings being overridden by later
        files. Paths may be relative to paths in `gin_search_paths`.
      overwrite_gin_bindings: individual gin bindings to be applied after the
        gin files are parsed. Will be applied in order with conflicting settings
        being overridden by later ones.

    Returns:
      An InferenceModel instance.
    """
    gin_patterns_to_skip = list(
        gin_patterns_to_skip) if gin_patterns_to_skip else []
    for pattern in _DEFAULT_GIN_PATTERNS_TO_SKIP:
      if pattern not in gin_patterns_to_skip:
        gin_patterns_to_skip.append(pattern)

    checkpoint_path_or_model_dir = os.path.normpath(
        checkpoint_path_or_model_dir)
    dirname = checkpoint_path_or_model_dir.split(os.sep)[-1]
    if not dirname.startswith('checkpoint_'):
      # The input is a model directory.
      model_dir = checkpoint_path_or_model_dir
      # Look for the latest checkpoint in the directory.
      checkpoints = gfile.glob(os.path.join(model_dir, 'checkpoint_*'))
      checkpoints = [path for path in checkpoints
                     if re.fullmatch('.*checkpoint_[\\d]+$', path)]
      checkpoint_path = sorted(checkpoints,
                               key=lambda s: int(s.split('_')[-1]))[-1]
      logging.info('Model dir: %s', model_dir)
      logging.info('Found the latest checkpoint path in the model dir: %s',
                   checkpoint_path)
    else:
      # The input is a model checkpoint path under the model directory.
      checkpoint_path = checkpoint_path_or_model_dir
      model_dir = os.path.dirname(checkpoint_path_or_model_dir)
      logging.info('Model dir: %s', model_dir)
      logging.info('Checkpoint path: %s', checkpoint_path)
    checkpoint_step = checkpoint_path.split('_')[-1]

    if model_gin_file is None:
      model_gin_file = config_from_model_dir(model_dir, gin_patterns_to_skip)

    gin_files = [model_gin_file]
    if overwrite_gin_files:
      gin_files.extend(overwrite_gin_files)
      logging.info('Model config will be overridden by the following files:\n'
                   '%s', overwrite_gin_files)
    gin_bindings = (list(overwrite_gin_bindings)
                    if overwrite_gin_bindings else [])
    gin_bindings.extend([
        'PjitPartitioner.num_partitions = 1',
        'MODEL_DIR = "/tmp/t5x"',
        'RETURN_ALL_DECODES = True',
        f'CHECKPOINT_PATH = "{checkpoint_path}"',
        f'STEP_OFFSET = {checkpoint_step}'
    ])
    logging.info('Model config will be overridden by the following bindings'
                 ':\n%s', overwrite_gin_bindings)
    with gin.unlock_config():
      gin_utils.parse_gin_flags(gin_search_paths=gin_search_paths,
                                gin_files=gin_files,
                                gin_bindings=gin_bindings)
    logging.info('Gin Configuration to restore model:\n%s', gin.config_str())

    vocabulary = gin.query_parameter('%VOCABULARY').scoped_configurable_fn()
    try:
      vocab_index_from = gin.query_parameter('%VOCAB_INDEX_FROM')
    except ValueError:
      vocab_index_from = t5_tasks.VOCAB_CC_ALL_100EXTRA_SIZE_WITHOUT_INT
    num_embeddings = gin.query_parameter('%NUM_EMBEDDINGS')

    model = gin.query_parameter('%MODEL').scoped_configurable_fn()
    partitioner = gin.get_configurable(partitioning.PjitPartitioner)()
    checkpoint_config = gin.get_configurable(utils.RestoreCheckpointConfig)(
        path=checkpoint_path)
    task_feature_lengths = gin.query_parameter('%TASK_FEATURE_LENGTHS')
    for k, v in task_feature_lengths.items():
      # If the length is a gin ConfigurableReference, get its value.
      if hasattr(v, 'scoped_configurable_fn'):
        task_feature_lengths[k] = v.scoped_configurable_fn()
    if 'target_inputs' not in task_feature_lengths:
      # Backward compatibility with old model configs without target_inputs.
      task_feature_lengths['target_inputs'] = task_feature_lengths['targets']
    sequence_length = task_feature_lengths
    # Restore model checkpoint.
    logging.info('Restoring model parameters from %s', checkpoint_path)
    train_state, train_state_axes = restore_train_state(
        model,
        batch_size,
        sequence_length,
        partitioner,
        checkpoint_config,
        from_scratch=False)
    logging.info('Model parameters restored.')

    kwargs = dict(
        model=model,
        checkpoint_path=checkpoint_path,
        model_dir=model_dir,
        params=train_state.params,
        train_state_axes=train_state_axes,
        partitioner=partitioner,
        batch_size=batch_size,
        vocab=vocabulary,
        vocab_index_from=vocab_index_from,
        num_embeddings=num_embeddings,
        task_feature_lengths=task_feature_lengths)

    try:
      num_initial_tokens = gin.query_parameter('%NUM_INITIAL_TOKENS')
      kwargs['num_initial_tokens'] = num_initial_tokens
    except ValueError:
      num_initial_tokens = None
      logging.warning('NUM_INITIAL_TOKENS is not found in the model config '
                      'file. Using the default value.')
    return cls(**kwargs)

  @classmethod
  def increase_infer_task_count(cls):
    cnt = cls.infer_task_count
    cls.infer_task_count += 1
    return cnt

  @property
  def model(self) -> models.BaseTransformerModel:
    return self._model

  @property
  def vocab(self) -> seqio.Vocabulary:
    return self._vocab

  @property
  def vocab_index_from(self) -> int:
    return self._vocab_index_from

  @property
  def study_converter(self) -> converters.Converter:
    return self._dataset_builder.study_converter

  @property
  def study_aux_list(self) -> List[Aux]:
    return self._dataset_builder.study_aux_list

  def _make_dataset_builder(self) -> t5_tasks.DatasetFromStudy:
    """Create a dataset builder through gin configuration.

    The gin configuration must include the following configurations:
    t5_tasks.add_tasks, STUDY_CONVERTER.

    Returns:
      t5_tasks.DatasetFromStudy object and the associated study converter.
    """
    cnt = self.increase_infer_task_count()
    add_tasks_fn = gin.get_configurable(t5_tasks.add_tasks)

    study_converter = gin.query_parameter(
        '%STUDY_CONVERTER').scoped_configurable_fn()
    study_converter.set_config(default_study_converter_kwargs(study_converter))

    task_name = f'infer_task_{cnt}'
    dataset_builder = t5_tasks.DatasetFromStudy(
        add_tasks_fn=add_tasks_fn,
        study_converter=study_converter,
        feature_converter_cls=self._model.FEATURE_CONVERTER_CLS,
        task_feature_lengths=self._task_feature_lengths,
        batch_size=self._batch_size,
        task_name=task_name)
    return dataset_builder

  def get_dataset(self,
                  study_list: Sequence[vz.ProblemAndTrials]) -> tf.data.Dataset:
    """Create a dataset from a list of studies."""
    return self._dataset_builder.dataset(study_list)

  def _pad_batch(self, batch: Batch) -> Tuple[Batch, int]:
    """Pad the feature batch size to a multiple of the jax device count."""
    base = jax.device_count()
    num_examples = next(iter(batch.values())).shape[0]
    num_padding = (num_examples + base - 1) // base * base - num_examples
    if num_padding > 0:
      batch = {k: jnp.concatenate(
          [v, jnp.tile(v[-1:], [num_padding] + [1] * (v.ndim-1))])
               for k, v in batch.items()}
    return batch, num_examples

  def _split_batch(self, batch: Batch) -> List[Batch]:
    """Split a large batch to small ones with a size upto self._batch_size."""
    num_examples = next(iter(batch.values())).shape[0]
    if num_examples <= self._batch_size:
      return [batch]
    batches = []
    for i in range(0, num_examples, self._batch_size):
      batches.append({
          k: v[i:min(num_examples, i + self._batch_size)]
          for k, v in batch.items()
      })
    return batches

  def compute_logits_from_batch(
      self,
      batch: Batch,
      restrict_vocab_index: bool = True,
      indices: Optional[List[int]] = None) -> jnp.ndarray:
    """Compute the logits given a batch of features."""
    padded_batch, num_examples = self._pad_batch(batch)
    logits_list = []
    for one_batch in self._split_batch(padded_batch):
      result = self._partitioned_compute_logits_from_batch(
          self._params, one_batch, restrict_vocab_index)
      if indices is not None:
        result = result[:, indices, :]
      logits_list.append(result)

    if len(logits_list) == 1:
      logits = logits_list[0]
    else:
      logits = jnp.concatenate(logits_list, axis=0)
    if num_examples < logits.shape[0]:
      logits = logits[:num_examples]
    return logits

  def _compute_logits_from_batch(
      self,
      params: train_state_lib.FrozenVariableDict,
      batch: Batch,
      restrict_vocab_index: bool = True) -> jnp.ndarray:
    """Wrapper of model._compute_logits."""
    logits = self.model._compute_logits(params, batch)  # pylint: disable=protected-access
    if restrict_vocab_index:
      logits = logits[:, :, self._vocab_index_from: self._vocab_index_to]
    return logits

  def compute_logits(
      self,
      study_list: Sequence[vz.ProblemAndTrials],
      trial_list: Optional[Sequence[vz.TrialSuggestion]] = None,
      restrict_vocab_index: bool = True,
  ) -> Tuple[jnp.ndarray, List[Batch], List[Aux]]:
    """Compute logits, with an optionally restricted vocabulary.

    Args:
      study_list: list of studies.
      trial_list: optional list of trials to append to the input studies.
      restrict_vocab_index: return the logits in the valid token value range.

    Returns:
      Logits array of shape [S, T, V] where S is the number of studies,
        T is the maximum target sequence length and V is the number of discrete
        function values.
      List of feature sequences batches, converted from the list of (appended)
        studies.
      List of study converter auxiliary output dicts, one per study.
    """
    # If trial_list is provided, append the trial to the study trial list.
    if trial_list is not None:
      if len(study_list) != len(trial_list):
        raise ValueError('Length of study_list does not match trial_list.')
      study_list = [self.pad_study(study, len(study.trials) + 1, trial)
                    for study, trial in zip(study_list, trial_list)]

    dataset = self.get_dataset(study_list).as_numpy_iterator()
    logits = []
    batches = []
    for batch in dataset:
      logits.append(self.compute_logits_from_batch(
          batch, restrict_vocab_index=restrict_vocab_index))
      batches.append(batch)
    logits = jnp.concatenate(logits)  # [study number, sequence len, vocab size]
    return logits, batches, self.study_aux_list

  def compute_log_probs_from_batch(
      self,
      batch: Batch,
      restrict_vocab_index: bool = True,
      ) -> jnp.ndarray:
    """Compute log probability, with an optionally restricted vocabulary."""
    logits = self.compute_logits_from_batch(
        batch, restrict_vocab_index=restrict_vocab_index)
    return logits_to_log_probs(logits)

  def compute_log_probs(
      self,
      study_list: Sequence[vz.ProblemAndTrials],
      restrict_vocab_index: bool = True,
  ) -> Tuple[jnp.ndarray, List[Batch], List[Aux]]:
    """Compute log probability, with an optionally restricted vocabulary."""
    logits, batches, aux_list = self.compute_logits(
        study_list, restrict_vocab_index=restrict_vocab_index)
    return logits_to_log_probs(logits), batches, aux_list

  def predict_fun_logits(self,
                         study_list: Sequence[vz.ProblemAndTrials],
                         trial_list: Sequence[vz.TrialSuggestion],
                         restrict_vocab_index: bool = True) -> jnp.ndarray:
    """Predict the function distribution of a trial given a study.

    Args:
      study_list: list of studies containing observations. Assuming all the
        trials are completed.
      trial_list: list of trials to predict the function value given parameters.
      restrict_vocab_index: return the logits in the valid token value range.

    Returns:
      Logits array of shape [S, V] where S is the number of studies, V is the
        number of discrete function values.
    """
    logits_seq, _, _ = self.compute_logits(
        study_list, trial_list, restrict_vocab_index)  # S, T, V.

    # Find the index of the function in the last trial.
    scheme = self.study_converter.trial_token_scheme
    f_indices = []
    for study, aux in zip(study_list, self.study_aux_list):
      f_indices.append(scheme.fun_index_in_trial(
          num_parameters=self.num_parameters(aux),
          trial_index=len(study.trials)))  # Index of the input trial.
    # For every study i, take the logits at index f_indices[i].
    f_logits = jnp.take_along_axis(
        logits_seq, jnp.array(f_indices)[:, None, None], axis=1)[:, 1]  # S, V.
    return f_logits

  def pad_study(
      self,
      study: vz.ProblemAndTrials,
      min_trials: int,
      trial_to_append: Optional[vz.TrialSuggestion] = None
  ) -> vz.ProblemAndTrials:
    """Pad study trials upto min_trials with trial_to_append if provided."""
    # The objective value in the trial_to_append is ignored.
    num_trials_to_append = min_trials - len(study.trials)
    if num_trials_to_append > 0:
      # Append trials.
      if trial_to_append is None:
        trial_to_append = self._dummy_trial(study)
      else:
        trial_to_append = self._set_dummy_fun_value(study, trial_to_append)
      study = copy.deepcopy(study)
      study.trials.extend([trial_to_append] * num_trials_to_append)
    return study

  def _dummy_trial(self, study: vz.ProblemAndTrials) -> vz.Trial:
    """Make a trial with dummy parameter and function values."""
    trial = vz.Trial()
    for pc in study.problem.search_space.parameters:
      if pc.type in [vz.ParameterType.CATEGORICAL, vz.ParameterType.DISCRETE]:
        value = pc.feasible_values[0]
      else:
        value = pc.bounds[0]  # Minimum value.
      trial.parameters[pc.name] = vz.ParameterValue(value=value)
    trial = self._set_dummy_fun_value(study, trial)
    return trial

  def _set_dummy_fun_value(self, study: vz.ProblemAndTrials,
                           trial: vz.TrialSuggestion) -> vz.Trial:
    """Set a dummy function value that does not affect normalization."""
    if len(study.problem.metric_information) != 1:
      raise ValueError('Study contains zero or multiple metric information.')
    metric_name = study.problem.metric_information.item().name
    if study.trials:
      ref_trial = study.trials[0]
      value = ref_trial.final_measurement.metrics[metric_name].value
    else:
      value = 1.0
    trial = trial.to_trial()
    trial.complete(vz.Measurement({metric_name: value}))
    return trial

  def _model_predict_fn(
      self,
      params: train_state_lib.FrozenVariableDict,
      batch: Batch,
      rng: jax.random.KeyArray,
      decode_until: Optional[int] = None,
      decoder_params: Optional[MutableMapping[str, Any]] = None,
      num_decodes: int = 1) -> Tuple[jnp.ndarray, Batch]:
    """Wrapper of model.predict_batch_with_aux to draw samples."""
    if decoder_params is None:
      decoder_params = dict()
    decoder_params = dict(decoder_params, decode_until=decode_until)

    # Vizier models' predict_batch_with_aux method has a different signature.
    model = typing.cast(vizier_models.VizierEncoderDecoderModel, self._model)
    return model.predict_batch_with_aux(
        params, batch,
        rng=rng,
        decoder_params=decoder_params,
        return_all_decodes=True,
        num_decodes=num_decodes,
        prompt_with_targets=True)

  def _setup_for_multistep_sampling(
      self,
      batch: Batch,
      num_trials: int,
      sample_last_function: bool,
      input_start_index: int,
  ) -> Tuple[MutableMapping[str, jnp.ndarray], Mapping[str, Union[List[int],
                                                                  int]]]:
    """Sets up batch for decoding for multiple timesteps.

    This function prepares a mask for logits that will be used for multi-step
    decoding. The mask index of a token is such that corresponding token in the
    decoder input tokens is off by - 1. It also updates the decoder input tokens
    to take the appropriate prompt for multi-trial decoding

    Args:
      batch: Mapping[str, jnp.ndarray] consisting of arrays corresponding to a
        batch of data to run through the model. Assumes that tasks in a batch
        have the same trial length.
      num_trials: The number of trials to decode the model for.
      sample_last_function: whether to sample the function value of the last
        trial.
      input_start_index: What index to start decoding the model from.

    Returns:
      batch: Input mapping is updated with logits for masking out invalid
        indices for specific parameters and modified batches for prompting the
        model.
      metadata: Mapping containing relevant information about function and trial
        start indices as well as the number of parameters of the task.
    """
    # Make the dict mutable
    batch = dict(batch)

    # Setup the special tokens
    special_tokens = self.vocab.encode('a*c|d')
    special_tokens = [special_tokens[1], special_tokens[3]]

    # Setup for decoding for multiple timesteps
    batch_size, seq_length = batch['decoder_input_tokens'].shape
    aux_batch = self.study_aux_list[-batch_size:]
    num_parameters = np.array([self.num_parameters(aux) for aux in aux_batch],
                              dtype=np.int32)
    max_num_parameters = np.max(num_parameters)

    if not jnp.all(num_parameters == max_num_parameters):
      raise ValueError('All tasks within a batch must have the'
                       ' same number of parameters per trial')

    fn_prediction_indices = []
    trial_start_indices = []

    # Setup the logits mask and the decoder input tokens for decoding for
    # multiple timeteps.
    logits_mask = np.full((batch_size, seq_length, self._num_embeddings), 0.0)
    # Get numpy version of decoder_input_tokens - easier to manipulate in-place.
    np_decoder_inputs = np.array(batch['decoder_input_tokens'])

    # Cannote decode past the last index of the beginning of the last trial.
    # A trial consists of max_num_parameters, 2 special tokens *,|
    # and the function prediction.
    scheme = self.study_converter.trial_token_scheme
    trial_length = scheme.trial_length(max_num_parameters)
    sampling_length = input_start_index + (num_trials * trial_length)
    if sampling_length > seq_length:
      raise ValueError('We cannot decode past the maximum sequence length.')

    for trial in range(num_trials):
      offset = trial*trial_length
      this_start_index = input_start_index + offset
      for p_ind in range(max_num_parameters):
        for i in range(batch_size):  # Example index.
          aux = aux_batch[i]
          j = min(p_ind, num_parameters[i]-1)  # Parameter index.
          p_config = list(aux['parameter_name_to_configs'].values())[j]

          # Find the value range.
          if p_config.type in [
              vz.ParameterType.CATEGORICAL, vz.ParameterType.DISCRETE
          ]:
            num_values = len(p_config.feasible_values)
          else:
            num_values = self.study_converter.num_quantized_values

          # First fill everything with the negative infinity value.
          # -1 because we want to mask the logits before generating the param.
          param_idx = this_start_index + p_ind - 1
          logits_mask[i, param_idx, :] = decoding.NEG_INF
          # Selectively set the appropriate values to 0.0
          logits_mask[i, param_idx,
                      self._vocab_index_from:(self._vocab_index_from +
                                              num_values)] = 0.0
      # setup for decoder_input_params and logits mask to do multi-trial
      # decoding.
      func_sep_idx = this_start_index + max_num_parameters
      logits_mask[:, func_sep_idx, :] = decoding.NEG_INF
      logits_mask[:, func_sep_idx, self._vocab_index_from:(
          self._vocab_index_from +
          self.study_converter.num_quantized_values)] = 0.0

      # Setup separator tokens as input prompts
      np_decoder_inputs[:, func_sep_idx] = special_tokens[0]
      np_decoder_inputs[:, func_sep_idx + 2] = special_tokens[1]

      # fn_prediction_indices contains indices in the decoder_input_tokens.
      # This is why we have a + 1. Mask indices are off by 1 to account for the
      # fact that we mask before we decode.
      fn_prediction_indices.append(func_sep_idx + 1)
      trial_start_indices.append(this_start_index)

    # We decode until the final trial separator token if sample_last_function,
    # otherwise, the last parameter dimension. Last index not inclusive.
    if sample_last_function:
      decode_until = func_sep_idx + 2
    else:
      decode_until = func_sep_idx
    # convert from numpy array to jax array
    logits_mask = jnp.asarray(logits_mask[:, :decode_until, :])

    # Batch is modified in-place with updated values for sampling.
    batch['decoder_input_tokens'] = jnp.asarray(
        np_decoder_inputs, dtype=batch['decoder_input_tokens'].dtype)
    batch['logits_mask'] = logits_mask

    metadata = {
        'fn_pred_indices': fn_prediction_indices,
        'trial_start_indices': trial_start_indices,
        'decode_until': decode_until,
        'max_num_parameters': max_num_parameters,
    }

    return batch, metadata

  def _multistep_sampling(
      self,
      batch: Batch,
      input_start_indices: np.ndarray,
      decoder_params: Optional[MutableMapping[str, Any]] = None,
      num_decodes: int = 1,
      num_trials: int = 1,
      sample_last_function: bool = False,
      prompt_end_indices: Optional[np.ndarray] = None,
  ) -> Tuple[List[jnp.ndarray], Batch, Mapping[str, Union[List[int], int]]]:
    """Sample parameter and function values for multiple trials.

    It samples the specified number of trials except for the function value at
    the last trial if sample_last_function is False. So if num_trials = 1, it
    only samples the parameters of the next trial. If num_trials = 3, it samples
    the next two full trials and the parameters of the third trial.

    The returned sample sequenth length is trial_lens * num_trials - 1 if
    sample_last_function is True (last separator token dropped), otherwise
    trial_lens * (num_trials - 1) + num_parameters

    Args:
      batch: input mapping to feed to the model.
      input_start_indices: index at which initial prompt to model ends and new
        trial sequence begins.
      decoder_params: additional (model-independent) parameters for the decoder.
      num_decodes: the number of rollouts to perform for each candidate.
      num_trials: number of timesteps to rollout the model for.
      sample_last_function: whether to sample the function value of the last
        trial.
      prompt_end_indices: index where the prompt ends. It should be
        not less than input_start_indices if given.

    Returns:
      List of sample arrays with the length of the input batch size. Each array
        has the shape of (num_decodes, sample_seq_length).
      Feature batch where each feature has the shape of
        (batch_size * num_decodes, seq_length, feature_dimension) and
        decoder_input_tokens is updated with samples.
      metadata mapping containing relevant information about function and trial
        start indices as well as the number of parameters of the task.
    """
    # Verify shapes.
    _verify_indices_shapes(batch, input_start_indices)

    # Make a new dict to avoid modifying the original one.
    batch = {k: jnp.asarray(v) for k, v in batch.items()}
    # Mask out everything except the prompt.
    batch_size, seq_length = batch['decoder_input_tokens'].shape

    # prompt_end_indices is useful for specifying when part of the prompt is
    # beyond input_start_indices.
    if prompt_end_indices is None:
      prompt_end_indices = input_start_indices
    mask = jnp.arange(seq_length) < prompt_end_indices[:, None]
    batch['decoder_input_tokens'] = batch['decoder_input_tokens'] * mask

    all_start_idxs_eq = jnp.all(input_start_indices == input_start_indices[0])
    if not all_start_idxs_eq:
      raise ValueError('Batch must have all instance start indices be equal.')

    # Setup to perform multi-step decoding
    batch, metadata = self._setup_for_multistep_sampling(
        batch, num_trials, sample_last_function, input_start_indices[0])

    # Setup params for decoding algorithm
    decoder_params = {} if decoder_params is None else decoder_params
    decoder_params['max_decode_len'] = seq_length - 1

    rng, self._rng = random.split(self._rng)
    decoder_params = FrozenDict(decoder_params)
    padded_batch, num_examples = self._pad_batch(batch)

    # For temperature sampling, we can split the batch across multiple devices
    # and still get a unique number of decodes. For beam_search however, since
    # it is deterministic, splitting across devices will return the same samples
    device_count = jax.device_count()
    original_num_decodes = num_decodes
    split_across_devices = batch_size == 1
    if split_across_devices:
      # Making modifications to take device count into consideration
      num_decodes = math.ceil(num_decodes / device_count)

    # Run the decoding algorithm and get samples
    full_samples, _ = self._partitioned_model_predict_fn(
        self._params, padded_batch, rng, metadata['decode_until'],
        decoder_params, num_decodes)

    del batch['logits_mask']
    # Reset num decodes
    num_decodes = original_num_decodes
    if split_across_devices:
      # Samples from all devices are decodes of the first example because
      # batch_size == 1.
      full_samples = jnp.reshape(full_samples, (1, -1, full_samples.shape[-1]))
      full_samples = full_samples[:, :num_decodes, :]
    else:
      # We padded the batch dimension so take the right subset.
      full_samples = full_samples[:num_examples, :, :]

    # Expand batch for returning decodes.
    batch = {k: jnp.repeat(v, num_decodes, axis=0) for k, v in batch.items()}

    # Sampled (target) sequence is shifted to the left by 1.
    sample_start_indices = input_start_indices - 1

    # We update 'decoder_input_tokens' in this block
    samples = []
    for i, start_index in enumerate(sample_start_indices):
      this_sample = full_samples[i, :,
                                 start_index:(metadata['decode_until'] - 1)]
      batch['decoder_input_tokens'] = batch['decoder_input_tokens'].at[
          i*num_decodes:(i+1)*num_decodes,
          (start_index+1):metadata['decode_until']].set(this_sample)
      samples.append(this_sample)
    return samples, batch, metadata

  def sample_parameters_of_next_trials(
      self,
      study_list: Sequence[vz.ProblemAndTrials],
      num_trials: int = 1,
      num_samples: int = 1,
      decoder_params: Optional[MutableMapping[str, Any]] = None,
  ) -> Tuple[List[jnp.ndarray], List[Optional[jnp.ndarray]], List[Mapping[
      str, jnp.ndarray]], List[Aux]]:
    """Sample parameters of the next trial given a study."""
    trial_indices = [len(study.trials) for study in study_list]
    return self.sample_parameters_in_trials(
        study_list=study_list,
        trial_indices=trial_indices,
        num_trials=num_trials,
        decoder_params=decoder_params,
        num_decodes=num_samples)

  def sample_parameters_in_trials(
      self,
      study_list: Sequence[vz.ProblemAndTrials],
      trial_indices: Sequence[int],
      num_trials: int = 1,
      decoder_params: Optional[MutableMapping[str, Any]] = None,
      num_decodes: int = 1,
  ) -> Tuple[List[jnp.ndarray], List[Optional[jnp.ndarray]], List[Mapping[
      str, jnp.ndarray]], List[Aux]]:
    """Sample parameter prediction at a given trial index.

    For an example below ("I" is the initial tokens in the target string):
            decoder_target_tokens: I a b * c | d e * f |  ...
             decoder_input_tokens: 0 I a b * c | d e * f  ...
                   input position: 0 1 2 3 4 5 6 7 8 9 10 ...
    To sample the 2nd trial (trial index = 1),
    the parameter index range in the input sequence: [7, 9).
    We return the samples in the target sequence at position 6 and 7.

    Args:
      study_list: study list.
      trial_indices: list of 0-based trial indices to sample the parameters at.
      num_trials: number of trials to sample.
      decoder_params: additional (model-independent) parameters for the decoder.
      num_decodes: the number of beams to use in beam search.

    Returns:
      List of parameter samples of length num_studies, each with shape of
        (num_decodes, num_trials, num_params).
      List of function samples of length num_studies if num_trials > 1,
        otherwise list of None. If num_trials > 1, each has a shape of
        (num_decodes, num_trials - 1).
      List of feature sequences batches, converted from the list of (padded)
        studies.
      List of study converter auxiliary outputs, one per study.
    """
    if not isinstance(self._model, vizier_models.VizierEncoderDecoderModel):
      raise NotImplementedError('Indexing parameters only supports '
                                'VizierEncoderDecoderModel.')

    # Make sure every study has at least `trial` trials with final measurements.
    if len(study_list) != len(trial_indices):
      raise ValueError('Length of study_list must match trial_indices.')
    study_list = copy.deepcopy(study_list)
    for i, (study, trial_index) in enumerate(zip(study_list, trial_indices)):
      if len(study.trials) < trial_index:
        raise ValueError(f'Cannot sample {trial_index}-th when the study has '
                         f'only {len(study.trials)} trials.')
      study_list[i] = self.pad_study(study, min_trials=trial_index+num_trials)

    # Feed the study list to the dataset and process by batches.
    dataset = self.get_dataset(study_list).as_numpy_iterator()
    scheme = self.study_converter.trial_token_scheme
    study_idx = 0
    samples = []
    batches = []
    num_params_list = []
    for batch in dataset:
      batch_size = batch['decoder_input_tokens'].shape[0]
      aux_batch = self.study_aux_list[-batch_size:]

      # Find the starting and ending index in the decoder input in a study.
      input_start_indices = np.zeros(batch_size, dtype=np.int32)
      for i in range(batch_size):
        num_params = self.num_parameters(aux_batch[i])
        num_params_list.append(num_params)
        trial_index = trial_indices[study_idx]
        study_idx += 1
        max_trials = self.max_trials(num_params)
        if trial_index >= max_trials:
          raise ValueError(f'The model supports sampling upto {max_trials-1}-th'
                           f' trial with {num_params} parameters, but a trial '
                           f'index of {trial_index} is given.')

        index_range = scheme.param_index_range_in_trial(num_params, trial_index)
        # Input sequence is shifted to the right by 1 token.
        input_start_indices[i] = index_range[0] + 1

      # Sampling the batch.
      sample, batch, _ = self._multistep_sampling(
          batch=batch,
          input_start_indices=input_start_indices,
          decoder_params=decoder_params,
          num_decodes=num_decodes,
          num_trials=num_trials,
          sample_last_function=False)

      samples.extend(sample)
      batches.append(batch)

    param_samples, fun_samples = self._split_parameter_and_function_samples(
        samples, num_params_list, num_trials)
    return param_samples, fun_samples, batches, self.study_aux_list

  def _split_parameter_and_function_samples(
      self,
      samples_list: List[jnp.ndarray],
      num_params_list: List[int],
      num_trials: int) -> Tuple[List[jnp.ndarray], List[Optional[jnp.ndarray]]]:
    """Split trial samples into parameter and function samples."""
    if len(samples_list) != len(num_params_list):
      raise ValueError('The length of samples_list must match the length of '
                       'num_params_list')
    scheme = self.study_converter.trial_token_scheme
    if num_trials == 1:
      # Samples are the single trial of parameters. No function samples.
      param_samples_list = [
          jnp.expand_dims(samples, -2) for samples in samples_list
      ]
      fun_samples_list = [None] * len(samples_list)
    else:
      param_samples_list = []
      fun_samples_list = []
      for samples, num_params in zip(samples_list, num_params_list):
        # Calculate the index of parameter and function offset.
        f_idx_trial_0 = scheme.fun_index_in_trial(num_params, 0)
        p_range_trial_0 = scheme.param_index_range_in_trial(num_params, 0)
        f_offset = f_idx_trial_0 - p_range_trial_0[0]
        trial_len = scheme.trial_length(num_params)

        # Let L be the trial length and P be the number of parameters.
        # [[0, 1,   ..., P-1],
        #  [L, L+1, ..., L+P-1],
        #  ...]
        p_indices = (jnp.arange(num_params)[None, :] +
                     jnp.arange(num_trials)[:, None] * trial_len)
        # Shape: ([num_decodes,] num_trials, num_params)
        param_samples_list.append(samples[..., p_indices])

        # [f_offset, L+f_offset, ..., ]
        f_indices = jnp.arange(num_trials-1) * trial_len + f_offset
        # Shape: ([num_decodes,] num_trials-1)
        fun_samples_list.append(samples[..., f_indices])

      if samples_list[0].ndim == 1:
        assert all([p.shape == (num_trials, num_p)
                    for p, num_p in zip(param_samples_list, num_params_list)])
        assert all([f.shape == (num_trials-1,)
                    for f in fun_samples_list])
      else:
        assert samples_list[0].ndim == 2
        num_decodes = samples_list[0].shape[0]
        assert all([p.shape == (num_decodes, num_trials, num_p)
                    for p, num_p in zip(param_samples_list, num_params_list)])
        assert all([f.shape == (num_decodes, num_trials-1)
                    for f in fun_samples_list])
    return param_samples_list, fun_samples_list

  def num_parameters(self, aux: Aux) -> int:
    """Number of the nonfixed parameters.

    This is also the number of parameter tokens of a trial in the target
    sequence. It could be different from the number of parameters in a study
    when study_converter._filter_fixed_parameters is True and there exist
    parameters with fixed values.

    Args:
      aux: study converter auxiliary output dict.

    Returns:
      Number of the non-fixed parameters.
    """
    return len(aux['parameter_name_to_configs'])

  def max_trials(self, num_parameters: Optional[int] = None,
                 aux: Optional[Aux] = None) -> int:
    """Compute the supported maximum number of trials to infer for a study."""
    if num_parameters is None:
      num_parameters = self.num_parameters(aux)
    scheme = self.study_converter.trial_token_scheme
    target_len = self._task_feature_lengths['targets']
    return scheme.max_trials(num_parameters, target_len)

  def fun_index_in_trial(self, trial_index: int,
                         num_parameters: Optional[int] = None,
                         aux: Optional[Aux] = None) -> int:
    """Index of the function token in a trial."""
    scheme = self.study_converter.trial_token_scheme
    if num_parameters is None:
      num_parameters = self.num_parameters(aux)
    return scheme.fun_index_in_trial(num_parameters, trial_index)

  def set_seed(self, seed: int):
    self._rng = random.PRNGKey(seed)
