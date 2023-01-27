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

"""Wrap a trained model into an OSS Vizier Designer."""

import copy
import enum
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union
from absl import logging

from jax import random
import jax.numpy as jnp
import ml_collections
import numpy as np
from optformer.data import converters
from optformer.t5x import inference_utils
from vizier import algorithms as vza
from vizier import pyvizier as vz

# Default arguments to create an inference_model for the Transformer designer.
BBOB_INFERENCE_MODEL_KWARGS = {
    'checkpoint_path_or_model_dir': 'gs://gresearch/optformer/model_checkpoints/bbob/checkpoint_700000',
    'model_gin_file': 'optformer/t5x/configs/tasks/bbob.gin',
    'batch_size': 1,
}
HPOB_INFERENCE_MODEL_KWARGS = {
    'checkpoint_path_or_model_dir': 'gs://gresearch/optformer/model_checkpoints/hpob/checkpoint_400000',
    'model_gin_file': 'optformer/t5x/configs/tasks/hpob.gin',
    'batch_size': 1,
}
DEFAULT_DESIGNER_NAME = 'designer_recursive_gp'
DEFAULT_RANKING_CONFIG = {
    'type': 'ei',
    'function_temperature': 1.0,
}

ConfigDict = Union[Dict[str, Any], ml_collections.ConfigDict]
Study = converters.Study


@enum.unique
class PolicyType(str, enum.Enum):
  PRIOR = 'prior'  # Sample from the prior policy.
  RANK_SAMPLES = 'rank_samples'  # Rank prior samples with acquisition.


def _maybe_to_py(x: Optional[jnp.ndarray]) -> Optional[np.ndarray]:
  return None if x is None else np.asarray(x)


def _update_with_default(config: Optional[ConfigDict],
                         default: Dict[str, Any]) -> Dict[str, Any]:
  if config is None:
    config = {}
  elif isinstance(config, ml_collections.ConfigDict):
    config = config.to_dict()
  return default | config


def has_single_minimize_goal(study_config: vz.ProblemStatement) -> bool:
  if not (study_config.is_single_objective and
          len(study_config.metric_information) == 1):
    raise ValueError('Study config contains multiple objectives or has '
                     'constraints.')
  mi = study_config.metric_information.item()
  return mi.goal.is_minimize


def flip_metric_values(measurement: vz.Measurement):
  for m in measurement.metrics:
    measurement.metrics[m] = -measurement.metrics[m].value


def flip_trial_metric_values(trial: vz.Trial):
  """Flip corresponding trial metrics' signs in place."""
  if trial.final_measurement:
    flip_metric_values(trial.final_measurement)
  for measurement in trial.measurements:
    flip_metric_values(measurement)


class OptFormerDesigner(vza.Designer):
  """Wraps a trained OptFormer model into a designer."""

  def __init__(
      self,
      study_config: vz.ProblemStatement,
      inference_model: inference_utils.InferenceModel,
      designer_name: Optional[str] = DEFAULT_DESIGNER_NAME,
      temperature: float = 1.0,
      policy_type: Union[PolicyType, str] = PolicyType.PRIOR,
      num_samples: int = 128,
      ranking_config: Optional[ConfigDict] = None,
      study_converter_config: Optional[ConfigDict] = None,
  ):
    """Create an OptFormer policy.

    Args:
      study_config: experimenter study config.
      inference_model: transformer inference model.
      designer_name: name of algorithm name to imitate. If None or empty, use
        the designer_name field in study_config.
      temperature: policy output temperature.
      policy_type: policy type defined in PolicyType enum.
      num_samples: number of parameter suggestions to sample before ranking.
      ranking_config: ranking config dict.
      study_converter_config: override study converter if provided.
    """
    self._study_config = copy.deepcopy(study_config)

    self._metric_flipped = False
    if has_single_minimize_goal(self._study_config):
      self._study_config.metric_information.item().flip_goal()
      self._metric_flipped = True

    if designer_name:
      self._study_config.metadata[
          converters.METADATA_ALGORITHM_KEY] = designer_name
    self._historical_study: converters.Study = Study(
        problem=self._study_config, trials=[])

    self._decoder_params = {'temperature': temperature}
    self._inference_model = inference_model
    self._vocab = inference_model.vocab
    self._policy_type = PolicyType(policy_type)
    self._num_samples = num_samples
    self._ranking_config = _update_with_default(ranking_config,
                                                DEFAULT_RANKING_CONFIG)

    if (self._policy_type == PolicyType.RANK_SAMPLES and
        self._ranking_config['type'] == 'thompson_sampling'):
      self._num_trials = self._ranking_config.get('num_trials', 1)
    self._num_trials = 1

    self._study_converter = self._inference_model.study_converter
    config = dict(
        inference_utils.default_study_converter_kwargs(self._study_converter))
    if study_converter_config:
      config.update(study_converter_config)
    self._study_converter.set_config(config)

  def suggest(self,
              count: Optional[int] = None) -> Sequence[vz.TrialSuggestion]:
    start_time = time.time()
    logging.info('Suggesting %d Trials with %d completed.', count,
                 len(self._historical_study.trials))
    if self._policy_type == PolicyType.PRIOR:
      samples = self.suggest_samples(count)
    elif self._policy_type == PolicyType.RANK_SAMPLES:
      samples = self.suggest_with_rank(count)
    else:
      raise ValueError(f'Unsupported policy type: {self._policy_type}')
    end_time = time.time()
    logging.info('Time it took to produce %d suggestions is: %f seconds.',
                 count, end_time - start_time)
    return samples

  def _sample(
      self, num_samples: int
  ) -> Tuple[np.ndarray, Optional[np.ndarray], Mapping[str, jnp.ndarray],
             converters.Aux]:
    """Return the samples of shape [num_samples, num_parameters] and batch.

    Args:
      num_samples: number of samples to return.

    Returns:
      A (num_samples, num_trials, num_params) array of parameter samples.
      A (num_samples, num_trials-1) array of function samples or None.
      The corresponding dict of feature sequences for the transformer model.
      Study converter auxiliary output dict.
    """
    study = Study(
        problem=self._historical_study.problem,
        trials=self._historical_study.trials)

    param_samples_list, fun_samples_list, batch_list, aux_list = (
        self._inference_model.sample_parameters_of_next_trials(
            study_list=[study],
            num_samples=num_samples,
            num_trials=self._num_trials,
            decoder_params=self._decoder_params))
    aux = aux_list[0]
    trial_ids = aux.get('trial_ids') or aux['trial_permutation']['trial_ids']
    if len(trial_ids) - self._num_trials != len(study.trials):
      raise ValueError('The number of converted trials - num_sampling_trials '
                       f'({len(trial_ids)} - {self._num_trials}) '
                       f'does not match the input study ({len(study.trials)}). '
                       'Some trials are not valid.')
    # Only one study in the output list.
    return (np.asarray(param_samples_list[0]),
            _maybe_to_py(fun_samples_list[0]), batch_list[0], aux_list[0])

  def suggest_samples(self, count: int = 1) -> Sequence[vz.TrialSuggestion]:
    if count != 1:
      raise NotImplementedError('Suggest function only supports count = 1.')
    samples, _, _, aux = self._sample(1)  # [1, 1, num_parameters]
    return [self.token_sample_to_trial(samples[0, 0], aux)]

  def function_logits(self, batch: Mapping[str, jnp.ndarray], trial_idx: int,
                      aux: converters.Aux) -> jnp.ndarray:
    """Use feature batch to predict the function value of the samples."""
    seq_logits = self._inference_model.compute_logits_from_batch(
        batch, restrict_vocab_index=True)  # [S, T, V].

    f_token_idx = self._inference_model.fun_index_in_trial(trial_idx, aux=aux)
    logits = seq_logits[:, f_token_idx]  # [S, V]

    logits /= self._ranking_config['function_temperature']
    return logits

  def token_sample_to_trial(self, samples: np.ndarray,
                            aux: converters.Aux) -> vz.TrialSuggestion:
    """Decodes samples into raw parameter values and make a trial."""
    if samples.ndim != 1:
      raise ValueError('Token samples should be a 1D array.')

    parameter_texts = [
        self._vocab.decode_tf([token]).numpy().decode() for token in samples
    ]

    trial = self._study_converter.parameter_texts_to_trial(
        aux=aux, parameter_texts=parameter_texts)
    return trial

  def suggest_with_rank(self, count: int = 1) -> Sequence[vz.TrialSuggestion]:
    # Notation of the variable dimensions:
    #   S: number of samples.
    #   T: number of trials.
    #   P: number of parameters.
    #   V: number of the function value discretization levels.
    if count != 1:
      raise NotImplementedError('Suggest function only supports count = 1.')
    trial_idx = len(self._historical_study.trials)

    # param_samples: [S, T, P]
    # fun_params: [S, T-1] if T > 1 else None
    param_samples, fun_samples, sample_feat_batch, aux = (
        self._sample(self._num_samples))
    # Convert function token index [0, V-1] to quantized integer in [0, Q-1].
    if fun_samples is not None:
      fun_samples = fun_samples - self._inference_model.vocab_index_from

    # Predict the function value of the samples.
    # [S, V].
    logits = self.function_logits(sample_feat_batch, trial_idx, aux)

    if 'type' not in self._ranking_config:
      raise ValueError('ranking_config must include a key of "type" when '
                       'policy_type = rank_samples.')
    ranking_type = self._ranking_config['type']
    if ranking_type == 'expected_mean':
      # Select the sample with the highest mean.
      scores = self._expected_mean(logits)
    elif ranking_type == 'ucb':
      scores = self._ucb(logits)
    elif ranking_type == 'thompson_sampling':
      scores = self._thompson_sampling(fun_samples, logits, aux)
    elif ranking_type == 'ei':
      scores = self._ei(logits, sample_feat_batch, aux)
    elif ranking_type == 'pi':
      scores = self._pi(logits, sample_feat_batch, aux)
    else:
      raise ValueError('Unknown ranking type: "{ranking_type}".')

    selected_sample_idx = np.argmax(scores)
    selected_sample = param_samples[selected_sample_idx, 0]

    # Convert a vector of parameter values to a Vizier trial.
    trial = self.token_sample_to_trial(selected_sample, aux)
    return [trial]

  def _ucb(
      self,
      logits: jnp.ndarray  # [S, V].
  ) -> np.ndarray:
    """Compute the p-quantile of discrete function distributions."""
    if logits.ndim != 2:
      raise ValueError('Input logits must be a 2D array.')
    p = self._ranking_config.get('upper_bound')
    if p is None or not 0 <= p <= 1:
      raise ValueError(
          'ranking_config must include a key of "upper_bound" in [0, 1] when '
          'ranking_config["type"] = "ucb".')
    log_probs = inference_utils.logits_to_log_probs(logits)
    cdf = jnp.cumsum(jnp.exp(log_probs), 1)
    quantile = (cdf < p).sum(1)  # [S]
    return np.asarray(quantile)

  def _best_y(self, feature_batch: Mapping[str, jnp.ndarray],
              aux: converters.Aux) -> int:
    """Extract the best quantized y value from sample feature batch dict."""
    num_trials = len(self._historical_study.trials)
    if num_trials == 0:
      return 0

    f_indices = [
        self._inference_model.fun_index_in_trial(trial_index=i, aux=aux)
        for i in range(num_trials)
    ]
    # Shape of decoder_target_tokens: [S, seq_length]
    # All samples share the same history function values.
    fs = feature_batch['decoder_target_tokens'][0, f_indices]  # [#trials]

    # Convert from token index [0, V-1] to quantized integer in [0, Q-1]
    fs = fs - self._inference_model.vocab_index_from
    return int(jnp.max(fs))

  def _pi(
      self,
      logits: jnp.ndarray,  # [S, V].
      feature_batch: Mapping[str, jnp.ndarray],
      aux: converters.Aux) -> np.ndarray:
    """Compute the expected improvement."""
    if logits.ndim != 2:
      raise ValueError('Input logits must be a 2D array.')
    best_y = self._best_y(feature_batch, aux)  # Quantized best y value.
    log_probs = inference_utils.logits_to_log_probs(logits)
    pis = jnp.exp(log_probs[:, best_y + 1:]).sum(1)
    return np.asarray(pis)  # [S]

  def _ei(
      self,
      logits: jnp.ndarray,  # [S, V].
      feature_batch: Mapping[str, jnp.ndarray],
      aux: converters.Aux) -> np.ndarray:
    """Compute the expected improvement."""
    if logits.ndim != 2:
      raise ValueError('Input logits must be a 2D array.')
    best_y = self._best_y(feature_batch, aux)  # Quantized best y value.
    if best_y == logits.shape[1] - 1:
      # Highest y value is already observed.
      return np.zeros(logits.shape[0])

    # Number of bins with higher value than best_y: Q - best_y - 1.
    imp_bins = logits.shape[1] - best_y - 1

    # y - best_y, forany y in [1, imp_bins].
    imp_y_range = jnp.arange(1, imp_bins + 1)  # [1, ..., imp_bins].

    # log probabilities of improved y bins.
    log_probs = inference_utils.logits_to_log_probs(logits)
    imp_log_probs = log_probs[:, best_y + 1:]  # [S, imp_bins].

    eis = (jnp.exp(imp_log_probs) * imp_y_range[None, :]).sum(1)
    return np.asarray(eis)  # [S]

  def _expected_mean(
      self,
      logits: jnp.ndarray  # [S, V].
  ) -> np.ndarray:
    """Compute the expected mean of discrete function distributions."""
    if logits.ndim != 2:
      raise ValueError('Input logits must be a 2D array.')
    y_range = jnp.arange(logits.shape[1])
    log_probs = inference_utils.logits_to_log_probs(logits)
    mean = (jnp.exp(log_probs) * y_range[None, :]).sum(1)
    return np.asarray(mean)  # [S]

  def _thompson_sampling(
      self,
      fun_samples: Optional[np.ndarray],  # [S, T-1] tokens or None
      logits: jnp.ndarray,  # [S, V].
      aux: converters.Aux,
  ) -> np.ndarray:
    # Sample the last function value.
    rng, _ = random.split(self._inference_model._rng)  # pylint:disable=protected-access
    last_funs = np.asarray(random.categorical(rng,
                                              logits).astype(jnp.int32))  # [S]
    if self._num_trials > 1:
      fun_samples = np.concatenate([fun_samples, last_funs[:, None]], axis=1)
    else:
      fun_samples = last_funs[:, None]
    max_funs = fun_samples.max(1)  # [S]
    return max_funs

  def update(self, delta: vza.CompletedTrials) -> None:
    completed_trials = []
    for trial in delta.completed:
      # A completed trial either has a final_measurement or is marked as
      # trial_infeasible.
      copied_trial = copy.deepcopy(trial)
      if self._metric_flipped:
        flip_trial_metric_values(copied_trial)
      completed_trials.append(copied_trial)
    self._historical_study.trials.extend(completed_trials)
