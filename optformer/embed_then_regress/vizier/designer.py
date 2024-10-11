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

"""Optimizes ICL regressor with Eagle. Forked from GP-Bandit."""

import copy
import random
from typing import Sequence

import attrs
import jax
import numpy as np
from optformer.embed_then_regress import regressor as regressor_lib
from optformer.embed_then_regress.vizier import serializers
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import quasi_random
from vizier._src.algorithms.designers.gp import acquisitions as acq_lib
from vizier._src.algorithms.optimizers import eagle_strategy as es
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier._src.jax import types
from vizier.pyvizier import converters
from vizier.pyvizier.converters import padding

default_optimizer_factory = vb.VectorizedOptimizerFactory(
    strategy_factory=es.VectorizedEagleStrategyFactory(
        eagle_config=es.EagleStrategyConfig()
    ),
    max_evaluations=1000,
    suggestion_batch_size=25,
)

default_scoring_function_factory = acq_lib.bayesian_scoring_function_factory(
    lambda _: acq_lib.UCB()
)


@attrs.define
class TransformerICLOptDesigner(vza.Designer):
  """Guides evolutionary search using Embed-then-Regress."""

  problem: vz.ProblemStatement = attrs.field()
  regressor: regressor_lib.StatefulICLRegressor = attrs.field()

  _num_seed_trials: int = attrs.field(default=1, kw_only=True)
  _optimizer_factory: vb.VectorizedOptimizerFactory = attrs.field(
      default=default_optimizer_factory, kw_only=True
  )
  _acq_fn: acq_lib.AcquisitionFunction = attrs.field(
      default=acq_lib.UCB(), kw_only=True
  )
  x_serializer: serializers.SuggestionSerializer = attrs.field(
      factory=serializers.XSerializer, kw_only=True
  )

  _rng: jax.Array = attrs.field(
      factory=lambda: jax.random.PRNGKey(random.getrandbits(32)), kw_only=True
  )
  _padding_schedule: padding.PaddingSchedule = attrs.field(
      factory=padding.PaddingSchedule, kw_only=True
  )

  # ------------------------------------------------------------------
  # Internal attributes which should not be set by callers.
  # ------------------------------------------------------------------
  _history: list[vz.Trial] = attrs.field(factory=list, init=False)

  def __attrs_post_init__(self):
    self._converter = converters.TrialToModelInputConverter.from_problem(
        self.problem,
        scale=True,
        max_discrete_indices=0,
        flip_sign_for_minimization_metrics=True,
    )
    self._optimizer = self._optimizer_factory(self._converter)
    self._quasi_random_sampler = quasi_random.QuasiRandomDesigner(
        self.problem.search_space,
        seed=int(jax.random.randint(self._rng, [], 0, 2**16)),
    )
    self.regressor.reset()
    self.regressor.set_metadata('')  # TODO: Use problem metadata.

  def _generate_seed_trials(self, count: int) -> Sequence[vz.TrialSuggestion]:
    """First trial is search space center, the rest are quasi-random."""
    seed_suggestions = []
    if not self._history:
      features = self._converter.to_features([])  # to extract shape.
      continuous = self._padding_schedule.pad_features(
          0.5 * np.ones([1, features.continuous.shape[1]])
      )
      categorical = self._padding_schedule.pad_features(
          np.zeros([1, features.categorical.shape[1]], dtype=types.INT_DTYPE)
      )
      model_input = types.ModelInput(continuous, categorical)
      parameters = self._converter.to_parameters(model_input)[0]
      suggestion = vz.TrialSuggestion(
          parameters, metadata=vz.Metadata({'seeded': 'center'})
      )
      seed_suggestions.append(suggestion)
    if (remaining_counts := count - len(seed_suggestions)) > 0:
      seed_suggestions.extend(
          self._quasi_random_sampler.suggest(remaining_counts)
      )
    return seed_suggestions

  def suggest(self, count: int | None = None) -> Sequence[vz.TrialSuggestion]:

    def score_fn(
        xs: types.ModelInput, seed: jax.Array | None = None
    ) -> types.Array:
      del seed  # TODO: Eventually use seed.
      x_trials = [
          vz.Trial(params) for params in self._converter.to_parameters(xs)
      ]
      x_strs = [self.x_serializer.to_str(x_trial) for x_trial in x_trials]

      if not self._history:  # If no history, use random scores.
        return jax.random.uniform(self._rng, shape=(len(x_strs),))
      else:
        dist = self.regressor.predict(x_strs)
        return self._acq_fn(dist)

    if len(self._history) < self._num_seed_trials:
      return self._generate_seed_trials(count)

    prior_features = vb.trials_to_sorted_array(self._history, self._converter)
    best_candidates: vb.VectorizedStrategyResults = self._optimizer(
        score_fn, prior_features=prior_features, count=count
    )
    return vb.best_candidates_to_trials(best_candidates, self._converter)

  def update(
      self, completed: vza.CompletedTrials, all_active: vza.ActiveTrials
  ) -> None:
    """Update the list of completed trials."""
    del all_active
    self._history.extend(copy.deepcopy(completed.trials))

    m_name = self.problem.metric_information.item().name
    xs, ys = [], []
    for trial in completed.trials:
      xs.append(self.x_serializer.to_str(trial))
      y = trial.final_measurement_or_die.metrics[m_name].value
      ys.append(y)

    self.regressor.absorb(xs, ys)
