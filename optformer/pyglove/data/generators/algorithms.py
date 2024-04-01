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

"""PyGlove Algorithm Factory for flume pipeline."""

import functools
import random
from typing import Optional

import attrs
from optformer.common.data import generators
import pyglove as pg

regularized_evolution = functools.partial(
    pg.evolution.regularized_evolution,
    population_size=50,
    tournament_size=7,
)

nsga2 = functools.partial(
    pg.evolution.nsga2,
    population_size=25,
)

hill_climb = functools.partial(
    pg.evolution.hill_climb,
    batch_size=5,
    init_population_size=10,
)

neat = functools.partial(
    pg.evolution.neat,
    population_size=25,
)


@attrs.define
class EvolutionaryAlgorithmFactory(
    generators.SeededFactory[pg.evolution.Evolution]
):
  """Factory for evolutionary algorithms."""

  algorithms = attrs.field(
      default=(regularized_evolution, nsga2, hill_climb, neat)
  )

  mutators = attrs.field(
      default=(pg.evolution.mutators.Uniform, pg.evolution.mutators.Swap)
  )

  def __call__(self, seed: Optional[int] = None) -> pg.evolution.Evolution:
    rng = random.Random(seed)

    mutator_ctr = rng.choice(self.mutators)
    mutator = mutator_ctr(seed=seed)

    alg_ctr = rng.choice(self.algorithms)
    return alg_ctr(mutator=mutator, seed=seed)
