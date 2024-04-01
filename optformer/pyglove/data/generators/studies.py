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

"""Methods for generating entire PyGlove studies."""

import random
from typing import Optional, Sequence

import attrs
from optformer.common.data import generators
from optformer.pyglove import experimenters as experimenters_lib
from optformer.pyglove import types
from optformer.pyglove.data.generators import algorithms
from optformer.pyglove.data.generators import experimenters
import pyglove as pg


@attrs.define
class SyntheticStudyFactory(generators.SeededFactory[types.PyGloveStudy]):
  """Generates studies from synthetic benchmarks."""

  # TODO: Make these actual factory objects, not classes.
  experimenter_factories: Sequence[
      generators.SeededFactory[experimenters_lib.PyGloveExperimenter]
  ] = attrs.field(
      default=(
          experimenters.BinomialExperimenterFactory(),
          experimenters.NestedExperimenterFactory(),
          experimenters.PermutationExperimenterFactory(),
          experimenters.SymbolicRegressionFactory(),
      )
  )
  algorithm_factories = attrs.field(
      default=(algorithms.EvolutionaryAlgorithmFactory,)
  )

  num_trials: int = attrs.field(default=400)

  def __call__(self, seed: Optional[int] = None) -> types.PyGloveStudy:
    rng = random.Random(seed)

    exptr_factory = rng.choice(self.experimenter_factories)
    experimenter = exptr_factory(seed)

    alg_factory_cls = rng.choice(self.algorithm_factories)
    alg_factory = alg_factory_cls()
    alg = alg_factory(seed)

    trials = []
    for suggestion, feedback in pg.sample(
        experimenter.search_space(), alg, num_examples=self.num_trials
    ):
      objective = experimenter.evaluate(suggestion)
      trials.append(types.PyGloveTrial(suggestion, objective))
      feedback(objective)

    return types.PyGloveStudy(
        search_space=experimenter.search_space(), trials=trials
    )
