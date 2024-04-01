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

"""PyGlove Experimenter factories."""

import random
from typing import Any, Callable, Optional, Sequence, Tuple

import attrs
from optformer.common.data import generators
from optformer.pyglove import experimenters as experimenters_lib


MAX_INT32 = 2**31 - 1


def _interval_validator(instance, attribute, value) -> None:
  del instance
  if value[0] > value[1]:
    raise ValueError(f'Bounds {value} of {attribute.name} are decreasing.')


@attrs.define
class PermutationExperimenterFactory(
    generators.SeededFactory[experimenters_lib.PermutationExperimenter]
):
  """Produces a random permutation-space experimenter."""

  # Callable Args: [n]
  experimenters: Sequence[
      Callable[[int], experimenters_lib.PermutationExperimenter]
  ] = attrs.field(
      default=(
          experimenters_lib.FSSExperimenter,
          experimenters_lib.LOPExperimenter,
          experimenters_lib.QAPExperimenter,
          experimenters_lib.QueenPlacementExperimenter,
          experimenters_lib.TSPExperimenter,
      )
  )

  permutation_size_range: Tuple[int, int] = attrs.field(
      default=(5, 15), validator=_interval_validator
  )
  scale_range: Tuple[float, float] = attrs.field(
      default=(0.01, 10.0), validator=_interval_validator
  )

  def __call__(
      self, seed: Optional[int] = None
  ) -> experimenters_lib.PermutationExperimenter:
    rng = random.Random(seed)
    permutation_size = rng.randint(*self.permutation_size_range)
    scale = rng.uniform(*self.scale_range)
    exptr_ctr = rng.choice(self.experimenters)
    return exptr_ctr(permutation_size, scale=scale, seed=seed)


@attrs.define
class BinomialExperimenterFactory(
    generators.SeededFactory[experimenters_lib.BinomialExperimenter]
):
  """Produces a random n-choose-k experimenter."""

  # Callable Args: [n, k]
  experimenters: Sequence[
      Callable[[int, int], experimenters_lib.BinomialExperimenter]
  ] = attrs.field(
      default=(
          experimenters_lib.CoverageExperimenter,
          experimenters_lib.ModularExperimenter,
          experimenters_lib.LogDeterminantExperimenter,
      )
  )

  n_range: Tuple[int, int] = attrs.field(
      default=(5, 20), validator=_interval_validator
  )
  k_to_n_ratio_range: Tuple[float, float] = attrs.field(
      default=(0.2, 0.8), validator=_interval_validator
  )
  scale_range: Tuple[float, float] = attrs.field(
      default=(0.01, 10.0), validator=_interval_validator
  )

  def __call__(
      self, seed: Optional[int] = None
  ) -> experimenters_lib.BinomialExperimenter:
    rng = random.Random(seed)
    n = rng.randint(*self.n_range)
    k_to_n_ratio = rng.uniform(*self.k_to_n_ratio_range)
    k = int(k_to_n_ratio * n)
    scale = rng.uniform(*self.scale_range)
    exptr_ctr = rng.choice(self.experimenters)
    return exptr_ctr(n, k, scale=scale, seed=seed)


@attrs.define
class NestedExperimenterFactory(
    generators.SeededFactory[experimenters_lib.SwitchExperimenter]
):
  """Generates nested "tree" space via repeated usage of `SwitchExperimenter`."""

  experimenter_factories: Sequence[Any] = attrs.field(
      default=(PermutationExperimenterFactory, BinomialExperimenterFactory)
  )

  # Min/Max depth of the tree.
  depth_range: Tuple[int, int] = attrs.field(
      default=(1, 5), validator=_interval_validator
  )
  # Min/Max number of choices for a non-leaf `SwitchExperimenter` tree node.
  num_switch_range: Tuple[int, int] = attrs.field(
      default=(2, 5), validator=_interval_validator
  )

  def __call__(
      self, seed: Optional[int] = None
  ) -> experimenters_lib.SwitchExperimenter:
    rng = random.Random(seed)
    depth = rng.randint(*self.depth_range)
    return self._recursive_helper(depth, rng)

  def _recursive_helper(
      self, current_depth: int, rng: random.Random
  ) -> experimenters_lib.SwitchExperimenter:
    """Helper function for creating a single node of the tree."""
    if current_depth == 0:
      temp_seed = rng.randint(0, MAX_INT32)
      exptr_factory = rng.choice(self.experimenter_factories)()
      return exptr_factory(temp_seed)

    num_switches = rng.randint(*self.num_switch_range)
    experimenters = [
        self._recursive_helper(current_depth - 1, rng)
        for _ in range(num_switches)
    ]
    return experimenters_lib.SwitchExperimenter(experimenters)


class SymbolicRegressionFactory(
    generators.SeededFactory[experimenters_lib.SymbolicRegressionExperimenter]
):
  """Generates random target functions to symbolically regress upon."""

  def __call__(
      self, seed: Optional[int] = None
  ) -> experimenters_lib.SymbolicRegressionExperimenter:
    return experimenters_lib.SymbolicRegressionExperimenter.from_seed(seed)
