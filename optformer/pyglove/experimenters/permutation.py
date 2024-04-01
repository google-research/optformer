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

"""Permutation-based experimenters."""

import abc
import collections
from typing import Optional

import attrs
import numpy as np
from optformer.pyglove.experimenters import base
import pyglove as pg
import scipy as sp

PermutationType = pg.List


@attrs.define
class PermutationExperimenter(base.PyGloveExperimenter):
  """Base class for permutation experimenters."""

  # Size of permutation.
  n: int = attrs.field(init=True, default=15, validator=attrs.validators.ge(1))

  # For scaling the output / problem attributes.
  scale: float = attrs.field(
      init=True, kw_only=True, default=1.0, validator=attrs.validators.gt(0.0)
  )

  seed: Optional[int] = attrs.field(init=True, kw_only=True, default=None)

  @abc.abstractmethod
  def evaluate(self, suggestion: PermutationType) -> float:
    """Evaluates a permutation."""

  def search_space(self) -> pg.hyper.ManyOf:
    return pg.permutate(range(self.n))


@attrs.define
class FSSExperimenter(PermutationExperimenter):
  """(Permuted) Flowshop Scheduling Problem.

  Given square (N x N) matrix C of costs, find the permutation P that leads to
  lowest:

  sum_{i=1 to N} C_{i, P(i)}.
  """

  _cost_matrix: np.ndarray = attrs.field(init=False)

  def __attrs_post_init__(self):
    rng = np.random.RandomState(self.seed)
    self._cost_matrix = rng.uniform(0.0, self.scale, (self.n, self.n))

  def evaluate(self, suggestion: PermutationType) -> float:
    cost = 0.0
    for i, index in enumerate(suggestion):
      cost += self._cost_matrix[i, index]
    return -1.0 * cost


@attrs.define
class LOPExperimenter(PermutationExperimenter):
  """Linear Ordering Problem.

  Given square matrix M, find the permutation, when applied simultaneously to
  M's rows and columns, that leads to the highest upper-triangular sum of
  entries.
  """

  _matrix: np.ndarray = attrs.field(init=False)

  def __attrs_post_init__(self):
    rng = np.random.RandomState(self.seed)
    self._matrix = rng.uniform(0.0, self.scale, (self.n, self.n))

  def evaluate(self, suggestion: PermutationType) -> float:
    permuted_matrix = self._matrix[np.ix_(suggestion, suggestion)]
    return np.sum(np.triu(permuted_matrix))


@attrs.define
class QAPExperimenter(PermutationExperimenter):
  """Quadratic Assignment Problem.

  Given weight matrix W and distance matrix D, tries to minimize over all
  permutation matrices P:

  Trace(W * P * A * P^T)
  """

  _weight_matrix: np.ndarray = attrs.field(init=False)
  _dist_matrix: np.ndarray = attrs.field(init=False)

  def __attrs_post_init__(self):
    rng = np.random.RandomState(self.seed)
    # TODO: Verify if these matrices need special properties.
    self._weight_matrix = rng.uniform(0.0, self.scale, (self.n, self.n))
    self._dist_matrix = rng.uniform(0.0, self.scale, (self.n, self.n))

  def evaluate(self, suggestion: PermutationType) -> float:
    permutation_matrix = np.zeros((self.n, self.n), dtype=int)
    for i, index in enumerate(suggestion):
      permutation_matrix[i, index] = 1

    first_half = np.matmul(self._weight_matrix, permutation_matrix)
    second_half = np.matmul(self._dist_matrix, permutation_matrix.T)
    return -1.0 * np.trace(np.matmul(first_half, second_half))


@attrs.define
class QueenPlacementExperimenter(PermutationExperimenter):
  """Classic N-Queens placement problem.

  Since the permutation search space already avoids row/column attacks,
  we count the number of pairs of queens diagonally attacking each other.

  The diagonal a queen (x,y) belongs on can be identified by (x+y) and (x-y), so
  we count the number of collisions in each type of diagonal.
  """

  def evaluate(self, suggestion: PermutationType) -> float:
    left_diagonal_ids = [suggestion[i] - i for i in range(self.n)]
    right_diagonal_ids = [suggestion[i] + i for i in range(self.n)]

    left_counter = collections.Counter(left_diagonal_ids)
    right_counter = collections.Counter(right_diagonal_ids)

    counter = 0.0
    counter += sum([sp.special.comb(v, 2) for v in left_counter.values()])
    counter += sum([sp.special.comb(v, 2) for v in right_counter.values()])

    return -1.0 * self.scale * counter


@attrs.define
class TSPExperimenter(PermutationExperimenter):
  """Travelling Salesman Problem.

  Given N randomly placed 2-D cities, goal is to find a path through all cities
  (i.e. a permutation) that minimizes distance travelled.
  """

  _cities: np.ndarray = attrs.field(init=False)

  def __attrs_post_init__(self):
    rng = np.random.RandomState(self.seed)
    self._cities = rng.uniform(
        low=-1.0 * self.scale, high=self.scale, size=(self.n, 2)
    )

  def evaluate(self, suggestion: PermutationType) -> float:
    distance = 0.0
    for i in range(self.n - 1):
      distance += np.linalg.norm(self._cities[i] - self._cities[i + 1])

    return -1.0 * distance
