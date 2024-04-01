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

"""Collection of binomial (n-choose-k) problems.

Many examples include https://en.wikipedia.org/wiki/Submodular_set_function.
"""

import abc
from typing import List, Optional, Set

import attrs
import numpy as np
from optformer.pyglove.experimenters import base
import pyglove as pg

ChoiceType = pg.List


@attrs.define
class BinomialExperimenter(base.PyGloveExperimenter):
  """Base class for binomial (n-choose-k) experimenters."""

  n: int = attrs.field(init=True, default=15, validator=attrs.validators.ge(1))
  k: int = attrs.field(init=True, default=4, validator=attrs.validators.ge(0))

  # For scaling the output / problem attributes.
  scale: float = attrs.field(
      init=True, kw_only=True, default=1.0, validator=attrs.validators.gt(0.0)
  )
  seed: Optional[int] = attrs.field(init=True, kw_only=True, default=None)

  @abc.abstractmethod
  def evaluate(self, suggestion: ChoiceType) -> float:
    """Evaluates a k-set from n indices."""

  def search_space(self) -> pg.hyper.ManyOf:
    return pg.manyof(self.k, range(self.n), distinct=True)


@attrs.define
class ModularExperimenter(BinomialExperimenter):
  """Linear (Modular) Function."""

  monotone: bool = attrs.field(init=True, kw_only=True, default=False)

  _weights: np.ndarray = attrs.field(init=False)

  def __attrs_post_init__(self):
    rng = np.random.RandomState(self.seed)

    self._weights = rng.uniform(
        low=-1.0 * self.scale, high=self.scale, size=(self.n,)
    )
    if self.monotone:
      self._weights = np.abs(self._weights)

  def evaluate(self, suggestion: ChoiceType) -> float:
    objective = 0.0
    for index in suggestion:
      objective += self._weights[index]
    return objective


@attrs.define
class CoverageExperimenter(BinomialExperimenter):
  """(Possibly weighted) Coverage Function."""

  support_size: int = attrs.field(
      init=True, kw_only=True, default=20, validator=attrs.validators.gt(0)
  )
  monotone: bool = attrs.field(init=True, kw_only=True, default=False)
  weighted: bool = attrs.field(init=True, kw_only=True, default=True)

  _weights: np.ndarray = attrs.field(init=False)
  _covers: List[Set[int]] = attrs.field(init=False)

  def __attrs_post_init__(self):
    rng = np.random.RandomState(self.seed)

    if self.weighted:
      self._weights = rng.uniform(low=-1.0, high=1.0, size=(self.support_size,))
    else:
      self._weights = np.ones((self.support_size,), dtype=float)
    if self.monotone:
      self._weights = np.abs(self._weights)
    self._weights = self.scale * self._weights

    # Sample n random covers from {0, 1, ..., support_size - 1}.
    self.covers = []
    for _ in range(self.n):
      cover_indicators = rng.choice([False, True], size=(self.support_size,))
      cover = np.where(cover_indicators)[0]
      self.covers.append(set(cover))

  def evaluate(self, suggestion: ChoiceType) -> float:
    selected_covers = [self.covers[i] for i in suggestion]
    union = set.union(*selected_covers)
    return sum(self._weights[list(union)])


@attrs.define
class LogDeterminantExperimenter(BinomialExperimenter):
  """Takes log-determinant of a selected submatrix of a larger PSD matrix."""

  _psd_matrix: np.ndarray = attrs.field(init=False)

  def __attrs_post_init__(self):
    rng = np.random.RandomState(self.seed)

    rand_matrix = rng.rand(self.n, self.n)
    self._psd_matrix = np.matmul(rand_matrix, rand_matrix.T) * self.scale

  def evaluate(self, suggestion: ChoiceType) -> float:
    submatrix = self._psd_matrix[np.ix_(suggestion, suggestion)]
    return np.log(np.linalg.det(submatrix))
