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

"""Normalizations for y-values."""

import abc
from typing import Callable
import jaxtyping as jt
import numpy as np


class StatefulWarper(abc.ABC):
  """y-value warper which has a train / inference mode.

  Assumes objectives are MAXIMIZATION.
  """

  @abc.abstractmethod
  def train(self, ys: jt.Float[np.ndarray, 'L']) -> None:
    """Must be called at least once before calling `warp` or `unwarp`."""
    pass

  @abc.abstractmethod
  def warp(self, ys: jt.Float[np.ndarray, 'K']) -> jt.Float[np.ndarray, 'K']:
    """Warps target y-values, but does not change internal state."""
    pass

  @abc.abstractmethod
  def unwarp(self, ys: jt.Float[np.ndarray, 'K']) -> jt.Float[np.ndarray, 'K']:
    """Unwarps values in normalized space."""
    pass


class MeanStd(StatefulWarper):
  """Standard (y - mean)/std normalizer."""

  def __init__(self):
    self._mean = None
    self._std = None

  def train(self, ys: jt.Float[np.ndarray, 'L']) -> None:
    self._mean = np.mean(ys)
    self._std = np.std(ys)

  def warp(self, ys: jt.Float[np.ndarray, 'K']) -> jt.Float[np.ndarray, 'K']:
    return (ys - self._mean) / self._std

  def unwarp(self, ys: jt.Float[np.ndarray, 'K']) -> jt.Float[np.ndarray, 'K']:
    return ys * self._std + self._mean


class HalfRankWarper(StatefulWarper):
  """Stateful version of Vizier's half-rank warper."""

  def __init__(self):
    self._median = None
    self._good_std = None

  def train(self, ys: jt.Float[np.ndarray, 'L']) -> None:
    raise NotImplementedError


class DampenWaper(StatefulWarper):
  """Applies linear mapping [y_min, y_max] -> [0,1] and a dampening function for any values outside."""

  def __init__(self, dampen_fn: Callable[[float], float]):
    raise NotImplementedError
