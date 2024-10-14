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
from typing import Sequence
import jaxtyping as jt
import numpy as np
from scipy import stats


_EPS = 1e-7


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

  def train(self, ys: jt.Float[np.ndarray, 'H']) -> None:
    self._mean = np.mean(ys)
    self._std = np.std(ys)

  def warp(self, ys: jt.Float[np.ndarray, 'L']) -> jt.Float[np.ndarray, 'L']:
    return (ys - self._mean) / self._std

  def unwarp(self, ys: jt.Float[np.ndarray, 'L']) -> jt.Float[np.ndarray, 'L']:
    return ys * self._std + self._mean


class HalfRankWarper(StatefulWarper):
  """Stateful version of Vizier's half-rank warper."""

  def __init__(self):
    self._median = None
    self._good_std = None

    self._original_data = None
    self._ranks = None

  def train(self, ys: jt.Float[np.ndarray, 'H']) -> None:
    self._median = np.median(ys)

    good_half = ys[ys >= self._median]
    self._good_std = np.sqrt(np.average((good_half - self._median) ** 2))

    self._original_data = ys
    self._ranks = stats.rankdata(ys)  # All ranks within [1, H]

  def warp(self, ys: jt.Float[np.ndarray, 'L']) -> jt.Float[np.ndarray, 'L']:
    # Only affects "bad values" (below the median).
    bad_idx = ys < self._median

    ranks = stats.rankdata(ys)
    # Convert bad value ranks to percentiles in (0, 0.5) exclusive.
    bad_rank_percentiles = ranks[bad_idx] / len(ys)
    # Obtain z-scores for bad ranks. These are in (-inf, 0.0) exclusive.
    bad_z_scores = stats.norm.ppf(bad_rank_percentiles)
    # Normally distribute them.
    replacement_values = self._median + bad_z_scores * self._good_std

    new_ys = np.copy(ys)
    new_ys[bad_idx] = replacement_values
    return new_ys

  def unwarp(self, ys: jt.Float[np.ndarray, 'L']) -> jt.Float[np.ndarray, 'L']:
    # If at least stored median, no change.
    # If below, invert the rank warping.
    bad_idx = ys < self._median

    bad_values = ys[bad_idx]
    bad_z_scores = (bad_values - self._median) / self._good_std
    bad_rank_percentiles = stats.norm.cdf(bad_z_scores)
    bad_ranks = bad_rank_percentiles * len(ys)

    del bad_ranks
    raise NotImplementedError('Not done yet')


class LinearScalingWarper(StatefulWarper):
  """Linearly scales y-values to [-1,1] based on min/max."""

  def __init__(self):
    self._y_min = None
    self._y_max = None

  def train(self, ys: jt.Float[np.ndarray, 'H']) -> None:
    self._y_min = np.min(ys)
    self._y_max = np.max(ys)

  def warp(self, ys: jt.Float[np.ndarray, 'L']) -> jt.Float[np.ndarray, 'L']:
    norm_diff = (ys - self._y_min) / (self._y_max - self._y_min) - 0.5
    return 2.0 * norm_diff

  def unwarp(self, ys: jt.Float[np.ndarray, 'L']) -> jt.Float[np.ndarray, 'L']:
    return (0.5 * ys + 0.5) * (self._y_max - self._y_min) + self._y_min


class LogDampenWaper(StatefulWarper):
  """Log-based dampening function to scale down really high values."""

  def __init__(self):
    pass

  def train(self, ys: jt.Float[np.ndarray, 'H']) -> None:
    pass

  def warp(self, ys: jt.Float[np.ndarray, 'L']) -> jt.Float[np.ndarray, 'L']:
    return np.sign(ys) * np.log1p(np.abs(ys))

  def unwarp(self, ys: jt.Float[np.ndarray, 'L']) -> jt.Float[np.ndarray, 'L']:
    return np.sign(ys) * (np.exp(np.abs(ys)) - 1)


class SigmoidDampenWarper(StatefulWarper):
  """Sigmoid dampening function to map R -> [-scale, scale]."""

  def __init__(self, curvature: float = 0.1, scale: float = 2.0):
    self._curvature = curvature
    self._scale = scale

  def train(self, ys: jt.Float[np.ndarray, 'H']) -> None:
    pass

  def warp(self, ys: jt.Float[np.ndarray, 'L']) -> jt.Float[np.ndarray, 'L']:
    warped = 2.0 / (1.0 + np.exp(-self._curvature * ys)) - 1.0
    return self._scale * warped

  def unwarp(self, ys: jt.Float[np.ndarray, 'L']) -> jt.Float[np.ndarray, 'L']:
    warped = ys / self._scale
    return -np.log(2.0 / (warped + 1.0) - 1 + _EPS) / self._curvature


class SequentialWarper(StatefulWarper):
  """Applies a sequence of warpers."""

  def __init__(self, warpers: Sequence[StatefulWarper]):
    self._warpers = warpers

  def train(self, ys: jt.Float[np.ndarray, 'H']) -> None:
    temp_ys = ys
    for warper in self._warpers:
      warper.train(temp_ys)
      temp_ys = warper.warp(temp_ys)

  def warp(self, ys: jt.Float[np.ndarray, 'L']) -> jt.Float[np.ndarray, 'L']:
    temp_ys = ys
    for warper in self._warpers:
      temp_ys = warper.warp(temp_ys)
    return temp_ys

  def unwarp(self, ys: jt.Float[np.ndarray, 'L']) -> jt.Float[np.ndarray, 'L']:
    temp_ys = ys
    for warper in reversed(self._warpers):
      temp_ys = warper.unwarp(temp_ys)
    return temp_ys


def default_warper() -> StatefulWarper:
  return SequentialWarper([
      HalfRankWarper(),
      LinearScalingWarper(),
      LogDampenWaper(),
  ])
