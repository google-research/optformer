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

"""Functions for scaling numeric inputs."""

from typing import Optional, Protocol, Tuple, Union

import attrs
import numpy as np
from optformer.original.numeric import base
from optformer.validation import runtime


def _interval_validator(instance, attribute, value) -> None:
  attrs.validators.instance_of((int, float))(instance, attribute, value[0])
  attrs.validators.instance_of((int, float))(instance, attribute, value[1])
  attrs.validators.instance_of(tuple)(instance, attribute, value)

  if value[0] > value[1]:
    raise ValueError(f'Bounds {value} of {attribute.name} are decreasing.')


FloatLike = Union[float, np.ndarray]
IntervalType = Tuple[float, float]


def _equal_bounds(interval: IntervalType) -> bool:
  return interval[0] == interval[1]


def _midpoint(interval: IntervalType) -> float:
  return (interval[0] + interval[1]) / 2.0


class IntervalSampler(Protocol):

  def __call__(self, seed: Optional[int] = None) -> IntervalType:
    """Returns a random interval."""


@attrs.define
class UniformIntervalSampler(IntervalSampler):
  """Samples a random subinterval of [0, 1].

  The interval length is sampled uniformly from provided bounds.
  length ~ U[a, b]
  min ~ U[0, 1-length]
  max = min + length
  """

  length_bounds: Tuple[float, float] = attrs.field(
      init=True,
      validator=_interval_validator,
      converter=tuple,
  )

  def __call__(self, seed: Optional[int] = None) -> IntervalType:
    rng = np.random.RandomState(seed)
    # Sample the interval length uniformly within the bounds.
    interval_range = rng.uniform(*self.length_bounds)
    # Sample the lower bound uniformly in (0, 1-range).
    interval_min = np.random.uniform(0, 1.0 - interval_range)
    interval_max = interval_min + interval_range
    return (interval_min, interval_max)


@attrs.define
class LinearIntervalScaler(base.NumericMapper[FloatLike, FloatLike]):
  """Linearly maps one point on an interval to another.

  If the domain interval has equal bounds, all points will map to the codomain
  interval's midpoint.
  """

  source_interval: IntervalType = attrs.field(
      init=True,
      validator=_interval_validator,
      kw_only=True,
  )
  target_interval: IntervalType = attrs.field(
      init=True,
      validator=_interval_validator,
      kw_only=True,
  )

  def map(self, x: FloatLike) -> FloatLike:
    """Input must be contained in the source interval."""
    runtime.assert_in_interval(self.source_interval, x)
    if _equal_bounds(self.source_interval):
      return _midpoint(self.target_interval)

    return np.interp(x, self.source_interval, self.target_interval)

  def unmap(self, y: FloatLike) -> FloatLike:
    """Input must be contained in the target interval."""
    runtime.assert_in_interval(self.target_interval, y)
    if _equal_bounds(self.target_interval):
      return _midpoint(self.source_interval)

    return np.interp(y, self.target_interval, self.source_interval)
