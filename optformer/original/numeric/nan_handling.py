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

"""For handling NaN values."""

import attrs
import numpy as np
from optformer.original.numeric import base


@attrs.define
class ObjectiveImputer(base.NumericMapper[np.ndarray, np.ndarray]):
  """Replaces NaN values with penalized values, based on the entire objective statistics of the study."""

  penalty_multiplier: float = attrs.field(
      kw_only=True,
      default=1.0,
      validator=attrs.validators.ge(0.0),
  )
  # Whether the objective is maximize (vs minimized) in this study.
  maximize: bool = attrs.field(
      kw_only=True,
      default=True,
  )

  def map(self, x: np.ndarray) -> np.ndarray:
    """Returns a copy of array replacing NaN values with penalized values."""
    if len(x.shape) != 1:
      raise ValueError(f"Array shape {x.shape} must be 1-D.")

    nan_indices = np.isnan(x)

    nonnan_values = x[~nan_indices]
    if nonnan_values.size == 0:
      raise ValueError(f"Array {x} contains all NaNs.")
    if nonnan_values.size == x.size:
      return x

    obj_min = nonnan_values.min()
    obj_max = nonnan_values.max()
    penalty = self.penalty_multiplier * (obj_max - obj_min)

    if self.maximize:
      imputed_nan_value = obj_min - penalty
    else:
      imputed_nan_value = obj_max + penalty

    obj = np.copy(x)
    obj[nan_indices] = imputed_nan_value
    return obj

  def unmap(self, y: np.ndarray) -> np.ndarray:
    raise NotImplementedError("This class has no unmap method.")
