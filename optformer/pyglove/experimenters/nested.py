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

"""Experimenters for creating nested spaces."""

from typing import Sequence

import attrs
from optformer.pyglove.experimenters import base
import pyglove as pg


@attrs.define
class SwitchExperimenter(base.PyGloveExperimenter):
  """Creates a conditional selection space among base experimenters."""

  experimenters: Sequence[base.PyGloveExperimenter] = attrs.field()

  def evaluate(self, suggestion: pg.List) -> float:
    i, child_suggestion = suggestion
    return self.experimenters[i].evaluate(child_suggestion)

  def search_space(self) -> pg.hyper.OneOf:
    indices_and_children = []
    for i, child in enumerate(self.experimenters):
      indices_and_children.append([i, child.search_space()])

    return pg.one_of(indices_and_children)


@attrs.define
class MultiSwitchExperimenter(base.PyGloveExperimenter):
  """Selects multiple child experimenters and aggregates their output."""

  experimenters: Sequence[base.PyGloveExperimenter] = attrs.field()
  k = attrs.field()

  distinct: bool = attrs.field(default=True)
  aggregation_fn = attrs.field(default=sum)

  def evaluate(self, suggestion: pg.List) -> float:
    child_values = []
    for i, child_suggestion in suggestion:
      child_value = self.experimenters[i].evaluate(child_suggestion)
      child_values.append(child_value)

    return self.aggregation_fn(child_values)

  def search_space(self) -> pg.hyper.ManyOf:
    indices_and_children = []
    for i, child in enumerate(self.experimenters):
      indices_and_children.append([i, child.search_space()])

    return pg.manyof(self.k, indices_and_children)
