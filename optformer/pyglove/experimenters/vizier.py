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

"""Wrapper around Vizier experimenters."""

import attrs
from optformer.pyglove.experimenters import base
import pyglove as pg
from vizier import pyglove as vzp
from vizier import pyvizier as vz
from vizier.benchmarks import experimenters


@attrs.define
class VizierToPyGloveExperimenter(base.PyGloveExperimenter):
  """Converts Vizier Experimenter to PyGlove Experimenter."""

  experimenter: experimenters.Experimenter = attrs.field(init=True)

  problem: vz.ProblemStatement = attrs.field(init=False)
  converter: vzp.VizierConverter = attrs.field(init=False)

  def __attrs_post_init__(self):
    problem = self.experimenter.problem_statement()
    if not problem.is_single_objective:
      raise ValueError("Vizier Experimenter must be single objective.")
    if problem.metric_information.item().goal.is_minimize:
      self.experimenter = experimenters.SignFlipExperimenter(self.experimenter)

    self.problem = self.experimenter.problem_statement()
    self.converter = vzp.VizierConverter.from_problem(self.problem)

  def evaluate(self, suggestion: pg.DNA) -> float:
    vz_trial = self.converter.to_trial(suggestion, fallback="raise_error")
    self.experimenter.evaluate([vz_trial])
    metric_name = self.problem.single_objective_metric_name
    # TODO: Deal with infeasible trials better.
    return vz_trial.final_measurement.metrics[metric_name].value  # pytype:disable=attribute-error

  def search_space(self) -> pg.DNASpec:
    return self.converter.dna_spec
