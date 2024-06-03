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

import math
from typing import Literal

import gin
import numpy as np
from optformer.original import numeric
from optformer.original import serializers
from optformer.vizier.serialization import tokens as vz_tokens_lib
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import grid
from vizier.testing import numpy_assertions
from vizier.testing import test_studies

from absl.testing import absltest
from absl.testing import parameterized


class QuantizedSerializersTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.problem = vz.ProblemStatement(
        search_space=test_studies.flat_space_with_all_types(),
        metric_information=[
            vz.MetricInformation("x1", goal=vz.ObjectiveMetricGoal.MAXIMIZE),
        ],
    )
    designer = grid.GridSearchDesigner(self.problem.search_space)

    self.completed_trial = designer.suggest()[0].to_trial()
    self.completed_trial.complete(vz.Measurement(metrics={"x1": 0.5}))

    self.no_params_trial = designer.suggest()[0].to_trial()
    self.no_params_trial.parameters.clear()

    self.pending_trial = designer.suggest()[0].to_trial()

    self.infeasible_trial = designer.suggest()[0].to_trial()
    self.infeasible_trial.complete(
        vz.Measurement(), infeasibility_reason="infeas"
    )

    self.missing_trial = designer.suggest()[0].to_trial()
    self.missing_trial.stopping_reason = "missing"  # Hack

    self.scaler = numeric.LinearIntervalScaler(
        source_interval=(-100.0, 100.0), target_interval=(0.0, 1.0)
    )
    self.suggest_serializer = serializers.QuantizedSuggestionSerializer(
        search_space=self.problem.search_space
    )
    self.measurement_serializer = serializers.QuantizedMeasurementSerializer(
        metrics_config=self.problem.metric_information,
        metrics_scalers={
            self.problem.metric_information.item().name: self.scaler
        },
    )
    self.trial_xy_serializer = vz_tokens_lib.TrialTokenSerializer(
        suggestion_serializer=self.suggest_serializer,
        measurement_serializer=self.measurement_serializer,
        order="xy",
    )
    self.trial_yx_serializer = vz_tokens_lib.TrialTokenSerializer(
        suggestion_serializer=self.suggest_serializer,
        measurement_serializer=self.measurement_serializer,
        order="yx",
    )

  def test_suggestion_serializer(self):
    # Grid search immediately picks corner, corresponding to all <0>'s.
    out = self.suggest_serializer.to_str(self.completed_trial)
    expected = "<0><0><0><0><0><0><0><0>"
    self.assertEqual(out, expected)

    out = self.suggest_serializer.to_str(self.no_params_trial)
    expected = """<MISSING_PARAM><MISSING_PARAM><MISSING_PARAM><MISSING_PARAM><MISSING_PARAM><MISSING_PARAM><MISSING_PARAM><MISSING_PARAM>"""
    self.assertEqual(out, expected)

  def test_suggestion_deserializer(self):
    s = "<0><0><0><0><0><0><0><0>"
    trial_params = self.suggest_serializer.from_str(s).parameters.as_dict()
    completed_trial_params = self.completed_trial.parameters.as_dict()
    for k, v in trial_params.items():
      if isinstance(v, float):  # Rounding errors from quantization.
        self.assertAlmostEqual(v, completed_trial_params[k], delta=1e-2)
      else:
        self.assertEqual(v, completed_trial_params[k])

    s = """<MISSING_PARAM><MISSING_PARAM><MISSING_PARAM><MISSING_PARAM><MISSING_PARAM><MISSING_PARAM><MISSING_PARAM><MISSING_PARAM>"""
    suggestion = self.suggest_serializer.from_str(s)
    self.assertEqual(suggestion.parameters, self.no_params_trial.parameters)

    with self.assertRaises(ValueError):
      s = """<0><0><MISSING>"""
      self.suggest_serializer.from_str(s)

  def test_measurement_serializer(self):
    out = self.measurement_serializer.to_str(self.completed_trial)
    self.assertEqual(out, "<502>")

    out = self.measurement_serializer.to_str(self.pending_trial)
    self.assertEqual(out, "<PENDING>")

    out = self.measurement_serializer.to_str(self.infeasible_trial)
    self.assertEqual(out, "<INFEASIBLE>")

    out = self.measurement_serializer.to_str(self.missing_trial)
    self.assertEqual(out, "<MISSING_METRIC>")

  def test_measurement_deserializer(self):
    measurement = self.measurement_serializer.from_str("<502>")
    np.testing.assert_almost_equal(
        measurement.metrics["x1"].value,
        self.completed_trial.final_measurement.metrics["x1"].value,  # pytype:disable=attribute-error
    )

    # MetricsConfig only allows single objectives.
    with self.assertRaises(ValueError):
      self.measurement_serializer.from_str("<502><2>")

    # <PENDING> not allowed during deserialization.
    with self.assertRaises(ValueError):
      self.measurement_serializer.from_str("<PENDING>")

    measurement = self.measurement_serializer.from_str("<INFEASIBLE>")
    self.assertIs(measurement.metrics["x1"].value, math.nan)

    measurement = self.measurement_serializer.from_str("<MISSING_METRIC>")
    self.assertEqual(measurement, vz.Measurement())

  @parameterized.parameters(
      dict(order="xy", expected="<0><0><0><0><0><0><0><0><*><502>"),
      dict(order="yx", expected="<502><*><0><0><0><0><0><0><0><0>"),
  )
  def test_trial_xy_serializer(self, order: Literal["xy", "yx"], expected: str):
    gin.parse_config(f"TrialTokenSerializer.order = '{order}'")
    trial_serializer = vz_tokens_lib.TrialTokenSerializer(
        suggestion_serializer=self.suggest_serializer,
        measurement_serializer=self.measurement_serializer,
    )
    out = trial_serializer.to_str(self.completed_trial)
    self.assertEqual(out, expected)
    gin.clear_config()

  def test_multimetric_serializer(self):
    problem = vz.ProblemStatement(
        search_space=test_studies.flat_space_with_all_types(),
        metric_information=[
            vz.MetricInformation("x1", goal=vz.ObjectiveMetricGoal.MAXIMIZE),
            vz.MetricInformation("x2", goal=vz.ObjectiveMetricGoal.MAXIMIZE),
        ],
    )
    designer = grid.GridSearchDesigner(problem.search_space)

    completed_trial = designer.suggest()[0].to_trial()
    measurement = vz.Measurement(metrics={"x1": 0.5, "x2": 1.0})
    completed_trial.complete(measurement)

    scaler = numeric.LinearIntervalScaler(
        source_interval=(-100.0, 100.0), target_interval=(0.0, 1.0)
    )
    measurement_serializer = serializers.QuantizedMeasurementSerializer(
        metrics_config=problem.metric_information,
        metrics_scalers={"x1": scaler, "x2": scaler},
    )
    out = measurement_serializer.to_str(completed_trial)
    self.assertEqual(out, "<502><505>")


class QuantizedSerializersConditionalSpaceTest(absltest.TestCase):
  """Tests for QuantizedSuggestionSerializer for conditional search spaces."""

  def test_basic(self):
    problem = vz.ProblemStatement(
        search_space=test_studies.conditional_automl_space(),
    )
    trial = vz.Trial(parameters={"model_type": "linear", "learning_rate": 0.03})
    serializer = serializers.QuantizedSuggestionSerializer(
        search_space=problem.search_space
    )
    out = serializer.to_str(trial)
    self.assertEqual(
        out, "<1><619><INACTIVE_CHILD><INACTIVE_CHILD><INACTIVE_CHILD>"
    )

    suggestion = serializer.from_str(out)
    numpy_assertions.assert_arraytree_allclose(
        suggestion.parameters.as_dict(),
        trial.parameters.as_dict(),
        rtol=0,
        atol=1e-3,
    )

  def test_missing_value(self):
    problem = vz.ProblemStatement(
        search_space=test_studies.conditional_automl_space(),
    )
    serializer = serializers.QuantizedSuggestionSerializer(
        search_space=problem.search_space
    )

    trial = vz.Trial(parameters={"model_type": "dnn", "learning_rate": 1e-2})
    out = serializer.to_str(trial)
    # "model_type='dnn'" activates "optimizer_type" child node, which is
    # unset and thus converted to <MISSING_PARAM>. Note that "learning_rate"
    # which is a child node of "optimizer_type='adam'" exists and gets
    # converted, even though "optimizer_type" is unset.
    self.assertEqual(
        out, "<0><500><MISSING_PARAM><INACTIVE_CHILD><INACTIVE_CHILD>"
    )
    suggestion = serializer.from_str(out)
    numpy_assertions.assert_arraytree_allclose(
        suggestion.parameters.as_dict(),
        trial.parameters.as_dict(),
        rtol=0,
        atol=1e-3,
    )

  def test_missing_value_and_bad_value(self):
    problem = vz.ProblemStatement(
        search_space=test_studies.conditional_automl_space(),
    )
    serializer = serializers.QuantizedSuggestionSerializer(
        search_space=problem.search_space
    )
    trial = vz.Trial(parameters={"model_type": "dnn", "NONEXISTENT": 1e-2})
    out = serializer.to_str(trial)
    # "model_type='dnn'" activates "optimizer_type" and "learning_rate" child
    # nodes. Both are unset and thus converted to <MISSING_PARAM>.
    # "NONEXISTENT" is silently dropped.
    self.assertEqual(
        out,
        "<0><MISSING_PARAM><MISSING_PARAM><INACTIVE_CHILD><INACTIVE_CHILD>",
    )


class QuantizedTrialsSerializersTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self.problem = vz.ProblemStatement(
        search_space=test_studies.flat_space_with_all_types(),
        metric_information=[
            vz.MetricInformation("x1", goal=vz.ObjectiveMetricGoal.MAXIMIZE),
            vz.MetricInformation("x2", goal=vz.ObjectiveMetricGoal.MINIMIZE),
        ],
    )
    designer = grid.GridSearchDesigner(self.problem.search_space)
    self.trials = [t.to_trial() for t in designer.suggest(4)]
    self.trials[0].complete(vz.Measurement(metrics={"x1": 25.0, "x2": 24.0}))
    self.trials[1].complete(vz.Measurement(metrics={"x1": 10.0, "x2": 9.0}))
    self.trials[2].complete(vz.Measurement(metrics={"x1": 30.0, "x2": 29.0}))

    self.study = vz.ProblemAndTrials(self.problem, self.trials)

  def test_quantized_trials_serializer(self):
    serializer = serializers.QuantizedTrialsSerializer()
    out = serializer.to_str(self.study)
    expected = "<|><0><0><0><0><0><0><0><0><*><750><750><|><111><0><0><0><0><0><0><0><*><0><0><|><222><0><0><0><0><0><0><0><*><999><999><|><333><0><0><0><0><0><0><0><*><PENDING><PENDING><|>"
    self.assertEqual(out, expected)

  def test_quantized_trials_empty(self):
    # Empty case
    serializer = serializers.QuantizedTrialsSerializer()
    out = serializer.to_str(vz.ProblemAndTrials(self.problem, []))
    expected = "<|>"
    self.assertEqual(out, expected)


if __name__ == "__main__":
  absltest.main()
