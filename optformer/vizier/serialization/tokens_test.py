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

from optformer.vizier.serialization import tokens as tokens_lib
from vizier import pyvizier as vz

from absl.testing import absltest
from absl.testing import parameterized


class MeasurementTokenSerializerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.metrics_config = vz.MetricsConfig(
        [vz.MetricInformation("x", goal=vz.ObjectiveMetricGoal.MAXIMIZE)]
    )

    self.active_trial = vz.Trial(final_measurement=None)
    self.assertEqual(self.active_trial.status, vz.TrialStatus.ACTIVE)

    self.infeasible_trial = vz.Trial(final_measurement=None)
    self.infeasible_trial._infeasibility_reason = "infeasible for testing"
    self.assertEqual(self.infeasible_trial.status, vz.TrialStatus.COMPLETED)

    self.nan_trial = vz.Trial(
        final_measurement=vz.Measurement(metrics={"x": math.nan})
    )
    self.assertEqual(self.nan_trial.status, vz.TrialStatus.COMPLETED)

    self.stopping_trial = vz.Trial(final_measurement=None)
    self.stopping_trial.stopping_reason = "made it stopping for testing"
    self.assertEqual(self.stopping_trial.infeasible, False)
    self.assertEqual(self.stopping_trial.status, vz.TrialStatus.STOPPING)

  def test_tokens_used(self):
    serializer = tokens_lib.MeasurementTokenSerializer(self.metrics_config)
    all_tokens = serializer.all_tokens_used()
    float_all_tokens = serializer.float_serializer.all_tokens_used()

    # Float tokens plus <PENDING>, <MISSING_METRIC>, and <INFEASIBLE>
    self.assertLen(all_tokens, 3 + len(float_all_tokens))

  @parameterized.parameters(
      ("<-><4><2><3><0><E0>", -4230.0),
      ("<-><5><0><4><0><E0>", -5040.0),
  )
  def test_serializer_deserializer_regular_measurement(
      self, target_str: str, deserialized_value: float
  ):
    serializer = tokens_lib.MeasurementTokenSerializer(self.metrics_config)

    # Serialization check
    trial = vz.Trial(
        final_measurement=vz.Measurement(metrics={"x": deserialized_value})
    )
    self.assertEqual(serializer.to_str(trial), target_str)

    # Deserialization check
    target_measurement = vz.Measurement(
        metrics={self.metrics_config.item().name: deserialized_value}
    )
    self.assertEqual(serializer.from_str(target_str), target_measurement)

  def test_serializer_deserializer_special_status_trial(self):
    serializer = tokens_lib.MeasurementTokenSerializer(self.metrics_config)
    n_tokens = serializer.float_serializer.num_tokens_per_obj

    # Serialization checks
    self.assertEqual(
        serializer.to_str(self.active_trial), "<PENDING>" * n_tokens
    )
    self.assertEqual(
        serializer.to_str(self.infeasible_trial),
        "<INFEASIBLE>" * n_tokens,
    )
    self.assertEqual(
        serializer.to_str(self.nan_trial), "<INFEASIBLE>" * n_tokens
    )
    self.assertEqual(
        serializer.to_str(self.stopping_trial),
        "<MISSING_METRIC>" * n_tokens,
    )

    # Deserialization checks
    with self.assertRaisesRegex(ValueError, "Cannot float-deserialize.*"):
      serializer.from_str("<PENDING>" * n_tokens)

    # <INFEASIBLE> maps to math.nan
    infeasible = serializer.from_str("<INFEASIBLE>" * n_tokens).metrics["x"]
    self.assertTrue(math.isnan(infeasible.value))
    self.assertIsNone(infeasible.std)
    self.assertEqual(
        serializer.from_str("<MISSING_METRIC>" * n_tokens), vz.Measurement()
    )

  def test_deserializing_incorrect_tokens(self):
    measurement_serializer = tokens_lib.MeasurementTokenSerializer(
        self.metrics_config
    )

    n_tokens = measurement_serializer.float_serializer.num_tokens_per_obj
    # Testing for incorrect number of tokens
    for length in [n_tokens - 1, n_tokens + 1, 0]:  # these lengths are wrong
      with self.assertRaisesRegex(ValueError, "Sequence length.*"):
        measurement_serializer.from_str("<PENDING>" * length)

    # inconsistent special status tokens
    special_tokens = ["<PENDING>", "<INFEASIBLE>", "<MISSING_METRIC>"]
    for t in special_tokens:
      for t_prime in special_tokens:
        if t_prime != t:
          token_list = [t] * n_tokens
          token_list[-1] = t_prime  # making the last token inconsistent
          # these cases would be relegated to float serializer
          with self.assertRaisesRegex(ValueError, "Cannot float-deserialize.*"):
            measurement_serializer.from_str("".join(token_list))


if __name__ == "__main__":
  absltest.main()
