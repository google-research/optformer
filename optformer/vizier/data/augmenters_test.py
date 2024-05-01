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

import copy
import random

import numpy as np
from optformer.vizier.data import augmenters
from vizier import pyvizier as vz

from absl.testing import absltest
from absl.testing import parameterized


class SearchSpacePermuterTest(absltest.TestCase):

  def test_e2e(self):
    ss = vz.SearchSpace()
    ss.add(vz.ParameterConfig.factory('A', bounds=(0, 1)))
    ss.add(vz.ParameterConfig.factory('B', bounds=(0, 1)))
    ss.add(vz.ParameterConfig.factory('C', bounds=(0, 1)))

    new_ss = augmenters.SearchSpacePermuter(seed=0).augment(ss)
    self.assertEqual([p.name for p in new_ss.parameters], ['A', 'C', 'B'])


class MetricsConfigPermuterTest(absltest.TestCase):

  def test_e2e(self):
    m1 = vz.MetricInformation(name='A', goal=vz.ObjectiveMetricGoal.MAXIMIZE)
    m2 = vz.MetricInformation(name='B', goal=vz.ObjectiveMetricGoal.MAXIMIZE)
    m3 = vz.MetricInformation(name='C', goal=vz.ObjectiveMetricGoal.MAXIMIZE)

    mc = vz.MetricsConfig([m1, m2, m3])
    study = vz.ProblemAndTrials(
        problem=vz.ProblemStatement(metric_information=mc), trials=[]
    )

    permuter = augmenters.MetricsConfigPermuter(seed=0)

    new_mc = permuter.augment(mc)
    self.assertEqual([m.name for m in new_mc], ['A', 'C', 'B'])

    new_study_mc = permuter.augment_study(study).problem.metric_information
    self.assertEqual([m.name for m in new_study_mc], ['A', 'C', 'B'])


class TrialsPermuterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self.study = vz.ProblemAndTrials(
        problem=vz.ProblemStatement(),
        trials=[vz.Trial(id=i) for i in range(1, 4)],
    )

  def test_e2e(self):
    new_study = augmenters.TrialsPermuter(seed=0).augment_study(self.study)
    trial_ids = [t.id for t in new_study.trials]

    self.assertEqual(trial_ids, [1, 3, 2])


class IncompleteTrialRemoverTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    trials = [vz.Trial(id=i) for i in range(1, 5)]
    for t in trials:
      if t.id % 2 == 0:
        t.complete(vz.Measurement())

    self.study = vz.ProblemAndTrials(
        problem=vz.ProblemStatement(),
        trials=trials,
    )

  def test_e2e(self):
    new_study = augmenters.IncompleteTrialRemover().augment_study(self.study)
    trial_ids = [t.id for t in new_study.trials]

    self.assertEqual(trial_ids, [2, 4])
    for t in new_study.trials:
      self.assertEqual(t.status, vz.TrialStatus.COMPLETED)


class ObjectiveNormalizerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    trial1 = vz.Trial(final_measurement=vz.Measurement(metrics={'m': 50}))
    trial2 = vz.Trial(final_measurement=vz.Measurement(metrics={'m': -100}))
    trial3 = vz.Trial(final_measurement=vz.Measurement(metrics={'m': 100}))
    self.trials = [trial1, trial2, trial3]

  @parameterized.parameters(
      dict(goal=vz.ObjectiveMetricGoal.MAXIMIZE, expected=[0.75, 0.0, 1.0]),
      dict(goal=vz.ObjectiveMetricGoal.MINIMIZE, expected=[0.75, 0.0, 1.0]),
  )
  def test_e2e(self, goal: vz.ObjectiveMetricGoal, expected: list[float]):
    m = vz.MetricInformation(name='m', goal=goal)
    problem = vz.ProblemStatement(metric_information=[m])
    study = vz.ProblemAndTrials(problem, trials=self.trials)

    new_study = augmenters.ObjectiveNormalizer().augment_study(study)
    metrics = [t.final_measurement.metrics['m'].value for t in new_study.trials]  # pytype:disable=attribute-error
    self.assertEqual(metrics, expected)


class TrialsSorterTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    trial1 = vz.Trial(final_measurement=vz.Measurement(metrics={'m': 0.5}))
    trial2 = vz.Trial(final_measurement=vz.Measurement(metrics={'m': 0.3}))
    trial3 = vz.Trial(final_measurement=vz.Measurement(metrics={'m': 1.0}))
    self.trials = [trial1, trial2, trial3]

  @parameterized.parameters(
      dict(goal=vz.ObjectiveMetricGoal.MAXIMIZE, expected=[0.3, 0.5, 1.0]),
      dict(goal=vz.ObjectiveMetricGoal.MINIMIZE, expected=[1.0, 0.5, 0.3]),
  )
  def test_e2e(self, goal: vz.ObjectiveMetricGoal, expected: list[float]):
    m = vz.MetricInformation(name='m', goal=goal)
    problem = vz.ProblemStatement(metric_information=[m])
    study = vz.ProblemAndTrials(problem, trials=self.trials)

    new_study = augmenters.TrialsSorter().augment_study(study)
    metrics = [t.final_measurement.metrics['m'].value for t in new_study.trials]  # pytype:disable=attribute-error
    self.assertEqual(metrics, expected)


class ParetoRankSortAndSubsampleTest(absltest.TestCase):

  def test_single_objective(self):
    problem = vz.ProblemStatement()
    problem.metric_information.append(
        vz.MetricInformation(name='loss', goal=vz.ObjectiveMetricGoal.MINIMIZE)
    )
    study = vz.ProblemAndTrials(problem=problem)
    for i in range(100):
      measurement = vz.Measurement(metrics={'loss': i})
      trial = vz.Trial(parameters={'x': i}, final_measurement=measurement)
      study.trials.append(trial)
    random.shuffle(study.trials)

    augmenter = augmenters.ParetoRankSortAndSubsample(num_trials=[10])
    new_study = augmenter.augment_study(study)

    self.assertSequenceEqual(
        [t.parameters.as_dict()['x'] for t in new_study.trials],
        np.flip(np.linspace(0, 99, 10)).astype(np.int_).tolist(),
    )
    self.assertTrue(study.problem.metadata['N'], '10')

  def test_multi_objectives(self):
    # TODO: Finish.
    pass


class BestTrialOnlyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    trial1 = vz.Trial(final_measurement=vz.Measurement(metrics={'m': 0.5}))
    trial2 = vz.Trial(final_measurement=vz.Measurement(metrics={'m': 0.3}))
    trial3 = vz.Trial(final_measurement=vz.Measurement(metrics={'m': 1.0}))
    self.trials = [trial1, trial2, trial3]

  @parameterized.parameters(
      dict(goal=vz.ObjectiveMetricGoal.MAXIMIZE, expected=1.0),
      dict(goal=vz.ObjectiveMetricGoal.MINIMIZE, expected=0.3),
  )
  def test_e2e(self, goal: vz.ObjectiveMetricGoal, expected: float):
    m = vz.MetricInformation(name='m', goal=goal)
    problem = vz.ProblemStatement(metric_information=[m])
    study = vz.ProblemAndTrials(problem, trials=self.trials)

    new_study = augmenters.BestTrialOnly().augment_study(study)
    self.assertLen(new_study.trials, 1)
    metric = new_study.trials[0].final_measurement.metrics['m'].value  # pytype:disable=attribute-error
    self.assertEqual(metric, expected)


class TrialsSubsamplerTest(absltest.TestCase):

  def test_skip_rate(self):
    problem = vz.ProblemStatement()
    problem.metric_information.append(
        vz.MetricInformation(name='loss', goal=vz.ObjectiveMetricGoal.MINIMIZE)
    )
    study = vz.ProblemAndTrials(problem=problem)
    for i in range(100):
      measurement = vz.Measurement(metrics={'loss': i})
      trial = vz.Trial(parameters={'x': i}, final_measurement=measurement)
      study.trials.append(trial)

    new_study = augmenters.TrialsSubsampler(
        skip_rate=2.0,
    ).augment_study(study)

    self.assertSequenceEqual(
        [t.parameters.as_dict()['x'] for t in new_study.trials],
        np.linspace(0, 99, 50).astype(np.int_).tolist(),
    )
    self.assertTrue(study.problem.metadata['N'], 50)


class HashProblemMetadataTest(absltest.TestCase):

  def test_e2e(self):
    problem = vz.ProblemStatement()
    problem.metric_information.append(
        vz.MetricInformation(name='loss', goal=vz.ObjectiveMetricGoal.MINIMIZE)
    )
    problem.metadata.ns('ns')['key'] = 'value'
    problem.metadata.ns('ns')['key2'] = 'value2'
    problem.metadata.ns('ns2')['key'] = 'value'
    study = vz.ProblemAndTrials(problem=problem)

    augmenters.HashProblemMetadata().augment_study(study)
    self.assertEmpty(study.problem.metadata.subnamespaces())
    self.assertLen(study.problem.metadata, 1)
    self.assertEqual(
        study.problem.metadata['H'],
        'd521814f734bbfcb35ab06c4d5c5da0c15d3a1045fef3f6642df23eabc1135eb',
    )


class ConvertToMaximizationProblemTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    problem = vz.ProblemStatement()
    m1 = vz.MetricInformation(name='m1', goal=vz.ObjectiveMetricGoal.MAXIMIZE)
    m2 = vz.MetricInformation(name='m2', goal=vz.ObjectiveMetricGoal.MINIMIZE)
    problem.metric_information.extend([m1, m2])

    meas = vz.Measurement()
    meas.metrics['m1'] = vz.Metric(value=1.0)
    meas.metrics['m2'] = vz.Metric(value=1.0)
    trial = vz.Trial(final_measurement=meas)

    self.study = vz.ProblemAndTrials(problem=problem, trials=[trial])

  def test_e2e(self):
    flipper = augmenters.ConvertToMaximizationProblem()
    study = flipper.augment(self.study)

    self.assertEqual(
        study.problem.metric_information._metrics[0],
        vz.MetricInformation(name='m1', goal=vz.ObjectiveMetricGoal.MAXIMIZE),
    )
    self.assertEqual(
        study.problem.metric_information._metrics[1],
        vz.MetricInformation(name='m2', goal=vz.ObjectiveMetricGoal.MAXIMIZE),
    )

    trial_metrics = study.trials[0].final_measurement.metrics  # pytype:disable=attribute-error
    self.assertEqual(trial_metrics['m1'].value, 1.0)
    self.assertEqual(trial_metrics['m2'].value, -1.0)

  def test_idempotent(self):
    flipper = augmenters.ConvertToMaximizationProblem()
    study = flipper.augment(self.study)

    idempotent_study = copy.deepcopy(study)
    for _ in range(5):
      study = flipper.augment(study)
      self.assertEqual(study, idempotent_study)


class RandomMetricFlipper(absltest.TestCase):

  def setUp(self):
    super().setUp()
    problem = vz.ProblemStatement()
    m1 = vz.MetricInformation(name='m1', goal=vz.ObjectiveMetricGoal.MAXIMIZE)
    m2 = vz.MetricInformation(name='m2', goal=vz.ObjectiveMetricGoal.MINIMIZE)
    problem.metric_information.extend([m1, m2])

    meas = vz.Measurement()
    meas.metrics['m1'] = vz.Metric(value=1.0)
    meas.metrics['m2'] = vz.Metric(value=1.0)
    trial = vz.Trial(final_measurement=meas)

    self.study = vz.ProblemAndTrials(problem=problem, trials=[trial])

  def test_e2e(self):
    flipper = augmenters.RandomMetricFlipper(seed=1)
    study = flipper.augment(self.study)

    trial_metrics = study.trials[0].final_measurement.metrics  # pytype:disable=attribute-error
    self.assertEqual(trial_metrics['m1'].value, -1.0)
    self.assertEqual(trial_metrics['m2'].value, -1.0)


if __name__ == '__main__':
  absltest.main()
