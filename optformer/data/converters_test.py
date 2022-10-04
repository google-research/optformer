"""Tests for converters."""
from absl import logging
from optformer.data import converters
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import grid
from vizier.testing import test_studies
from absl.testing import absltest


class ConvertersTest(absltest.TestCase):

  def setUp(self):
    problem = vz.ProblemStatement()
    problem.search_space = test_studies.flat_space_with_all_types()
    metric_name = 'single_max'
    problem.metric_information.append(
        vz.MetricInformation(
            name=metric_name, goal=vz.ObjectiveMetricGoal.MAXIMIZE))

    # Add metadata.
    problem.metadata['name'] = 'flat_search_space'
    problem.metadata['algorithm'] = 'grid_search'

    designer = grid.GridSearchDesigner(problem.search_space)
    suggestions = designer.suggest(2)
    trials = []
    for i, suggestion in enumerate(suggestions):
      trial = suggestion.to_trial(i + 1)
      trial.complete(vz.Measurement(metrics={metric_name: float(i)}))
      trials.append(trial)

    self.example_flat_study = vz.ProblemAndTrials(
        problem=problem, trials=trials)
    # Configuration overrdies:
    # min_trials = 0: allows a small test study
    # randomize_parameter_order = False: fix the parameter order in test.
    self._config = dict(min_trials=0, randomize_parameter_order=False)
    self.optformer_converter = converters.OptFormerConverter(**self._config)
    super().setUp()

  def test_default_option(self):
    """The default configuration."""

    study_texts = self.optformer_converter.study_to_texts(
        self.example_flat_study)
    correct_study_inputs = """N:"flat_search_space",A:"grid_search",O:"single_max",G:<1>&&N:"boolean",P:<3>,L:<2>,C:["False","True"]*N:"categorical",P:<3>,L:<3>,C:["a","aa","aaa"]*N:"discrete_double",S:<1>,P:<4>,L:<3>,F:[-0.5,1,1.2]*N:"discrete_int",S:<1>,P:<4>,L:<3>,F:[-1,1,2]*N:"discrete_logdouble",S:<2>,P:<4>,L:<3>,F:[1e-05,0.01,0.1]*N:"integer",S:<0>,P:<2>,m:<-2>,M:<2>*N:"lineardouble",S:<1>,P:<1>,m:-1,M:2*N:"logdouble",S:<2>,P:<1>,m:-9.21,M:4.61"""
    correct_study_targets = """<0><0><0><0><0><0><0><0>*<0>|<0><0><0><0><0><0><10><0>*<999>"""

    logging.info(study_texts.inputs)
    self.assertMultiLineEqual(study_texts.inputs, correct_study_inputs)

    logging.info(study_texts.targets)
    self.assertMultiLineEqual(study_texts.targets, correct_study_targets)


if __name__ == '__main__':
  absltest.main()
