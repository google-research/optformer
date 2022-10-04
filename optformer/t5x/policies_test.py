"""Tests for policies."""
from optformer.t5x import inference_utils
from optformer.t5x import policies
from vizier import algorithms as vza
from vizier import benchmarks
from absl.testing import absltest


class PoliciesTest(absltest.TestCase):

  def test_e2e(self):
    experimenter = benchmarks.IsingExperimenter(lamda=0.01)

    inference_model = inference_utils.InferenceModel.from_checkpoint(
        **policies.DEFAULT_INFERENCE_MODEL_KWARGS)
    designer = policies.OptFormerDesigner(
        experimenter.problem_statement(), inference_model=inference_model)

    for _ in range(2):
      suggestions = designer.suggest(1)
      trials = [suggestion.to_trial() for suggestion in suggestions]
      experimenter.evaluate(trials)
      designer.update(vza.CompletedTrials(trials))


if __name__ == '__main__':
  absltest.main()
