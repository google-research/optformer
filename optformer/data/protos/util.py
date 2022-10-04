"""Methods to convert protos to PyVizier objects."""
from optformer.data.protos import study_pb2
from vizier.service import pyvizier as oss_vz


class ProblemAndTrialsConverter:
  """A wrapper for study_pb2.ProblemAndTrials."""

  @classmethod
  def from_proto(cls,
                 proto: study_pb2.ProblemAndTrials) -> oss_vz.ProblemAndTrials:
    """Converts proto to Python object."""
    problem = oss_vz.ProblemStatementConverter.from_proto(proto.problem)
    trials = oss_vz.TrialConverter.from_protos(proto.trials)
    return oss_vz.ProblemAndTrials(problem, trials)

  @classmethod
  def to_proto(
      cls, problem_and_trials: oss_vz.ProblemAndTrials
  ) -> study_pb2.ProblemAndTrials:
    """Converts Python object to proto."""
    problem_proto = oss_vz.ProblemStatementConverter.to_proto(
        problem_and_trials.problem)
    trial_protos = oss_vz.TrialConverter.to_protos(problem_and_trials.trials)
    return study_pb2.ProblemAndTrials(
        problem=problem_proto, trials=trial_protos)
