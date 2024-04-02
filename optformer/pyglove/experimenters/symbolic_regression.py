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

"""Symbolic regression tasks.

SymbolicRegressionEvolvable adapted from
`pyglove/oss/docs/notebooks/evolution/function_regression.ipynb`.
"""

import inspect
import math
import random
from typing import Any, Callable, Optional, Sequence, Tuple, Type, Union

import attrs
import numpy as np
from optformer.pyglove.experimenters import base
import pyglove as pg

SingleVarFn = Callable[[float], float]
DoubleVarFn = Callable[[float, float], float]

Assign = pg.mutfun.Assign
BinaryOperator = pg.mutfun.BinaryOperator
Function = pg.mutfun.Function
Instruction = pg.mutfun.Instruction
Var = pg.mutfun.Var


# TODO: How to inject seed into `node_transform`?
class SymbolicRegressionEvolvable(pg.hyper.Evolvable):
  """Builds a evolvable symbolic regression problem."""

  @classmethod
  def from_random_function(
      cls,
      num_inputs: int,
      min_lines: int = 1,
      max_lines: int = 20,
      constants: Sequence[int] = (1, -1),
      seed: Optional[int] = None,
  ) -> 'SymbolicRegressionEvolvable':
    """Creates a random initial function and initializes the class."""
    rng = random.Random(seed)

    def where(cls: Type['Instruction']) -> bool:
      return issubclass(cls, BinaryOperator) and cls is not BinaryOperator

    binary_ops = list(Instruction.select_types(where))

    # Randomly generate instructions.
    instructions = []
    for i in range(rng.randint(min_lines, max_lines)):
      num_existing_vars = i + num_inputs

      # Randomly pick a binary op.
      binary_op = rng.choice(binary_ops)

      # Randomly pick operands.
      operands = []
      for _ in range(2):
        # Use variable or constant as operand.
        if rng.choice([True, False]):
          var_index = rng.randint(0, num_existing_vars - 1)
          operands.append(Var(f'v{var_index}'))
        else:
          operands.append(rng.choice(constants))

      # Instruction: Assign node i+1 with binary op output.
      instructions.append(Assign(f'v{i + 1}', binary_op(*operands)))

    input_args = [f'v{i}' for i in range(num_inputs)]
    initial_function = Function('h', instructions, args=input_args)
    return cls(initial_value=initial_function)

  def node_transform(self, k: pg.KeyPath, v: Any, p: pg.Symbolic) -> Any:
    """Modifies a program by swapping out a binary op or a variable."""
    del k, p
    if isinstance(v, BinaryOperator):
      # Randomly replace a binary op
      def where(cls: Type['Instruction']) -> bool:
        return issubclass(cls, BinaryOperator) and cls not in (
            BinaryOperator,
            v.__class__,
        )

      other_binary_ops = list(Instruction.select_types(where))
      another_op = random.choice(other_binary_ops)
      return another_op(**v.sym_init_args)
    elif isinstance(v, Var):
      # Or randomly replace a variable
      variables = v.seen_vars() - set([v.name, v.parent_func().name])
      if variables:
        return Var(random.choice(list(variables)))

    return v

  def weights(
      self, mt: pg.hyper.MutationType, k: pg.KeyPath, v: Any, p: pg.Symbolic
  ) -> float:
    del mt, k, p
    # Evolving only binary op and var.
    if isinstance(v, (BinaryOperator, Var)):
      return 1.0
    return 0.0


def _interval_validator(instance, attribute, value) -> None:
  del instance
  if value[0] > value[1]:
    raise ValueError(f'Bounds {value} of {attribute.name} are decreasing.')


@attrs.define
class SymbolicRegressionExperimenter(base.PyGloveExperimenter):
  """Symbolic regression experimenter.

  Given a target function f', the goal is to find a function f to match f'.
  Objective is distance(f, f'), which uses sampled evaluations f(x) and f'(x).

  See Table 2 of https://arxiv.org/pdf/1912.04871.pdf for a list of tasks.
  """

  target_function: Union[SingleVarFn, DoubleVarFn] = attrs.field(init=True)

  # Attributes for generating random sample points from function.
  x_range: Tuple[float, float] = attrs.field(
      init=True,
      kw_only=True,
      default=(-1.0, 1.0),
      validator=_interval_validator,
  )
  num_samples: int = attrs.field(
      init=True, kw_only=True, default=20, validator=attrs.validators.ge(1)
  )
  seed: Optional[int] = attrs.field(init=True, kw_only=True, default=None)

  # Numerical stability constants.
  eps: float = attrs.field(init=True, kw_only=True, default=1e-7)

  # Evaluation points for determining function distances.
  _xs: np.ndarray = attrs.field(init=False)
  _ys: np.ndarray = attrs.field(init=False)
  _ys_std: float = attrs.field(init=False)

  # Search space.
  evolvable: SymbolicRegressionEvolvable = attrs.field(init=False)

  def __attrs_post_init__(self):
    """Generates function evaluation points and search space."""
    rng = np.random.RandomState(self.seed)
    num_args = len(inspect.signature(self.target_function).parameters.items())
    if num_args == 0:
      raise ValueError(f'Target {self.target_function} has 0 arguments.')

    self._xs = rng.uniform(*self.x_range, size=(self.num_samples, num_args))
    self._ys = np.array([self.target_function(*p) for p in self._xs])
    self._ys_std = np.std(self._ys) + self.eps

    num_inputs = len(inspect.getfullargspec(self.target_function).args)
    self.evolvable = SymbolicRegressionEvolvable.from_random_function(
        num_inputs=num_inputs, seed=self.seed
    )

  def evaluate(self, suggestion: pg.mutfun.Function) -> float:
    """Computes squashed normalized reward."""
    try:
      f = suggestion
      suggested_ys = np.array([f(*x) for x in self._xs])

      # Normalize by MSE. See Page 5 of https://arxiv.org/pdf/1912.04871.pdf.
      mse = np.mean(np.square(self._ys - suggested_ys))
      normalized_rmse = np.sqrt(mse) / self._ys_std
      return 1.0 / (1.0 + normalized_rmse)

    except (ZeroDivisionError, OverflowError, ValueError):
      return 0.0

  def search_space(self) -> pg.hyper.Evolvable:
    return self.evolvable

  @classmethod
  def from_nguyen1(cls) -> 'SymbolicRegressionExperimenter':
    target_function = lambda x: x + x**2 + x**3
    return SymbolicRegressionExperimenter(
        target_function, x_range=(-1.0, 1.0), num_samples=20
    )

  @classmethod
  def from_nguyen11(cls) -> 'SymbolicRegressionExperimenter':
    target_function = lambda x1, x2: x1**x2
    return SymbolicRegressionExperimenter(
        target_function, x_range=(0.0, 1.0), num_samples=20
    )

  @classmethod
  def from_seed(cls, seed: int) -> 'SymbolicRegressionExperimenter':
    """Returns a randomized study."""
    target_function = _generate_valid_single_var_function(seed)
    return SymbolicRegressionExperimenter(
        target_function, x_range=(-1.0, 1.0), num_samples=20, seed=seed
    )


# Helper functions
def _generate_single_var_function(seed: int) -> SingleVarFn:
  """Returns a random single-var function."""

  # The generated function is of form:
  # f1(x) op f2(x)
  # where op is a random sample of +,-,*,/
  # f_i(x) has the form of a_0 + a_1 * x + ... + a_4 * x^4
  # where a_j is sampled from [0, 1, -1].

  rng = random.Random(seed)
  coeffs = [rng.choice([0, 1, -1]) for _ in range(5)]
  # pylint: disable=g-long-lambda
  f1 = lambda x: sum(
      [math.prod(p) for p in zip([1, x, x**2, x**3, x**4], coeffs)]
  )
  coeffs2 = [rng.choice([0, 1, -1]) for _ in range(5)]
  f2 = lambda x: sum(
      [math.prod(p) for p in zip([1, x, x**2, x**3, x**4], coeffs2)]
  )
  # pylint: enable=g-long-lambda
  op = rng.choice(['+', '-', '*', '/'])
  if op == '+':
    return lambda x: f1(x) + f2(x)
  elif op == '*':
    return lambda x: f1(x) * f2(x)
  elif op == '-':
    return lambda x: f1(x) - f2(x)
  else:
    return lambda x: f1(x) / f2(x)


def _generate_valid_single_var_function(seed: int) -> SingleVarFn:
  """Returns a valid single-var function (not producing nan)."""
  rng = random.Random(seed)
  xs = [rng.uniform(-1, 1) for _ in range(20)]
  while True:
    seed = rng.randint(1, 1000000000)
    f = _generate_single_var_function(seed)
    if not any(np.isnan([f(x) for x in xs])):
      return f
