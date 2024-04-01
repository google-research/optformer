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

from optformer.pyglove.experimenters import symbolic_regression
import pyglove as pg
from absl.testing import absltest

Assign = pg.mutfun.Assign
Function = pg.mutfun.Function
Var = pg.mutfun.Var


class SymbolicRegressionTest(absltest.TestCase):

  def test_ground_truth_fitness(self):
    exptr = symbolic_regression.SymbolicRegressionExperimenter.from_nguyen1()
    f = Function(
        'f',
        [
            Assign('t1', Var('x') ** 2),  # x^2
            Assign('t2', Var('x') ** 3),  # x^3
            Assign('t3', Var('x') + Var('t1')),  # x + x^2
            Var('t3') + Var('t2'),  # x + x^2 + x^3
        ],
        args=['x'],
    )
    self.assertEqual(exptr.evaluate(f), 1.0)

  def test_suboptimal_fitness(self):
    exptr = symbolic_regression.SymbolicRegressionExperimenter.from_nguyen1()
    f = Function('f', [Var('x') + Var('x')], args=['x'])
    self.assertBetween(exptr.evaluate(f), 0.0, 0.99)

  def test_two_variable_functions(self):
    exptr = symbolic_regression.SymbolicRegressionExperimenter.from_nguyen11()
    f = Function('x1 ** x2', [Var('x1') ** Var('x2')], args=['x1', 'x2'])
    g = Function('x1 + x2', [Var('x1') + Var('x2')], args=['x1', 'x2'])
    self.assertEqual(exptr.evaluate(f), 1.0)
    self.assertBetween(exptr.evaluate(g), 0.0, 0.99)

  def test_from_seed(self):
    _ = symbolic_regression.SymbolicRegressionExperimenter.from_seed(1)

  def test_evolve(self):
    exptr = symbolic_regression.SymbolicRegressionExperimenter.from_nguyen1()
    search_space = exptr.search_space()
    search_algo = pg.evolution.Evolution(
        (
            pg.evolution.selectors.Random(20)
            >> pg.evolution.selectors.Top(1)
            >> pg.evolution.mutators.Uniform()
        ),
        population_init=(pg.geno.Random(), 1),
        population_update=pg.evolution.selectors.Last(100),
    )

    for example, feedback in pg.sample(
        search_space, search_algo, num_examples=20
    ):
      # We only test logic runs through.
      feedback(exptr.evaluate(example))


if __name__ == '__main__':
  absltest.main()
