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

from typing import Any

from optformer.pyglove import types
from optformer.pyglove.serialization import key_value
import pyglove as pg

from absl.testing import absltest
from absl.testing import parameterized


class KeyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.serializer = key_value.KeySerializer()

  @parameterized.parameters(
      ('a', 'a'),
      (0, '0'),
      (pg.geno.ConditionalKey(index=1, num_choices=5), '=1'),
  )
  def test_serialization(self, key: Any, expected: str):
    self.assertEqual(self.serializer.to_str(key), expected)


class KeyPathTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.serializer = key_value.KeyPathSerializer()

  @parameterized.parameters(
      (pg.KeyPath('a'), 'a'),
      (pg.KeyPath(['a', 'b']), 'a.b'),
  )
  def test_serialization(self, path: pg.KeyPath, expected: str):
    self.assertEqual(self.serializer.to_str(path), expected)


class SearchSpaceTest(parameterized.TestCase):
  """Collection of various search spaces too long as test arguments."""

  def setUp(self):
    super().setUp()

    self.all_spaces = {
        'oneof': pg.one_of(['a', 'b', 'c']),
        'manyof': pg.manyof(2, ['hi', 'cool', 'goodbye', 'b', 'ez']),
        'flat_dict': pg.Dict(
            x=pg.one_of([1, 2, 3]),
            y=pg.one_of(['a', 1, 'c']),
            z=pg.one_of(['hi', 'hello', 'goodbye']),
        ),
        'unnamed_nested': pg.oneof(
            [pg.oneof(['hi', 'hello']), pg.manyof(2, ['bye', 'goodbye'])]
        ),
        'named_nested': pg.Dict(
            x=pg.oneof([
                dict(y=pg.oneof(['hi', 'hello'])),
                dict(z=pg.manyof(2, ['bye', 'goodbye'])),
            ])
        ),
        'same_name_nested': pg.Dict(
            a=pg.oneof([
                pg.Dict(x=pg.oneof(range(3)), y=pg.oneof(range(4))),
                pg.Dict(x=pg.oneof(range(3)), y=pg.oneof(range(4))),
            ])
        ),
    }
    self.all_specs = {k: pg.dna_spec(v) for k, v in self.all_spaces.items()}


class DNASpecTest(SearchSpaceTest):

  def setUp(self):
    super().setUp()
    self.serializer = key_value.DNASpecKeyValueSerializer()

  @parameterized.parameters(
      ('oneof',),
      ('manyof',),
      ('flat_dict',),
      ('unnamed_nested',),
      ('named_nested',),
      ('same_name_nested',),
  )
  def test_structure(self, name: str):
    all_structures = {
        'oneof': {'$': [0, 1, 2]},
        'manyof': {'0': [0, 1, 2, 3, 4], '1': [0, 1, 2, 3, 4]},
        'flat_dict': {'x': [0, 1, 2], 'y': [0, 1, 2], 'z': [0, 1, 2]},
        'unnamed_nested': {
            '$': ['$=0', '$=1'],
            '$=0': [0, 1],
            '$=1': ['$=1'],
            '$=1.0': [0, 1],
            '$=1.1': [0, 1],
        },
        'named_nested': {
            'x': ['x=0', 'x=1'],
            'x=0': ['x=0.y'],
            'x=0.y': [0, 1],
            'x=1': ['x=1.z'],
            'x=1.z.0': [0, 1],
            'x=1.z.1': [0, 1],
        },
        'same_name_nested': {
            'a': ['a=0', 'a=1'],
            'a=0': ['a=0.x', 'a=0.y'],
            'a=0.x': [0, 1, 2],
            'a=0.y': [0, 1, 2, 3],
            'a=1': ['a=1.x', 'a=1.y'],
            'a=1.x': [0, 1, 2],
            'a=1.y': [0, 1, 2, 3],
        },
    }

    self.assertEqual(
        all_structures[name],
        self.serializer.structure_from_dna_spec(self.all_specs[name]),
    )

  @parameterized.parameters(
      ('oneof',),
      ('manyof',),
      ('flat_dict',),
      ('unnamed_nested',),
      ('named_nested',),
      ('same_name_nested',),
  )
  def test_serialization(self, name: str):
    all_strings = {
        'oneof': """{'$': '$'}""",
        'manyof': """{'$': '$'}""",
        'flat_dict': """{'x': 'x', 'y': 'y', 'z': 'z'}""",
        'unnamed_nested': """{'$': '$'}""",
        'named_nested': """{'x': 'x', 'y': 'y', 'z': 'z'}""",
        'same_name_nested': """{'a': 'a', 'x': 'x', 'y': 'y'}""",
    }

    self.assertEqual(
        all_strings[name], self.serializer.to_str(self.all_specs[name])
    )


class DNASerializerTest(SearchSpaceTest):

  def setUp(self):
    super().setUp()
    self.serializer = key_value.DNAKeyValueSerializer()

  @parameterized.parameters(
      ('oneof', pg.DNA([1]), """{'$': 1}"""),
      ('manyof', pg.DNA([0, 1]), """{'0': 0, '1': 1}"""),
      ('flat_dict', pg.DNA([0, 1, 2]), """{'x': 0, 'y': 1, 'z': 2}"""),
      ('unnamed_nested', pg.DNA([(0, [1])]), """{'$=0': 1}"""),
      ('unnamed_nested', pg.DNA([(1, [0, 1])]), """{'$=1.0': 0, '$=1.1': 1}"""),
      ('named_nested', pg.DNA([(0, [1])]), """{'x=0.y': 1}"""),
      (
          'named_nested',
          pg.DNA([(1, [0, 1])]),
          """{'x=1.z.0': 0, 'x=1.z.1': 1}""",
      ),
      (
          'same_name_nested',
          pg.DNA([(1, [0, 1])]),
          """{'a=1.x': 0, 'a=1.y': 1}""",
      ),
  )
  def test_serialization(self, name: str, dna: pg.DNA, expected: str):
    spec = self.all_specs[name]
    dna.use_spec(spec)

    self.assertEqual(expected, self.serializer.to_str(dna))


def _generate_study() -> types.PyGloveStudy:
  search_space = pg.one_of(candidates=[0, 1, 2, 3])
  suggestion = next(pg.random_sample(search_space, seed=0))
  objective = 1.0

  trial = types.PyGloveTrial(suggestion=suggestion, objective=objective)
  return types.PyGloveStudy(search_space=search_space, trials=[trial])


class SuggestionsSerializerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.serializer = key_value.SuggestionsKeyValueSerializer()

  def test_serialization(self):
    study = _generate_study()
    expected = """{'$': '$'}|{'$': 3}"""
    out = self.serializer.to_str(study)
    self.assertEqual(out, expected)


if __name__ == '__main__':
  absltest.main()
