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

"""Additional types for PyGlove objects."""
from typing import Sequence, Union
import pyglove as pg

# Once `pg.typing.XXX` is made a PyType annotation, we can use these as regular
# pytypes, in e.g. `PyGloveExperimenter`.
SearchSpace = pg.typing.Object(pg.Symbolic)
Suggestion = pg.typing.Any()

# Over time this may be updated to contain np.float32 and np.float64.
ObjectiveValue = pg.typing.Float()


@pg.members(
    [
        ('suggestion', Suggestion, 'Optional name for the Study.'),
        ('objective', ObjectiveValue, 'Objective value.'),
    ],
    metadata={'init_arg_list': ['suggestion', 'objective']},
)
class PyGloveTrial(pg.Object):
  pass


PyGloveTrials = pg.typing.List(pg.typing.Object(PyGloveTrial))


@pg.members(
    [
        ('search_space', SearchSpace, 'Search Space.'),
        ('trials', PyGloveTrials, 'List of evaluated trials.'),
    ],
    metadata={
        'init_arg_list': [
            'search_space',
            'trials',
        ]
    },
)
class PyGloveStudy(pg.Object):
  pass


DNAValue = Union[
    None,
    pg.DNA,
    float,
    int,
    str,
    Sequence[pg.DNA],
    Sequence[int],
    Sequence[str],
]
