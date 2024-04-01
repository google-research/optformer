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

"""Serializers which represent PyGlove objects as flattened {id:value} maps.

In PyGlove, every object has a 'KeyPath' for keeping track of its global
location in e.g. a nested structure. Thus all nested structures in PyGlove can
be represented as a flattened mapping:

  {KeyPath0:Value0, KeyPath1:Value1, ...}

A 'KeyPath' is a sequence of unit keys, e.g. [key0, key1, ...].
"""

from typing import Any, Dict, List, Union

import attrs
from optformer.common import serialization as s_lib
from optformer.pyglove import types
import pyglove as pg


KeyType = Any


class KeySerializer(s_lib.Serializer[KeyType]):
  """Serializes unit keys."""

  def to_str(self, key: KeyType, /) -> str:
    if isinstance(key, pg.geno.ConditionalKey):
      # Conditional keys contain branching information via `.index`.
      return f'={key.index}'
    else:
      # Regular string unit key.
      # TODO: Tokenize keys.
      return str(key)


@attrs.define(auto_attribs=False)
class KeyPathSerializer(s_lib.Serializer[pg.KeyPath]):
  """Serializes PyGlove KeyPaths."""

  key_serializer: s_lib.Serializer[KeyType] = attrs.field(factory=KeySerializer)

  ROOT_KEY: str = '$'
  HIERARCHY_SEPARATOR: str = '.'

  def to_str(self, path: pg.KeyPath, /) -> str:
    """Returns custom "ID" representation from a PyGlove KeyPath object.

    Below are examples comparing to PyGlove's canonical ID naming:

    Example 1:
      Canonical ID: 'a[=0/2].x'
      Keys: ['a', ConditionalKey, 'x']
      Custom ID (Ours): 'a=0.x'

    Example 2:
      Canonical ID: 'a[=0/2].x[0]'
      Keys: ['a', ConditionalKey, 'x', 0]
      Custom ID (Ours): 'a=0.x.0'

    Args:
      path: KeyPath / location of the object.

    Returns:
      Our custom ID.
    """

    serialized_keys = []

    # Check if we need root key.
    if not path.keys or isinstance(path.keys[0], pg.geno.ConditionalKey):
      serialized_keys.append(self.ROOT_KEY)

    serialized_keys.extend([self.key_serializer.to_str(k) for k in path.keys])

    custom_id = self.HIERARCHY_SEPARATOR.join([str(k) for k in serialized_keys])
    return custom_id.replace('.=', '=')


def _unit_keys_from_dna_spec(dna_spec: pg.DNASpec, root_key: str) -> List[str]:
  """Obtains all possible unit keys from a DNASpec and sorts them."""
  key_set = set()
  for decision_id in dna_spec.decision_ids:
    decision_key_set = set(
        k for k in decision_id.keys if not isinstance(k, pg.geno.ConditionalKey)
    )
    key_set.update(decision_key_set or {root_key})

  return sorted(key_set)


CandidatesType = List[Union[str, int]]


@attrs.define
class DNASpecKeyValueSerializer(s_lib.Serializer[pg.DNASpec]):
  """Serializes DNASpec using key-value representation.

  The serialization should contain two types of information:

  1. The actual search space structure (e.g. its hierarchy and feasible values).
  2. Mappings from unit key strings to index tokens (i.e. "pointers") to help
  reduce decoder length
    Ex: '{key0:<k0>, key1:<k1>, ...}'
  """

  keypath_serializer: KeyPathSerializer = attrs.field(factory=KeyPathSerializer)

  def to_str(self, dna_spec: pg.DNASpec, /) -> str:
    # TODO: Use `structure_from_dna_spec` to output structure too.
    unit_keys = _unit_keys_from_dna_spec(dna_spec, KeyPathSerializer.ROOT_KEY)
    unit_key_aliases = {
        k: self.keypath_serializer.key_serializer.to_str(k) for k in unit_keys
    }

    return str(unit_key_aliases)

  def structure_from_dna_spec(
      self, dna_spec: pg.DNASpec
  ) -> Dict[str, CandidatesType]:
    """Returns nested structured information about the DNASpec.

    Example: If search space is `pg.oneof([A, B])` where A, B are search
    spaces themselves, then the dictionary will contain:
    {
      kp_root: [kp_A, kp_B],
      kp_A: [...],
      kp_B: [...],
      ...
      kp_leaf: [0, 1, 2]
    }
    where `kp_` denotes stringified KeyPath.

    Args:
      dna_spec:

    Returns:
      Dictionary whose items consist of {keypath: child_candidates} for every
      decision point in the search space.
    """
    kp_to_candidates = {}
    # We perform a flattened DFS on all decision points.
    for dp in dna_spec.decision_points:
      child_kp_to_candidates = {}

      # If current decision pt is categorical, we inspect its candidate values.
      # A candidate value could be either the name of a child decision point
      # or a number which represents the index of a selected leaf node.
      if isinstance(dp, pg.geno.Choices):
        applicable_values = []
        for i, s in enumerate(dp.candidates):
          if s.is_constant:
            # This candidate is a constant space (leaf node), thus we use the
            # index as the name.
            applicable_values.append(i)
          else:
            # This candidate is a sub-space (non-leaf node), thus we want to
            # include all the immediate child decision names as the candidate
            # names.
            child_kp = dp.id + pg.geno.ConditionalKey(i, len(dp.candidates))
            child_kp_str = self.keypath_serializer.to_str(child_kp)

            applicable_values.append(child_kp_str)

            child_applicable_values = [
                self.keypath_serializer.to_str(c.id) for c in s.elements
            ]
            child_kp_to_candidates[child_kp_str] = child_applicable_values
      else:
        raise NotImplementedError(f'Floats not supported yet: {dp}')

      kp_to_candidates[self.keypath_serializer.to_str(dp.id)] = (
          applicable_values
      )
      kp_to_candidates.update(child_kp_to_candidates)
    return kp_to_candidates


@attrs.define
class DNAKeyValueSerializer(s_lib.Serializer[pg.DNA]):
  """Serializes a DNA.

  Currently the serialization will look like:
    {kp0: v0, kp1: v1, ...}

  where:
    kp = Serialized KeyPath
    v = Serialized Value

  We only consider leaf items with primitive values to reduce serialization
  length. Branch selections from search space will be indicated by the KeyPaths.
  """

  keypath_serializer: s_lib.Serializer[pg.KeyPath] = attrs.field(
      factory=KeyPathSerializer
  )

  def to_str(self, dna: pg.DNA, /) -> str:
    out = {}
    # TODO: Consider if we need to change to 'literal' value_types.
    dna_dict = dna.to_dict(key_type='dna_spec', value_type='dna')
    for k, v in dna_dict.items():
      if not v.children:
        out[self.keypath_serializer.to_str(k.id)] = v.value

    # TODO: Use a primitive serializer instead of `str()`.
    return str(out)


@attrs.define(auto_attribs=False)
class SuggestionsKeyValueSerializer(s_lib.Serializer[types.PyGloveStudy]):
  """Serializes a DNASpec and a list of proposed DNAs for encoder processing.

  The current tokenization scheme consists of:
    [DNASpec, |, DNA1, |, DNA2, |, ...]
  """

  dna_serializer: s_lib.Serializer[pg.DNA] = attrs.field(
      init=True, kw_only=True, factory=DNAKeyValueSerializer
  )
  dna_spec_serializer: s_lib.Serializer[pg.DNASpec] = attrs.field(
      init=True, kw_only=True, factory=DNASpecKeyValueSerializer
  )

  SUGGESTION_SEPARATOR: str = '|'

  def to_str(self, study: types.PyGloveStudy, /) -> str:
    # Global information about study.
    dna_spec = pg.dna_spec(study.search_space)
    encoder_objects = [self.dna_spec_serializer.to_str(dna_spec)]

    for trial in study.trials:
      dna = pg.DNA(trial.suggestion, spec=dna_spec)
      encoder_objects.append(self.dna_serializer.to_str(dna))

    return self.SUGGESTION_SEPARATOR.join(encoder_objects)
