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

"""Converts Vizier metadata to strings."""

from typing import Dict

import attrs
from optformer.common import serialization as s_lib
from vizier import pyvizier as vz


@attrs.define(frozen=True, kw_only=True)
class MetadataSerializer(s_lib.Serializer[vz.Metadata]):
  """Serializes Vizier ProblemStatement metadata to string.

  Attributes:
    primitive_serializer: Used for serializing metadata keys and values.
    use_all_namespaces: The Metadata can be nested (via "namespacing"). If this
      attribute is True, then all nested metadata are serialized. If False, only
      the root namespace (typically written by users) is serialized and nested
      namespaces (typically written by algorithms and synthetic benchmarks) are
      ignored.
  """

  primitive_serializer: s_lib.Serializer[Dict[str, str]] = attrs.field(
      factory=s_lib.PrimitiveSerializer,
  )

  use_all_namespaces: bool = attrs.field(default=True)

  def to_str(self, metadata: vz.Metadata, /) -> str:
    """Serializes Vizier metadata, ignoring `any_pb2` protos."""
    if self.use_all_namespaces:
      metadata_as_dict: Dict[str, str] = {
          '/'.join(tuple(ns) + (k,)): v
          for ns, k, v in metadata.all_items()
          if isinstance(v, str)
      }
    else:
      metadata_as_dict: Dict[str, str] = {
          k: v for k, v in metadata.abs_ns().items() if isinstance(v, str)
      }
    return self.primitive_serializer.to_str(metadata_as_dict)
