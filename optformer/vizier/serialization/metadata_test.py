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

from optformer.vizier.serialization import metadata as metadata_lib
from vizier import pyvizier as vz
from absl.testing import absltest


class MetadataSerializerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.serializer = metadata_lib.MetadataSerializer()

    self.nested_metadata = vz.Metadata()
    self.nested_metadata['root_key'] = 'root_value'
    self.nested_metadata.ns('depth')['depth_key'] = 'depth_value'

  def test_serialize_root_only(self):
    serializer = metadata_lib.MetadataSerializer(use_all_namespaces=False)

    # Serializer ignores nested metadata.
    out = serializer.to_str(self.nested_metadata)
    expected = '{root_key:"root_value"}'
    self.assertEqual(out, expected)

    # Serializer still looks at the root namespace, even if the current
    # namespace isn't pointing at root.
    depth_metadata = self.nested_metadata.abs_ns(['depth'])
    out = serializer.to_str(depth_metadata)
    expected = '{root_key:"root_value"}'
    self.assertEqual(out, expected)

  def test_serialize_all_namespaces(self):
    serializer = metadata_lib.MetadataSerializer(use_all_namespaces=True)

    # Serializer uses nested metadata.
    out = serializer.to_str(self.nested_metadata)
    expected = '{root_key:"root_value",depth/depth_key:"depth_value"}'
    self.assertEqual(out, expected)

    # Serializer uses all  namespaces, even if the current
    # namespace isn't pointing at root.
    depth_metadata = self.nested_metadata.abs_ns(['depth'])
    out = serializer.to_str(depth_metadata)
    expected = '{root_key:"root_value",depth/depth_key:"depth_value"}'
    self.assertEqual(out, expected)

  def test_empty(self):
    out = self.serializer.to_str(vz.Metadata())
    expected = '{}'
    self.assertEqual(out, expected)


if __name__ == '__main__':
  absltest.main()
