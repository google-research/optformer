# Copyright 2022 Google LLC.
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

"""Tests for T5x tasks."""
import os

from absl import flags
from absl import logging
import gin

from optformer.data import tasks
from absl.testing import absltest

FLAGS = flags.FLAGS


class VizierTaskWithInputsTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super(VizierTaskWithInputsTest, cls).setUpClass()
    cls.root = os.path.join(FLAGS.test_srcdir,
                            'optformer/t5x/configs')
    gin.add_config_file_search_path(cls.root)

  def setUp(self):
    super().setUp()
    gin.clear_config()

  def test_vizier_task_data_loader(self):
    path = os.path.join(self.root, 'bbob.gin')
    gin.parse_config_file(path)
    gin.finalize()  # Check for required values, etc.

    build_dataset_with_gin = gin.get_configurable(
        tasks.build_vizier_dataset_pyfunc)
    ds = build_dataset_with_gin(split='test', shuffle_files=False)
    data_dict = next(iter(ds))
    for key, value in data_dict.items():
      logging.info(key)
      logging.info(value)


if __name__ == '__main__':
  absltest.main()
