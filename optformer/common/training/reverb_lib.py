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

"""Library functions for interacting with Reverb."""

import datetime
import threading
import time

from absl import logging
import reverb


def checkpoint_reverb(reverb_client: reverb.Client, interval_seconds: float):
  """Periodically sends checkpoint request to the reverb server."""
  while True:
    try:
      reverb_client.checkpoint()
      logging.info('Reverb checkpointed at %s.', datetime.datetime.now())
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning('Reverb checkpoint failed: %s', e)
    finally:
      time.sleep(interval_seconds)


def launch_checkpoint_thread(reverb_address: str):
  """Put checkpointing client into a background thread."""
  reverb_client = reverb.Client(reverb_address)
  checkpoint_thread = threading.Thread(
      target=checkpoint_reverb, args=(reverb_client, 60 * 15), daemon=True
  )
  checkpoint_thread.start()
