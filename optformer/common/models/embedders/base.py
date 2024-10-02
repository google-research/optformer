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

"""Base class for LLM embedders (PaLM-2, etc.).

This is written to support both blackbox and autodifferentiable embedders.
"Blackbox" here means we do not have access / the model is too large to load +
fine-tune, but the only iteraction is a input-output method (e.g. RPC-based
service).
"""

import abc
from typing import Generic, TypeVar
import jax
import jaxtyping as jt

_T = TypeVar('_T')


class Embedder(abc.ABC, Generic[_T]):
  """Base class for LLM embedders."""

  @abc.abstractmethod
  def embed(self, obj: _T) -> jt.Float[jax.Array, '*B D']:
    """Turns an object into embedding vector."""

  @property
  @abc.abstractmethod
  def dimension(self) -> int:
    """The dimensionality (D) of the embedding vector."""
