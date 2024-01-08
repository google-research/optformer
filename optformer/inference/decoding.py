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

"""Useful decoding-related classes."""

import abc
from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from optformer.inference import sequence_utils as seq_utils
from t5x import decoding


class IndexLogitRestrictor(decoding.LogitCallbackFn):
  """Restricts logit values depending only on the index."""

  def __call__(
      self,
      logits: Float[Array, "BS E"],
      state: decoding.SamplingLoopState,
      shift: Optional[Int[Array, "B"]] = None,
  ) -> Float[Array, "BS E"]:
    """Uses shifted current index to obtain logit mask index.

    Args:
      logits: Decoder logits used for sampling vocabulary indices at a specific
        time-slice. NOTE: `E >= V` assumed, where `E` is last-axis size (size of
        embedding table).
      state: State of the sampling loop. Most shapes of form [B*S, ...].
      shift: Shift on current index to determine mask index. If `None`,
        `mask_index` is defaulted to `state.step`, equivalent to when `shift` is
        the start of decoding block (usually the case).

    Returns:
      Restricted logits on unmasked tokens.
    """
    if shift is None:
      mask_index = state.step  # Scalar
    else:
      cur_index = jnp.reshape(state.cur_index, (shift.shape[0], -1))  # [B, S]
      mask_index = jnp.reshape(cur_index - shift, (-1,))  # [B*S]

    # Will be broadcasted along the final axis of logits.
    curr_mask: Float[Array, "BS V"] = self.logit_mask(mask_index)
    # Pad w/ (E-V) zeros to deal w/ extra embeddings.
    curr_mask: Float[Array, "BS E"] = seq_utils.rpad(curr_mask, logits)

    return (1.0 - curr_mask) * decoding.NEG_INF + curr_mask * logits

  @abc.abstractmethod
  def logit_mask(self, index: jnp.ndarray) -> Float[Array, "BS V"]:
    """Returns logit mask at index."""
