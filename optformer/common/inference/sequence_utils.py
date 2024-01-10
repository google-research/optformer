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

"""Utilities for manipulating sequences.

Unless otherwise stated, sequence-manipulation functions below should be assumed
to parallelize over the final axis (interpreted as a sequence).
"""

from typing import Dict, Sequence, Union
import jax
from jax.experimental import checkify
import jax.numpy as jnp
from optformer.validation import checkify as _checkify


def count_not_from(
    seq: jnp.ndarray, not_from: Sequence[int] = (0, 1)
) -> jnp.ndarray:
  """Counts the number of elements which are NOT part of `not_from`.

  Useful for collecting the initial index of a token sequence.

  Args:
    seq: Token (int) sequence to be filter-counted. Shape [..., L].
    not_from: Token IDs to ignore. Defaulted to (BOS, EOS) token IDs.

  Returns:
    Filtered count. Last axis is reduce summed. Shape [...].
  """

  where_cond = False
  for ignore_int in not_from:
    where_cond = (seq == ignore_int) | where_cond
  return jnp.where(where_cond, 0, 1).sum(axis=-1)


def reduce_eq(ind: jnp.ndarray) -> jnp.ndarray:
  """Validates if the last axis contains all equal values and reduces.

  e.g. [[1, 1, 1, 1], [2, 2, 2, 2]] -> [1, 2]

  Useful for reducing index tensors which have repeated values.

  Args:
    ind: Possible (int) sequence to be reduced. Shape [..., S].

  Returns:
    Reduced indices of shape [...] if the final axis has repeated values.
    Otherwise raises checkify error.
  """
  if _checkify.enabled():
    all_same = jnp.all(ind == jnp.expand_dims(ind[..., 0], -1), axis=-1)
    checkify.check(
        jnp.all(ind == jnp.expand_dims(ind[..., 0], -1)),
        msg=(
            '`seq` must have repeated values. Offending sequence: '
            f'{ind[jnp.argmin(all_same)]}'
        ),
    )
  return ind[..., 0]


def shift_right(
    seq: jnp.ndarray, insert_left: Sequence[int] = (0,)
) -> jnp.ndarray:
  """Shifts sequence to the right, and inserts new tokens on the left.

  Useful for taking the output of model `[x,y,z,0,0,...]` and turning it
  back into a proper input `[0,x,y,z,0,...]`.

  Args:
    seq: Token (int) sequence to be filter-counted. Shape [..., L].
    insert_left: Token IDs to insert on the left. Defaulted to BOS token.

  Returns:
    Shifted sequence. Shape [..., L].
  """

  shifted_seq = jnp.roll(seq, shift=len(insert_left), axis=-1)
  return shifted_seq.at[..., 0 : len(insert_left)].set(insert_left)


def broadcast_batch(
    batch: Dict[str, jnp.ndarray], sizes: Sequence[int]
) -> Dict[str, jnp.ndarray]:
  """Broadcasts all arrays in a batch.

  Args:
    batch: Dictionary of sequences. Shape [...].
    sizes: New axes introduced.

  Returns:
    Batch with newly broadcasted elements. Shape `sizes + [...]`.
  """

  return {k: jax.lax.broadcast(v, sizes) for k, v in batch.items()}


def find(seq: jnp.ndarray, elem: int, *, not_found: int = -1) -> jnp.ndarray:
  """Finds first occurrence index of `elem` in a sequence.

  Args:
    seq: Token (int) sequence. Shape [..., L].
    elem: Element value to find.
    not_found: Value to return if elem is not found. Defaulted to -1, to output
      a "special value" token and no-op common use-cases zero-ing all elements
      to the right of the index.

  Returns:
    First token index whose value is `elem`, else `not_found`. Shape [...].
  """

  bool_arr = jnp.where(seq == elem, 1, 0)
  maybe_index = jnp.argmax(bool_arr, axis=-1)

  not_found_cond = jnp.sum(bool_arr, axis=-1) == 0
  return jnp.where(not_found_cond, not_found, maybe_index)


def rfind(seq: jnp.ndarray, elem: int, *, not_found: int = -1) -> jnp.ndarray:
  """Same format as `find`, but for last occurrence.

  Useful for finding the last location of a special token (e.g. separator
  token).

  Args:
    seq: See `find`
    elem: See `find`
    not_found: See `find`

  Returns:
    Last token index whose value is `elem`, else `not_found`. Shape [...].
  """

  bool_arr = jnp.where(seq == elem, 1, 0)
  flipped_bool_arr = jnp.flip(bool_arr, axis=-1)
  maybe_index = bool_arr.shape[-1] - 1 - jnp.argmax(flipped_bool_arr, axis=-1)

  not_found_cond = jnp.sum(bool_arr, axis=-1) == 0
  return jnp.where(not_found_cond, not_found, maybe_index)


def append_to_output(
    seq: jnp.ndarray, elems: Sequence[int], *, bos: int = 0
) -> jnp.ndarray:
  """Appends elems to a decoding output sequence.

  Exact location starts from first occurrence of the BOS token. Useful for
  appending values to decoder output sequences.

  Args:
    seq: Token (int) sequence. Shape [..., L].
    elems: Elements to append.
    bos: BOS token ID to determine initial index to append.

  Returns:
    Sequence w/ elements appended, or no-op if appending will overwrite non-bos
      tokens. Shape [..., L].
  """
  # TODO: Raise error if `seq` doesn't look lke a decode output.
  # TODO: Implement.
  raise NotImplementedError


def dynamic_slice_broadcast(
    operand: jax.Array, slice_indices: jax.Array, slice_size: int
) -> jax.Array:
  """Broadcasting version of jax.lax.dynamic_slice_in_dim."""
  fn = jax.lax.dynamic_slice_in_dim
  for i in range(operand.ndim - slice_indices.ndim - 1):
    fn = jax.vmap(fn, in_axes=(i, None, None), out_axes=i)
  for i in range(slice_indices.ndim):
    fn = jax.vmap(
        fn,
        in_axes=(i + operand.ndim - slice_indices.ndim - 1, i, None),
        out_axes=i + operand.ndim - slice_indices.ndim - 1,
    )
  return fn(operand, slice_indices, slice_size)


def rpad(seq: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
  """Right-pads sequence with 0's to match w/ target sequence.

  Args:
    seq: Token (int) sequence. Shape [..., L].
    target: Token (int) sequence to match on the inner dimension. Shape [...,
      L'] where the outer dimensions can be different from seq's.

  Returns:
    Padded sequence. Shape [..., L'].
  """
  noop_paddings = [(0, 0) for _ in range(len(seq.shape) - 1)]
  paddings = noop_paddings + [(0, target.shape[-1] - seq.shape[-1])]
  return jnp.pad(seq, paddings, 'constant')


def slice_update(
    seq: jnp.ndarray, start: Union[int, jnp.ndarray], elems: Sequence[int]
) -> jnp.ndarray:
  """Jittable version of `seq[..., start:start+len(elems)].set(elems)`."""
  # TODO: Finish case when `start` is non-scalar.
  for i, elem in enumerate(elems):
    seq = seq.at[..., start + i].set(elem)
  return seq


def value_mask(seq: jnp.ndarray, masked_values: Sequence[int]) -> jnp.ndarray:
  """Computes value-matched mask from sequence.

  Ex: If `masked_values` are * and |, then:
    seq: [*, |, x, y]
    mask: [0, 0, 1, 1]

  Args:
    seq: Token (int) sequence. Shape [..., L].
    masked_values: Values to mask out.

  Returns:
    Mask of shape [..., L].
  """

  mask = jnp.full(seq.shape, True, dtype=bool)

  for v in masked_values:
    mask = jnp.logical_and(mask, jnp.not_equal(seq, v))

  return mask


# pyformat: disable
def between_mask(seq: jnp.ndarray, left: int, right: int) -> jnp.ndarray:
  """Computes the mask for a sequence given delimiters.

  Ex: If left/right delimiters are '*' and '|', then
    seq: [*, w, x, y, |, *, z, |]
    mask: [0, 1, 1, 1, 0, 0, 1, 0]

  Args:
    seq: Token (int) sequence. Shape [..., L].
    left: Left delimiter.
    right: Right delimiter.

  Returns:
    Mask of shape [..., L].
  """

  left_match = jnp.equal(seq, left)
  right_match = jnp.equal(seq, right)

  if _checkify.enabled():
    # Check if count(left) == count(right)
    left_count = jnp.sum(left_match, axis=-1)
    right_count = jnp.sum(right_match, axis=-1)
    eq_count = jnp.all(left_count == right_count)
    checkify.check(eq_count, '`seq` has imbalanced delimiters.')

  # If our example tensor is [x, *, y, |], then example outputs are commented:
  left_cs = jnp.cumsum(left_match, axis=-1)  # [0, 1, 1, 1]
  right_cs = jnp.cumsum(right_match, axis=-1)  # [0, 0, 0, 1]
  left_cs_slice = left_cs[..., :-1]  # [0, 1, 1]
  zeros = jnp.zeros(shape=list(left_cs_slice.shape[:-1]) + [1], dtype=jnp.int32)
  shifted_left_cs = jnp.concatenate((zeros, left_cs_slice), axis=-1)  # [0, 0, 1, 1]  # pylint: disable=line-too-long
  mask = shifted_left_cs - right_cs  # [0, 0, 1, 0]

  if _checkify.enabled():
    # Check if there are no -1's (from wrong right -> left orderings).
    all_ones_and_zeros = jnp.all((mask == 0) | (mask == 1))
    checkify.check(all_ones_and_zeros, '`seq` has imbalanced delimiters.')

  return mask.astype(jnp.bool_)
# pyformat: enable
