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

"""LongT5 utilities for transformering tensors (JAX arrays)."""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from flaxformer.types import Array


def pad_to_multiple(array: Array,
                    factor: int,
                    axis: int,
                    mode: Optional[str] = 'constant',
                    constant_values=0) -> Array:
  """Pads `array` on a given `axis` to be a multiple of `factor`.

  Padding will be concatenated to the end of the axis only, not the beginning.
  If the length along `axis` is already a multiple of `factor`, this is
  effectively a no-op.

  Args:
    array: Array with rank >= 1 to pad.
    factor: Positive integer factor to pad for.
    axis: A valid axis in `array` to pad.
    mode: The padding mode to use according to `jnp.pad`. Defaults to
      'constant'.  See `jax.numpy.pad` documentation for more.
    constant_values: For 'constant' mode, the pad value to use within `jnp.pad`.
      Defaults to 0.

  Returns:
    The padded Array result.
  """
  array = jnp.asarray(array)

  if factor < 1:
    raise ValueError(f'`factor` must be positive but got {factor}.')
  rank = array.ndim
  if axis < -rank or axis >= rank:
    raise ValueError(
        f'`axis` ({axis}) out of bounds for `array` rank ({rank}).')

  axis_len = array.shape[axis]
  pad_len = -axis_len % factor
  pad_width = [(0, 0)] * rank
  pad_width[axis] = (0, pad_len)
  kwargs = {}
  if mode == 'constant':
    kwargs['constant_values'] = constant_values
  return jnp.pad(array=array, pad_width=pad_width, mode=mode, **kwargs)


def split_into_blocks(array: Array,
                      block_len: int,
                      axis: int,
                      pad_value=0) -> Array:
  """Splits an array into blocks along the given `axis`.

  If the axis length isn't a multiple of `block_len`, it'll be padded via
  `pad_to_multiple` first.

  Args:
    array: Array of shape [..., axis_len, ...].
    block_len: Positive integer length of each block.
    axis: A valid axis in `array` to split along.
    pad_value: The scalar pad value to use.  Defaults to 0.  Must be of the same
      type as `array`.

  Returns:
    Array of shape [..., num_blocks, block_len, ...], where
    num_blocks = ceiling(axis_len / block_len).
  """
  array = jnp.asarray(array)

  if block_len < 1:
    raise ValueError(f'`block_len` must be positive but got {block_len}.')
  rank = array.ndim
  if axis < -rank or axis >= rank:
    raise ValueError(
        f'`axis` ({axis}) out of bounds for `array` rank ({rank}).')
  if axis < 0:
    axis += rank

  padded_array = pad_to_multiple(
      array, factor=block_len, axis=axis, constant_values=pad_value)
  padded_len = padded_array.shape[axis]
  num_blocks = padded_len // block_len
  output_shape = (
      array.shape[:axis] + (num_blocks, block_len) + array.shape[(axis + 1):])
  return padded_array.reshape(output_shape)


def concat_3_blocks(blocked_seq: Array,
                    block_axis: int,
                    seq_axis: int,
                    pad_value=0) -> Array:
  """Concatenates 3 consecutive blocks for each input block for local attention.

  This is meant to be called on a blocked sequence as returned by
  `split_into_blocks` for example.  This function augments each block with its
  adjacent left and right blocks so that every token from the original block
  can access all other tokens `block_len` away from it.  The first and last
  input blocks will have `pad_value`-padded blocks to their left and right,
  respectively.

  Args:
    blocked_seq: [..., num_blocks, block_len, ...] shaped Array.
    block_axis: integer axis of the `num_blocks` dimension.
    seq_axis: integer axis of the `block_len` dimension.
    pad_value: The scalar pad value to use for the first and last input blocks.
      Defaults to 0.

  Returns:
    Array of shape [..., num_blocks, 3 * block_len, ...].
  """
  blocked_seq = jnp.asarray(blocked_seq)

  pad_width = [(0, 0)] * blocked_seq.ndim
  pad_width[block_axis] = (1, 1)

  # [..., num_blocks + 2, block_len, ...]
  padded_blocked_seq = jnp.pad(
      blocked_seq, pad_width, constant_values=pad_value)

  num_blocks = blocked_seq.shape[block_axis]
  blocks_list = []
  for i in range(3):
    # We use indexing approach here:
    # https://numpy.org/doc/stable/user/basics.indexing.html#dealing-with-variable-numbers-of-indices-within-programs
    indices = [slice(0, None)] * blocked_seq.ndim
    indices[block_axis] = slice(i, i + num_blocks)
    indices = tuple(indices)
    blocks_list.append(padded_blocked_seq[indices])
  return jnp.concatenate(blocks_list, axis=seq_axis)


def concat_3_blocks_one_hot(blocked_seq: Array,
                            block_axis: int,
                            seq_axis: int,
                            pad_value=0) -> Array:
  """Concatenates 3 consecutive blocks for each input block for local attention.

  This is an alternative implementation to `concat_3_blocks` that should
  return the same output.  It is slightly slower for typical LongT5
  configurations but is substantially faster when training with `scan` due
  to some current inefficiencies in the XLA:SPMD compilation.

  Args:
    blocked_seq: [..., num_blocks, block_len, ...] shaped Array.
    block_axis: integer axis of the `num_blocks` dimension.
    seq_axis: integer axis of the `block_len` dimension.
    pad_value: The scalar pad value to use for the first and last input blocks.
      Defaults to 0.

  Returns:
    Array of shape [..., num_blocks, 3 * block_len, ...].
  """
  # TODO: This implementation follows a roll, then concat, then slice
  # with one-hot `tensordot` strategy.  It may be worth considering other
  # alternative strategies like "slice with one-hot then concat" or
  # "one-hot-like convolutions" which could turn out to be faster if we try
  # them out and benchmark.
  blocked_seq = jnp.asarray(blocked_seq)
  num_blocks = blocked_seq.shape[block_axis]

  pad_width = [(0, 0)] * blocked_seq.ndim
  pad_width[block_axis] = (1, 1)

  # [..., num_blocks + 2, block_len, ...]
  padded_blocked_seq = jnp.pad(
      blocked_seq, pad_width, constant_values=pad_value)

  blocks_list = []

  # Left block
  blocks_list.append(padded_blocked_seq)

  # Center block
  blocks_list.append(jnp.roll(padded_blocked_seq, -1, axis=block_axis))

  # Right block
  blocks_list.append(jnp.roll(padded_blocked_seq, -2, axis=block_axis))

  # [..., num_blocks + 2, 3 * block_len, ...]
  result = jnp.concatenate(blocks_list, axis=seq_axis)

  # Use one-hot `tensordot` to drop the last two blocks so that the final shape
  # is [..., num_blocks, 3 * block_len, ...].  We avoid simple slicing here
  # since it results in poor XLA:SPMD compilations when training a model with
  # `scan`.

  # [num_blocks, num_blocks + 2]
  one_hot_matrix = jnp.eye(num_blocks, num_blocks + 2, dtype=result.dtype)

  # [..., 3 * block_len, ..., num_blocks]
  result = jnp.tensordot(result, one_hot_matrix, axes=([block_axis], [1]))

  # [..., num_blocks, 3 * block_len, ...]
  result = jnp.moveaxis(result, -1, block_axis)

  return result


def make_3block_local_att_mask(block_len: int,
                               input_mask: Array,
                               segment_ids: Optional[Array] = None,
                               use_full_block_att: bool = False,
                               use_causal_mask: bool = False) -> Array:
  """Makes a 3-blocked local attention mask.

  For example, let's say `block_len` is 2 and we have the following
  `input_mask` representing a single example containing 3 tokens padded to
  maximum `seq_len` 5:
    [[1, 1, 1, 0, 0]].
  With other arguments kept as defaults, the output is:
    [[
        [[0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0]],  #
        [[0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]],  #
        [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],  #
    ]]
  The output has `num_blocks = 3`, and each non-padding token is constrained
  to attend locally (local_radius = block_len - 1 = 1).  Padding tokens have
  uniformly 0 mask values.

  Args:
    block_len: integer length of each block.
    input_mask: [batch, seq_len] shaped boolean Array.
    segment_ids: optional [batch, seq_len] shaped integer Array.
    use_full_block_att: if True, attention is not explicitly masked to prevent
      reaching beyond `block_len` (enforcing `local_radius`) and may reach any
      other tokens in the 3 blocks.  Default False.
    use_causal_mask: if True, the attention is explicitly masked to prevent
      attending to tokens with positive relative position to the current
      query token.  Default False.

  Returns:
    [batch, num_blocks, block_len, 3 * block_len] boolean Array with `True`
    for valid attention pairs and `False` for masking attention.
  """

  # [batch, num_blocks, block_len] shape.
  input_mask_blocked = split_into_blocks(
      input_mask, block_len, axis=-1, pad_value=False)

  # [batch, num_blocks, 3 * block_len] shape.
  input_mask_3blocked = concat_3_blocks(
      input_mask_blocked, block_axis=-2, seq_axis=-1, pad_value=False)

  # [batch, num_blocks, block_len, 3 * block_len] shape.
  attention_mask = jnp.logical_and(input_mask_blocked[..., jnp.newaxis],
                                   input_mask_3blocked[..., jnp.newaxis, :])

  if not use_full_block_att:
    # Enforce that tokens are not allowed to attend farther than `local_radius`.
    # Note that `block_len = local_radius + 1`.

    # [block_len, 3 * block_len] shape
    relative_position = make_3block_relative_position(block_len)
    locality_mask = jnp.abs(relative_position) < block_len
    # [1, 1, block_len, 3 * block_len] shape
    locality_mask = locality_mask[jnp.newaxis, jnp.newaxis, :, :]

    attention_mask = jnp.logical_and(attention_mask, locality_mask)

  if use_causal_mask:
    # Enforce that tokens are not allowed to attend to tokens appearing 'later'
    # in the sequence

    # [block_len, 3 * block_len] shape
    relative_position = make_3block_relative_position(block_len)
    causal_mask = relative_position <= 0
    # [1, 1, block_len, 3 * block_len] shape
    causal_mask = causal_mask[jnp.newaxis, jnp.newaxis, :, :]

    attention_mask = jnp.logical_and(attention_mask, causal_mask)

  if segment_ids is None:
    return attention_mask

  padding_segment_id = -1

  # [batch, num_blocks, block_len] shape.
  segment_ids_blocked = split_into_blocks(
      segment_ids, block_len, axis=-1, pad_value=padding_segment_id)

  # [batch, num_blocks, 3 * block_len] shape.
  segment_ids_3blocked = concat_3_blocks(
      segment_ids_blocked,
      block_axis=-2,
      seq_axis=-1,
      pad_value=padding_segment_id)

  # [batch, num_blocks, block_len, 3 * block_len] shape.
  segment_id_att_mask = jnp.equal(segment_ids_blocked[..., jnp.newaxis],
                                  segment_ids_3blocked[..., jnp.newaxis, :])

  return jnp.logical_and(attention_mask, segment_id_att_mask)


def make_3block_relative_position(block_len: int) -> np.ndarray:
  """Makes 3-blocked relative positions for local attention.

  Args:
    block_len: integer length of each block.

  Returns:
    [block_len, 3 * block_len] integer Array of relative positions.

  Note: The sign convention we use is that the relative position is the position
    of the key minus the position of the query; i.e. it is the query position
    which receives a minus sign.
  """
  pos_ids = np.arange(3 * block_len, dtype=np.int32)
  center_pos_ids = pos_ids[block_len:-block_len]
  return pos_ids[np.newaxis, :] - center_pos_ids[:, np.newaxis]


def make_custom_3block_relative_position(block_len: int,
                                         positions: Array) -> Array:
  """Makes customized 3-blocked relative positions for local attention.

  Unlike `make_3block_relative_position`, this function takes the
  `positions` input to customize the relative attention pattern, which may
  be different for each example.

  Args:
    block_len: integer length of each block.
    positions: [batch, seq_len] shaped integer Array.

  Returns:
    [batch, num_blocks, block_len, 3 * block_len] integer Array of relative
    positions.

  Note: The sign convention we use is that the relative position is the position
    of the key minus the position of the query; i.e. it is the query position
    which receives a minus sign.
  """
  positions = jnp.asarray(positions)
  padding_position = -1

  # [batch, num_blocks, block_len] shape.
  positions_blocked = split_into_blocks(
      positions, block_len, axis=-1, pad_value=padding_position)

  # [batch, num_blocks, 3 * block_len] shape.
  positions_3blocked = concat_3_blocks(
      positions_blocked, block_axis=-2, seq_axis=-1, pad_value=padding_position)

  # [batch, num_blocks, block_len, 3 * block_len] shape.
  return (positions_3blocked[..., jnp.newaxis, :] -
          positions_blocked[..., jnp.newaxis])


def constant_init(value, dtype=jnp.float32):
  """Returns an initializer that initializes all values to a constant."""

  def init(unused_key, shape, dtype=dtype):
    return jnp.ones(shape, dtype) * value

  return init


def positions_from_segment_ids(segment_ids: Array) -> Array:
  """Computes packed positions from segment_ids.

  See the following for an example of how packed inputs are represented
  by `segment_ids` and `positions`:
  https://github.com/google/seqio/blob/main/seqio/utils.py#L292

  This functions derives the positions based on the segment_ids alone.

  Args:
    segment_ids: <int32>[batch, length] array of segmentation info for packed
      examples.

  Returns:
    <int32>[batch, length] array of position info for packed examples.
  """
  segment_ids = jnp.asarray(segment_ids)

  # Indicate where new segments start, other than the first segment.
  start_indicator = segment_ids - jnp.pad(
      segment_ids[:, :-1], ((0, 0), (1, 0)), constant_values=1)

  raw_range = jnp.arange(segment_ids.shape[-1])
  reset_offset = jax.lax.cummax(start_indicator * raw_range, axis=1)

  input_mask = segment_ids > 0

  return (raw_range - reset_offset) * input_mask


