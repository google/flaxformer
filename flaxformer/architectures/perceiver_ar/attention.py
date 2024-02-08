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

"""Perceiver AR attention utilities."""

from typing import Optional
import jax.numpy as jnp

from flaxformer.architectures.perceiver_ar import slicing
from flaxformer.components.attention import dense_attention
from flaxformer.types import Array
from flaxformer.types import DType


def make_causal_mask(x: Array,
                     sequence_lengths: Array,
                     num_latents: int,
                     dtype: DType = jnp.float32) -> Array:
  """Make a causal mask for self-attention.

  The self-attention weights will be `[batch, heads, num_latents, len]` and this
  function will produce a causal mask of shape `[batch, 1, num_latents, len]`.

  Note that a causal mask does not depend on the values of x; it only depends on
  the shape. If x has padding elements, they will not be treated in a special
  manner.

  Args:
    x: Input array of shape `[batch, len]`
    sequence_lengths: Input sequence lengths of shape `[batch]`
    num_latents: Number of Perceiver AR latents.
    dtype: Mask return dtype

  Returns:
    A `[batch..., 1, len, len]` shaped causal mask for 1d attention.
  """
  if x.ndim != 2:
    raise ValueError(
        f'Inputs must have a shape of [batch, len], but got {x.shape}')
  # [batch, num_latents]
  query_idxs = jnp.broadcast_to(
      jnp.arange(num_latents, dtype=jnp.int32), x.shape[:-1] + (num_latents,))

  # [batch]
  query_idxs_offset = jnp.maximum(0, sequence_lengths - num_latents)
  # Expand to [batch, 1]
  query_idxs_offset = jnp.expand_dims(query_idxs_offset, axis=-1)
  # [batch, num_latents]
  query_idxs += query_idxs_offset

  # [batch, input_length]
  key_idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
  # [batch, 1, num_latents, input_length]
  return dense_attention.make_attention_mask(
      query_idxs, key_idxs, jnp.greater_equal, dtype=dtype)


def make_decoder_mask(
    decoder_target_tokens: Array,
    sequence_lengths: Array,
    num_latents: int,
    dtype: DType,
    decoder_causal_attention: Optional[Array] = None) -> Array:
  """Compute the self-attention mask for a decoder.

  Same as dense_attention.make_decoder_mask, but includes slicing to create the
  correct mask size for Perceiver AR usage.

  Args:
    decoder_target_tokens: decoder output tokens. [batch, length]
    sequence_lengths: Input sequence lengths.
    num_latents: Number of Perceiver AR latents.
    dtype: dtype of the output mask.
    decoder_causal_attention: a binary mask indicating which position should
      only attend to earlier positions in the sequence. Others will attend
      bidirectionally. [batch, length]

  Returns:
    the combined decoder mask.
  """
  masks = []
  # The same mask is applied to all attention heads. So the head dimension is 1,
  # i.e., the mask will be broadcast along the heads dim.
  # [batch, 1, num_latents, length]
  causal_mask = make_causal_mask(
      decoder_target_tokens,
      num_latents=num_latents,
      sequence_lengths=sequence_lengths,
      dtype=dtype)

  # Positions with value 1 in `decoder_causal_attention` can attend
  # bidirectionally.
  if decoder_causal_attention is not None:
    # [batch, 1, num_latents, length]
    inputs_mask = dense_attention.make_attention_mask(
        # [batch, num_latents]
        query_input=slicing.slice_sequences_vmap(
            decoder_causal_attention,
            sequence_lengths=sequence_lengths,
            num_latents=num_latents,
            axis_within_vmap=-1),
        # [batch, input_length]
        key_input=decoder_causal_attention,
        pairwise_fn=jnp.logical_and,
        dtype=dtype)
    masks.append(jnp.logical_or(causal_mask, inputs_mask).astype(dtype))
  else:
    masks.append(causal_mask)

  # Padding mask.
  masks.append(
      dense_attention.make_attention_mask(
          # [batch, num_latents]
          query_input=slicing.slice_sequences_vmap(
              decoder_target_tokens,
              sequence_lengths=sequence_lengths,
              num_latents=num_latents,
              axis_within_vmap=-1) > 0,
          # [batch, input_length]
          key_input=decoder_target_tokens > 0,
          dtype=dtype))

  return dense_attention.combine_masks(*masks, dtype=dtype)
