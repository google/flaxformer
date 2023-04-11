# Copyright 2023 Google LLC.
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

"""An implementation of memory-efficient attention.

Original version published here: https://arxiv.org/abs/2112.05682

Also known as Flash Attention: https://arxiv.org/abs/2205.14135
"""

import functools
from typing import Callable, NamedTuple, Optional

import jax
from jax import lax
from jax import numpy as jnp
from jax import random

from flaxformer.types import DType
from flaxformer.types import PRNGKey

Array = jax.Array


def _causal_bias(
    q_len: int,
    k_len: int,
    offset: Optional[int] = None,
    mask_to_bias_factor: float = 1e6,
) -> Array:
  q_idxs = lax.broadcasted_iota(dtype=jnp.int32, shape=(q_len, 1), dimension=0)
  k_idxs = lax.broadcasted_iota(dtype=jnp.int32, shape=(1, k_len), dimension=1)
  if offset is not None:
    q_idxs += offset
  inverted_mask = q_idxs < k_idxs  # broadcasts to shape (q_len, k_len)
  return inverted_mask * (-1 * mask_to_bias_factor)


def _local_causal_bias(
    q_len: int,
    k_len: int,
    query_offset: int,
    key_offset: int,
) -> Array:
  offset = query_offset - key_offset
  return _causal_bias(q_len, k_len, offset=offset)


class _AttentionSummary(NamedTuple):
  """The summary of the attention over a segment of keys and values."""

  # Sum of the values weighted by the exponentiated scores. Array of shape
  # `[batch, queries, heads, value_features]`.
  exp_values: Array
  # Sum of the exponentiated scores per query. Array of shape
  # `[batch, queries, heads]`.
  exp_scores: Array
  # Maximum score encountered per query. Array of shape
  # `[batch, queries, heads]`.
  max_score: Array


def _summarize_chunk(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array],
    precision=None,
) -> _AttentionSummary:
  """MultiQuery attention for a segment of queries, keys, and values.

  Args:
    query: An array of shape `[batch, q_length, heads, qk_depth_per_head]`.
    key: An array of shape `[batch, kv_length, qk_depth_per_head]`.
    value: An array of shape `[batch, kv_length, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch, heads, q_length, kv_length]` This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    The summary for this segment, consisting of sum of the sum of the
    values weighted by their exponentiated attention scores, the exponentiated
    attention scores, and the maximum score of this segment.
  """
  batch, q_len, q_heads, q_feat = query.shape
  del q_feat

  attn_weights = jnp.einsum('bqhd,bkd->bqhk', query, key, precision=precision)

  if bias is not None:
    bias = jnp.moveaxis(bias, 1, 2)
    attn_weights += bias

  max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
  # The stop gradient significantly speeds up the computation (up to ~30%)
  # during the backward pass.
  max_score = jax.lax.stop_gradient(max_score)
  exp_weights = jnp.exp(attn_weights - max_score)
  # We flip the location of the head dimension to match the expected order
  # of the dimensions in the output:
  exp_values = jnp.einsum(
      'bvf,bqhv->bqhf', value, exp_weights, precision=precision
  )
  return _AttentionSummary(
      exp_values,
      exp_weights.sum(axis=-1),
      max_score.reshape((batch, q_len, q_heads)),
  )


def _memory_efficient_attention(
    query,
    key,
    value,
    bias_fn: Callable[[int, int], Array],
    query_chunk_size: int,
    key_chunk_size: int,
    precision=None,
    dtype=jnp.float32,
    use_extra_logit: bool = False,
    causal_mask: bool = False,
):
  """Computes dot-product multiquery-attention given query, key, and value."""
  batch, num_q, heads, q_feat = query.shape
  batch, num_kv, k_features = key.shape
  batch, num_kv, v_features = value.shape

  num_q_chunks = num_q // query_chunk_size
  num_kv_chunks = num_kv // key_chunk_size

  query = query.reshape((batch, num_q_chunks, query_chunk_size, heads, q_feat))
  key = key.reshape((batch, num_kv_chunks, key_chunk_size, k_features))
  value = value.reshape((batch, num_kv_chunks, key_chunk_size, v_features))
  # We move the chunk_idx axis to the front to iterate over it with lax.map.
  query = jnp.moveaxis(query, 1, 0)
  key = jnp.moveaxis(key, 1, 0)
  value = jnp.moveaxis(value, 1, 0)

  input_dtype = dtype

  # The zero_chunk is the output of _summarize_chunk when the inputs are zeros.
  # We define the zero_chunk outside of the loops to prevent the compiler from
  # re-creating these arrays in every loop iteration.
  zero_chunk = _AttentionSummary(
      jnp.zeros((batch, query_chunk_size, heads, v_features)),  # exp_values
      jnp.zeros((batch, query_chunk_size, heads)),  # exp_weights
      jnp.zeros((batch, query_chunk_size, heads)),  # max_score
  )

  def _query_chunk_attention(args):
    query_chunk, query_chunk_idx = args

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def conditional_summarize_fn(args):
      key_chunk, value_chunk, key_chunk_idx = args

      skip_block = jnp.array(False)
      if causal_mask:
        skip_block = query_chunk_idx < key_chunk_idx

      def cond_fn(query, key, value, key_chunk_idx):
        with jax.named_scope('compute_bias'):
          chunk_bias = bias_fn(query_chunk_idx, key_chunk_idx)
        return _summarize_chunk(
            query, key, value, chunk_bias, precision=precision
        )

      return jax.lax.cond(
          skip_block,
          lambda a, b, c, d: zero_chunk,
          cond_fn,
          query_chunk,
          key_chunk,
          value_chunk,
          key_chunk_idx,
      )

    chunk_values, chunk_weights, chunk_max = lax.map(
        conditional_summarize_fn, xs=(key, value, jnp.arange(0, num_kv_chunks))
    )

    assert chunk_values.shape == (
        num_kv_chunks,
        batch,
        query_chunk_size,
        heads,
        v_features,
    ), chunk_values.shape

    with jax.named_scope('renormalization'):
      global_max = jnp.max(chunk_max, axis=0, keepdims=True)
      max_diffs = jnp.exp(chunk_max - global_max)
      # Add dimension to be broadcasted to v_features
      chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
      chunk_weights *= max_diffs

    # sum over key chunks
    all_values = chunk_values.sum(axis=0)
    all_weights = chunk_weights.sum(axis=0)
    if use_extra_logit:
      all_weights += jnp.exp(-global_max.reshape(global_max.shape[1:]))
    all_weights = jnp.expand_dims(all_weights, -1)
    return (all_values / all_weights).astype(input_dtype)

  res = lax.map(_query_chunk_attention, xs=(query, jnp.arange(0, num_q_chunks)))

  assert res.shape == (
      num_q_chunks,
      batch,
      query_chunk_size,
      heads,
      v_features,
  )
  res = jnp.moveaxis(res, 0, 1)
  return res.reshape(batch, num_q, heads, value.shape[-1])


def dot_product_attention_multiquery(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array] = None,
    broadcast_dropout: bool = True,
    rescale_logits: bool = False,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    enable_dropout: bool = True,
    dtype: DType = jnp.float32,
    precision: Optional[lax.Precision] = None,
    use_extra_logit: bool = False,
    float32_logits: bool = False,
    causal_mask: bool = False,
    query_chunk_size: int = 1024,
    key_chunk_size: int = 2048,
) -> Array:
  """Computes dot-product multiquery-attention given query, key, and value.

  This is a variant of the multi-head dot product attention introduced in
  https://arxiv.org/abs/1706.03762 and implemented in `dot_product_attention`.
  In this function, the key and the value have 1 head whereas query has 1 or
  more heads. This variant is called "multi-query" attention.

  This function is improved by the memory-efficient attention algorithm
  (https://arxiv.org/abs/2112.05682), which is also called FlashAttention
  (https://arxiv.org/abs/2205.14135).

  Note: query, key, value needn't have any batch dimensions.

  Args:
    query: queries for calculating attention with shape of `[batch..., q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch..., kv_length,
      qk_depth_per_head]`.
    value: values to be used in attention with shape of `[batch..., kv_length,
      v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]` This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    rescale_logits: bool. Whether to rescale `query` logits by 1/sqrt(depth_kq).
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    enable_dropout: bool, whether to apply dropout
    dtype: the dtype of the computation (default: float32)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    use_extra_logit: whether to include a virtual extra logit equal to zero.
    float32_logits: bool, if True then compute logits in float32 to avoid
      numerical issues with bfloat16.
    causal_mask: Apply a causal mask. This can be used alternatively or in
      addition to the given bias.
    query_chunk_size: Positive integer to control the size of the query chunks.
    key_chunk_size: Positive integer to control the size of the key chunks.

  Returns:
    Output of shape `[batch..., length, num_heads, v_depth_per_head]`.
  """
  assert (
      key.ndim == value.ndim
  ), f'k, v must have same rank. key: {key.shape}, value: {value.shape}'
  assert (
      query.shape[:-3] == key.shape[:-2] == value.shape[:-2]
  ), f'q, k, v batch dims must match. query: {query.shape}'

  assert key.shape[-2] == value.shape[-2], 'k, v lengths must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # Ensure that we have exactly one batch dimension
  query = query.reshape(-1, *query.shape[-3:])
  key = key.reshape(-1, *key.shape[-2:])
  value = value.reshape(-1, *value.shape[-2:])

  batch_size, query_length, heads, _ = query.shape
  _, key_length, _ = key.shape

  assert query_length % query_chunk_size == 0 or query_length < query_chunk_size
  assert key_length % key_chunk_size == 0 or key_length < key_chunk_size

  query_chunk_size = min(query_chunk_size, query_length)
  key_chunk_size = min(key_chunk_size, key_length)

  if bias is not None:
    broadcastable_to = (
        batch_size,
        heads,
        query_length,
        key_length,
    )
    # Check that bias is broadcastable as expected:
    for bias_dim, broadcast_dim in zip(bias.shape, broadcastable_to):
      if bias_dim not in [1, broadcast_dim]:
        raise ValueError(
            f'Expected bias dimensions {bias.shape} to be broadcastable to'
            f' {broadcastable_to}.'
        )

  if enable_dropout and dropout_rate > 0.0:
    # Precompute dropout
    if broadcast_dropout:
      # We mimick the semantics of T5 and broadcast along the "length" dim.
      d_shape = (batch_size, heads, 1, key_length)
    else:
      d_shape = (batch_size, heads, query_length, key_length)
      # Here we deviate from the cheat and always broadcast dropout along the
      # batch and head dims.
    precomputed_dropout = random.bernoulli(dropout_rng, dropout_rate, d_shape)

  def bias_fn(
      query_chunk_idx: int,
      key_chunk_idx: int,
  ) -> Array:
    query_offset = query_chunk_idx * query_chunk_size
    key_offset = key_chunk_idx * key_chunk_size

    local_bias = jnp.zeros((1, 1, 1, 1))
    if bias is not None:
      # If bias is not broadcasted yet, dynamic slice would fail with full slice
      # size. In this case we keep the bias unbroadcasted.
      slice_q_len = min(bias.shape[-2], query_chunk_size)
      slice_k_len = min(bias.shape[-1], key_chunk_size)
      local_bias = lax.dynamic_slice(
          bias,
          # query_offset and key_offset might be > 1 but bias dims might
          # not yet be broadcasted. We rely on the protection against
          # out-of-bounds array accesses built into dynamic_slice.
          start_indices=(0, 0, query_offset, key_offset),
          slice_sizes=(*bias.shape[:2], slice_q_len, slice_k_len),
      )
    if causal_mask:
      causal = _local_causal_bias(
          query_chunk_size, key_chunk_size, query_offset, key_offset
      )
      local_bias += causal.reshape(1, 1, *causal.shape)
    # We implement dropout as part of the bias, which is additive to the
    # attention scores. In other implementations it is compute as a
    # multiplicative factor applied to the probabilities after softmax.
    if enable_dropout and dropout_rate > 0.0:
      with jax.named_scope('dropout'):
        # If dropout is not broadcasted yet, we need the collapsed dims.
        slice_q_len = min(precomputed_dropout.shape[-2], query_chunk_size)
        slice_k_len = min(precomputed_dropout.shape[-1], key_chunk_size)
        dropout_slice = lax.dynamic_slice(
            precomputed_dropout,
            # query_offset and key_offset might be > 1 but dropout dims might
            # not yet be broadcasted. We rely on the protection against
            # out-of-bounds array accesses built into dynamic_slice.
            start_indices=(0, 0, query_offset, key_offset),
            slice_sizes=(
                *precomputed_dropout.shape[:2],
                slice_q_len,
                slice_k_len,
            ),
        )
        local_bias -= dropout_slice * 1e6
    return local_bias

  # NOTE: T5 does not explicitly rescale the attention logits by
  #       1/sqrt(depth_kq)!  This is folded into the initializers of the
  #       linear transformations, which is equivalent under Adafactor.
  if rescale_logits:
    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)

  # Casting logits and softmax computation for float32 for model stability.
  if float32_logits:
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)

  result = _memory_efficient_attention(
      query,
      key,
      value,
      bias_fn,
      query_chunk_size=query_chunk_size,
      key_chunk_size=key_chunk_size,
      precision=precision,
      dtype=dtype,
      use_extra_logit=use_extra_logit,
      causal_mask=causal_mask,
  )
  result = result.reshape(batch_size, *result.shape[1:])
  return result
