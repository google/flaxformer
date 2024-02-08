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

"""Dense attention classes and mask/weighting functions."""

# pylint: disable=attribute-defined-outside-init,g-bare-generic

import abc
import functools
from typing import Callable, Optional, Tuple, Union

from aqt.jax_legacy.jax import flax_layers as aqt_flax_layers
from aqt.jax_legacy.jax import quant_config as aqt_config
from aqt.jax_legacy.jax import quantization as aqt
import chex
from flax import linen as nn
from flax.core import variables
from flax.linen import initializers
from flax.linen import partitioning as flax_partitioning
from flax.linen.linear import default_kernel_init
from flax.training import common_utils
import jax
from jax import lax
from jax import random
import jax.numpy as jnp

from flaxformer import activation_partitioning
from flaxformer.components import dense
from flaxformer.components import embedding
from flaxformer.types import Array
from flaxformer.types import DType
from flaxformer.types import Initializer
from flaxformer.types import PRNGKey


RulesFallback = flax_partitioning.RulesFallback


def _softmax_with_extra_logit(
    x: Array,
    axis: Optional[Union[int, Tuple[int, ...]]] = -1,
) -> Array:
  """Softmax function with an additional virtual logit equal to zero.

  For compatibility with some previously trained models.

  This is equivalent to adding one to the denominator.
  In the context of attention, it allows you to attend to nothing.

  Args:
    x: input to softmax
    axis: the axis or axes along which the softmax should be computed. Either an
      integer or a tuple of integers.

  Returns:
    A tensor with the same shape as x.
  """
  m = jnp.maximum(lax.stop_gradient(x.max(axis, keepdims=True)), 0)
  unnormalized = jnp.exp(x - m)
  # After shift, extra logit is -m. Add exp(-m) to denominator
  denom = unnormalized.sum(axis, keepdims=True) + jnp.exp(-m)
  return unnormalized / denom


# ------------------------------------------------------------------------------
# Fast attention layers.
# ------------------------------------------------------------------------------


def dot_product_attention_weights(
    query: Array,
    key: Array,
    bias: Optional[Array] = None,
    broadcast_dropout: bool = True,
    rescale_logits: bool = False,
    rescale_weights: bool = False,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    enable_dropout: bool = True,
    dtype: DType = jnp.float32,
    precision: Optional[lax.Precision] = None,
    use_extra_logit: bool = False,
    float32_logits: bool = False,
) -> Array:
  """Computes dot-product attention weights given query and key.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key.

  Note: query and key needn't have any batch dimensions.

  Args:
    query: queries for calculating attention with shape of `[batch..., q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch..., kv_length,
      num_heads, qk_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]` This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    rescale_logits: bool. Whether to rescale `query` logits by 1/sqrt(depth_kq).
    rescale_weights: bool. Whether to rescale attention weights by
      1/sqrt(depth_kq). This is applied before bias and softmax.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    enable_dropout: bool, whether to apply dropout
    dtype: the dtype of the computation (default: float32)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    use_extra_logit: whether to include a virtual extra logit equal to zero.
    float32_logits: bool, if True then compute logits in float32 to avoid
      numerical issues with bfloat16.

  Returns:
    Attention weights of shape `[batch..., num_heads, q_length, kv_length]`.
  """
  assert query.ndim == key.ndim, 'q, k must have same rank.'
  assert query.shape[:-3] == key.shape[:-3], 'q, k batch dims must match.'
  assert query.shape[-2] == key.shape[-2], 'q, k num_heads must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'
  if rescale_logits and rescale_weights:
    raise ValueError(
        'Only one of rescale_logits or rescale_weights may be True.'
    )

  depth = query.shape[-1]

  # Calculate attention matrix.
  # NOTE: T5 does not explicitly rescale the attention logits by
  #       1/sqrt(depth_kq)!  This is folded into the initializers of the
  #       linear transformations, which is equivalent under Adafactor.
  if rescale_logits:
    query = query / jnp.sqrt(depth).astype(dtype)

  # Casting logits and softmax computation for float32 for model stability.
  if float32_logits:
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)

  # `attn_weights` shape is (batch..., num_heads, q_length, kv_length)
  attn_weights = jnp.einsum(
      '...qhd,...khd->...hqk', query, key, precision=precision
  )
  if rescale_weights:
    attn_weights = attn_weights / jnp.sqrt(depth).astype(dtype)

  # Apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias.astype(attn_weights.dtype)

  # Normalize the attention weights.
  attn_weights = (
      _softmax_with_extra_logit if use_extra_logit else jax.nn.softmax
  )(attn_weights).astype(dtype)

  # Apply attention dropout.
  if enable_dropout and dropout_rate > 0.0:
    keep_prob = 1.0 - dropout_rate
    if broadcast_dropout:
      # T5 broadcasts along the "length" dim, but unclear which one that
      # corresponds to in positional dimensions here, assuming query dim.
      dropout_shape = list(attn_weights.shape)
      dropout_shape[-2] = 1
      keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
      keep = jnp.broadcast_to(keep, attn_weights.shape)
    else:
      keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
    multiplier = keep.astype(attn_weights.dtype) / jnp.asarray(
        keep_prob, dtype=dtype
    )
    attn_weights = attn_weights * multiplier

  return attn_weights


def apply_dot_product_attention_weights_to_values(
    attention_weights: Array,
    value: Array,
    precision: Optional[lax.Precision] = None,
) -> Array:
  """Applies the attention weights to the values.

  Args:
    attention_weights: The attention weights, e.g., computed by
      dot_product_attention_weights.
    value: The values to apply the attention to.
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    The weighted sum over values for each query position.
  """
  return jnp.einsum(
      '...hqk,...khd->...qhd', attention_weights, value, precision=precision
  )


def dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array] = None,
    broadcast_dropout: bool = True,
    rescale_logits: bool = False,
    rescale_weights: bool = False,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    enable_dropout: bool = True,
    dtype: DType = jnp.float32,
    precision: Optional[lax.Precision] = None,
    use_extra_logit: bool = False,
    float32_logits: bool = False,
):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Note: query, key, value needn't have any batch dimensions.

  Args:
    query: queries for calculating attention with shape of `[batch..., q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch..., kv_length,
      num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of `[batch..., kv_length,
      num_heads, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]` This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    rescale_logits: bool. Whether to rescale `query` logits by 1/sqrt(depth_kq).
    rescale_weights: bool. Whether to rescale attention weights by
      1/sqrt(depth_kq). This is applied before bias and softmax.
   dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    enable_dropout: bool, whether to apply dropout
    dtype: the dtype of the computation (default: float32)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    use_extra_logit: whether to include a virtual extra logit equal to zero.
    float32_logits: bool, if True then compute logits in float32 to avoid
      numerical issues with bfloat16.

  Returns:
    Output of shape `[batch..., length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert (
      query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
  ), 'q, k, v batch dims must match.'
  assert (
      query.shape[-2] == key.shape[-2] == value.shape[-2]
  ), 'q, k, v num_heads must match.'
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  attn_weights = dot_product_attention_weights(
      query,
      key,
      bias=bias,
      broadcast_dropout=broadcast_dropout,
      rescale_logits=rescale_logits,
      rescale_weights=rescale_weights,
      dropout_rng=dropout_rng,
      dropout_rate=dropout_rate,
      enable_dropout=enable_dropout,
      dtype=dtype,
      precision=precision,
      use_extra_logit=use_extra_logit,
      float32_logits=float32_logits,
  )

  return apply_dot_product_attention_weights_to_values(
      attn_weights, value, precision=precision
  )


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
) -> Array:
  """Computes dot-product multiquery-attention given query, key, and value.

  This is a variant of the multi-head dot product attention introduced in
  https://arxiv.org/abs/1706.03762 and implemented in `dot_product_attention`.
  In this function, the key and the value have 1 head whereas query has 1 or
  more heads. This variant is called "multi-query" attention.

  It calculates the attention weights given query and key and combines the
  values using the attention weights.

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

  # calculate attention matrix
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

  # attn weight shape is (batch..., num_heads, q_length, kv_length)
  attn_weights = jnp.einsum(
      '...qhd,...kd->...hqk', query, key, precision=precision
  )

  # apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias.astype(attn_weights.dtype)

  # normalize the attention weights
  attn_weights = (
      _softmax_with_extra_logit if use_extra_logit else jax.nn.softmax
  )(attn_weights).astype(dtype)

  # apply attention dropout
  if enable_dropout and dropout_rate > 0.0:
    keep_prob = 1.0 - dropout_rate
    if broadcast_dropout:
      # T5 broadcasts along the "length" dim, but unclear which one that
      # corresponds to in positional dimensions here, assuming query dim.
      dropout_shape = list(attn_weights.shape)
      dropout_shape[-2] = 1
      keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
      keep = jnp.broadcast_to(keep, attn_weights.shape)
    else:
      keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
    multiplier = keep.astype(attn_weights.dtype) / jnp.asarray(
        keep_prob, dtype=dtype
    )
    attn_weights = attn_weights * multiplier

  # return weighted sum over values for each query position
  return jnp.einsum(
      '...hqk,...kd->...qhd', attn_weights, value, precision=precision
  )


class DenseAttention(metaclass=abc.ABCMeta):
  """API for attention classes that compute a full key-query attention matrix.

  This allows for 2D matrices masking or re-weighting the attention between
  specific key/query pairs.
  """

  @abc.abstractmethod
  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      mask: Optional[Array] = None,
      bias: Optional[Array] = None,
      *,
      precomputed_qkv: Optional[Array] = None,
      decode: bool = False,
      enable_dropout: bool = True,
  ) -> Array:
    """Applies attention on the input data.

    Args:
      inputs_q: input queries of shape `[batch_sizes..., q_length, q_features]`.
      inputs_kv: key/values of shape `[batch_sizes..., kv_length, kv_features]`.
      mask: attention mask of shape `[batch_sizes..., num_heads, q_length,
        kv_length]`.
      bias: attention bias of shape `[batch_sizes..., num_heads, q_length,
        kv_length]`.
      precomputed_qkv: when using fused implementations QKVO are defined outside
        this module and we only use the module to run computations.
      decode: Whether to prepare and use an autoregressive cache.
      enable_dropout: Enables dropout if set to True.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """


class MultiHeadDotProductAttention(nn.Module, DenseAttention):
  """Multi-head dot-product attention.

  Attributes:
    num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    use_bias: bool: whether pointwise QKVO dense transforms use bias.
    dtype: the dtype of the computation (default: float32)
    qkv_features: dimension of the key, query, and value.
    head_dim: dimension of each head. If unspecified, it defaults to
      qkv_features // num_heads.
    out_features: dimension of the last projection
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rate: dropout rate
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the kernel of the Dense layers.
    qkv_kernel_init: optional initializer for the fused qkv kernel. If None,
      kernel_init will be used instead.
    kv_kernel_init: optional initializer for the fused kv kernel. If None,
      kernel_init will be used instead.
    q_kernel_init: optional initializer for the query (q) kernel. If None,
      kernel_init will be used instead.
    bias_init: initializer for the bias of the Dense layers.
    attention_fn: dot_product_attention or compatible function. Accepts query,
      key, value, and returns output of shape `[bs, dim1, dim2, ..., dimN,,
      num_heads, value_channels]``
    use_extra_logit: whether to include a virtual extra logit equal to zero.
    float32_logits: bool, if True then compute logits in float32 to avoid
      numerical issues with bfloat16.
    output_projection: Project the output of `attention_fn` to `out_features`.
      If False, returns the output of `attention_fn` without a projection.
    sow_intermediates: whether to track intermediates using Module.sow.
    split_head_kernel: whether to store QKVO variables with a split head
      dimension.
    kernels_to_fuse: Which kernels to fuse, if any.
    use_rotary_embedding: whether to use rotary embeddings.
  """

  num_heads: int
  use_bias: bool
  dtype: DType = jnp.float32
  qkv_features: Optional[int] = None
  head_dim: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.0
  precision: Optional[lax.Precision] = None
  kernel_init: Initializer = (
      default_kernel_init  # pytype: disable=annotation-type-mismatch  # jax-types
  )
  qkv_kernel_init: Optional[Initializer] = None
  kv_kernel_init: Optional[Initializer] = None
  q_kernel_init: Optional[Initializer] = None
  bias_init: Initializer = initializers.zeros
  rescale_logits: bool = False
  attention_fn: Callable[[Array, Array, Array], Array] = staticmethod(
      dot_product_attention
  )
  use_extra_logit: bool = False
  float32_logits: bool = False
  output_projection: bool = True
  # TODO: Remove out_features and output_projection.
  sow_intermediates: bool = False
  split_head_kernel: bool = False
  kernels_to_fuse: Optional[str] = None
  use_rotary_embedding: bool = False
  rotary_embedding_max_timescale: float = 1e4
  # Whether to shard over the head dimension, setting this to False when the
  # number of heads is not divisible your activation num_partitions
  sharding_over_head_dimension: bool = True
  q_conv: Optional[nn.Module] = None
  k_conv: Optional[nn.Module] = None
  v_conv: Optional[nn.Module] = None
  dense_general_factory: Callable[..., nn.Module] = dense.DenseGeneral

  def update_cache_prefill(
      self,
      key: Array,
      value: Array,
      cached_key: variables.Variable,
      cached_value: variables.Variable,
      cache_index: variables.Variable,
      prefill_lengths: Array,
  ) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """Update the autoregressive cache for multiple timesteps at once.

    This is useful for things like a prefix-lm where the encoder section of the
    input is visible bidirectionally. The key and value for this section need to
    be computed in a single shot, as a step by step approach would result in
    causal attention.

    Args:
      key: The calculated key used in attention. [batch..., length, num_heads,
        features_per_head]
      value: The calculated value used in attention. [batch..., length,
        num_heads, features_per_head]
      cached_key: The cache of previous keys. [batch..., num_heads,
        features_per_head, length]
      cached_value: The cache of previous values. [batch..., num_heads,
        features_per_head, length]
      cache_index: The timestep that we are currently calculating the key and
        value for. [batch]
      prefill_lengths: The number of timesteps we should fill in the cache.
        [batch]

    Returns:
      The key, value, and the last timestep we just filled in the cache.
      We also return the new cache values for now because assigning to a
      variable inside of a method doesn't work. These returns will be removed
      eventually.
    """
    # Make a reference to the data underlaying the variable for ease of
    # use.
    cache_index.value = prefill_lengths
    # Note, the cache index is now a vector
    # of batch size so that each example can start just after it's
    # prefix which can be different lengths for different examples.
    cur_index = cache_index.value
    # Move the sequence dimension to the end to match the cache shapes.
    key_cached = jnp.moveaxis(key, -3, -1)
    value_cached = jnp.moveaxis(value, -3, -1)
    # Reshape the index so the batch is at the beginning, default
    # broadcasting behavior is to add singleton dims to the front but
    # we need them at the end.
    batch_first_index = jnp.reshape(
        cur_index, (-1,) + tuple(1 for _ in range(cached_key.value.ndim - 1))
    )
    # Calculate a mask that will set any position past the prefix to zero
    # when applied to the key.
    key_mask = (
        lax.broadcasted_iota(
            jnp.int32, cached_key.value.shape, cached_key.value.ndim - 1
        )
        < batch_first_index
    )
    value_mask = (
        lax.broadcasted_iota(
            jnp.int32, cached_value.value.shape, cached_value.value.ndim - 1
        )
        < batch_first_index
    )
    # Set the caches with the calculated key and values but hide anything
    # past the prefix.
    cached_key_value = key_cached * key_mask
    cached_value_value = value_cached * value_mask
    return (
        key,
        value,
        cur_index,
        cached_key_value,
        cached_value_value,
        prefill_lengths,
    )

  def update_cache_decode(
      self,
      key: Array,
      value: Array,
      cached_key: variables.Variable,
      cached_value: variables.Variable,
      cache_index: variables.Variable,
  ) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """Update the next timestep in the autoregressive cache.

    This is used during step by step decoding where each key and value we get
    are a single (the next) timestep.

    Args:
      key: The calculated key used in attention. [batch..., 1, num_heads,
        features_per_head]
      value: The calculated value used in attention. [batch..., 1, num_heads,
        features_per_head]
      cached_key: The cache of previous keys. [batch..., num_heads,
        features_per_head, length]
      cached_value: The cache of previous values. [batch..., num_heads,
        features_per_head, length]
      cache_index: The timestep that we are currently calculating the key and
        value for. [batch] if we are decoding after doing a prefill or [1] if we
        are starting with step-by-step decoding.

    Returns:
      The key, value, and the last timestep we just filled in the cache. Note:
      this index is the last timestep we just fill, the actual value of the
      `cache_index` is already increased to point to the next timestep to fill.
      We also return the new cache values for now because assigning to a
      variable inside of a method doesn't work. These returns will be removed
      eventually.
    """
    cache_length = cached_key.value.shape[-1]
    # Create a OHE of the current index. NOTE: the index is increased
    # below.
    # Note: We reshape the index into a column vector so that it will work
    # if the index is a scalar or a vector with different cache positions
    # from different elements in a batch.
    cur_index = jnp.reshape(cache_index.value, (-1,))
    one_hot_indices = jax.nn.one_hot(cur_index, cache_length, dtype=key.dtype)
    # In order to update the key, value caches with the current key and
    # value, we move the length axis to the back, similar to what we did
    # for the cached ones above.
    # Note these are currently the key and value of a single position,
    # since we feed one position at a time.
    one_token_key = jnp.moveaxis(key, -3, -1)
    one_token_value = jnp.moveaxis(value, -3, -1)
    # The one hot indices are now either [1, length] for a scalar index or
    # [batch size, length] for examples where there are different lengths
    # of prefixes. We need to add dims for num_heads and num_features as
    # broadcasting doesn't work for the batched version.
    one_hot_indices = jnp.expand_dims(
        jnp.expand_dims(one_hot_indices, axis=1), axis=1
    )
    # Update key, value caches with our new 1d spatial slices.
    # We implement an efficient scatter into the cache via one-hot
    # broadcast and addition.
    # Key/Value have seq lengths of 1 while one_hot has a seq_length
    # of length. key/value will broadcast their value to each timestep
    # and the onehot will mask all but the correct timesteps.
    key = cached_key.value + one_token_key * one_hot_indices
    value = cached_value.value + one_token_value * one_hot_indices
    cached_key_value = key
    cached_value_value = value
    cache_index_value = cache_index.value + 1
    # Move the keys and values back to their original shapes.
    key = jnp.moveaxis(key, -1, -3)
    value = jnp.moveaxis(value, -1, -3)
    return (
        key,
        value,
        cur_index,
        cached_key_value,
        cached_value_value,
        cache_index_value,
    )

  @nn.compact
  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      mask: Optional[Array] = None,
      bias: Optional[Array] = None,
      *,
      precomputed_qkv: Optional[Array] = None,
      decode: bool = False,
      enable_dropout: bool = True,
      prefill: bool = False,
      prefill_lengths: Optional[Array] = None,
  ) -> Array:
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    There are two modes: decoding and non-decoding (e.g., training). The mode is
    determined by `decode`.

    During decoding mode, this method is called twice, by `init` and
    `apply`. In the former, inputs_q: [batch..., length, qkv_features] and
    inputs_kv: [batch..., length, qkv_features]

    During apply, query, key and value all have the shape: [batch * beam, 1,
    qkv_features] where the batch dimension is added to include multiple beams.
    Note that the batch dimension is different during the init and apply calls.
    This is because the cached variables are directly passed-in during `apply`
    method. In other words, the cache variables such as `cached_key` are
    initialized with `batch` dim, expanded by tiling in the beam search function
    to `batch * beam` dimension, and passed to the `apply` method as part of a
    variable dict.

    Args:
      inputs_q: input queries of shape `[batch_sizes..., q_length, q_features]`.
      inputs_kv: key/values of shape `[batch_sizes..., kv_length, kv_features]`.
      mask: attention mask of shape `[batch_sizes..., num_heads, q_length,
        kv_length]`.
      bias: attention bias of shape `[batch_sizes..., num_heads, q_length,
        kv_length]`.
      precomputed_qkv: when using fused implementations QKVO are defined outside
        this module and we only use the module to run computations.
      decode: Whether to prepare and use an autoregressive cache.
      enable_dropout: Enables dropout if set to True.
      prefill: Whether to run a partial sequence to prefill the cache.
      prefill_lengths: The length of each partial sequence we are filling in the
        cache, lengths are inferred from the mask if not provided.

    Returns:
      If output_projection is True, then output of shape
      `[batch_sizes..., length, out_features]`, where out_features is set to
      features if not provided. If output_projection is False, then output of
      shape `[batch_sizes..., length, num_heads, head_dim]`.
    """
    validate_dense_attention_call_parameter_shapes(
        inputs_q, inputs_kv, mask, bias, self.num_heads
    )

    qkv_kernel_init = (
        self.qkv_kernel_init
        if self.qkv_kernel_init is not None
        else self.kernel_init
    )
    kv_kernel_init = (
        self.kv_kernel_init
        if self.kv_kernel_init is not None
        else self.kernel_init
    )
    q_kernel_init = (
        self.q_kernel_init
        if self.q_kernel_init is not None
        else self.kernel_init
    )

    if precomputed_qkv is not None:
      raise ValueError('Support for precomputed QKVO not implemented.')

    rotary_index = None
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    if self.head_dim is None:
      head_dim = qkv_features // self.num_heads
    else:
      head_dim = self.head_dim

    if self.kernels_to_fuse and not self.split_head_kernel:
      raise ValueError(
          'Un-reshaped kernels are required when using QKV fused '
          'kernel optimization.'
      )

    # Is attention logit rescaling explicit or folded into initializer?
    if self.rescale_logits:
      query_init = q_kernel_init
    else:
      if self.kernels_to_fuse:
        raise ValueError(
            'Cannot fold in logit normalization to query '
            'initializer when using fused kernels.'
        )
      depth_scaling = jnp.sqrt(head_dim).astype(self.dtype)
      query_init = lambda *args: q_kernel_init(*args) / depth_scaling

    make_dense = functools.partial(
        self.dense_general_factory,
        axis=-1,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        precision=self.precision,
        reshape_kernel=not self.split_head_kernel,
    )

    # Project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, num_heads, features_per_head]
    if self.kernels_to_fuse is None:
      query = make_dense(
          kernel_init=query_init,
          features=(self.num_heads, head_dim),
          kernel_axis_names=['embed', 'heads', 'kv'],
          name='query',
      )(inputs_q)
      key = make_dense(
          kernel_init=self.kernel_init,
          features=(self.num_heads, head_dim),
          kernel_axis_names=['embed', 'heads', 'kv'],
          name='key',
      )(inputs_kv)
      value = make_dense(
          kernel_init=self.kernel_init,
          features=(self.num_heads, head_dim),
          kernel_axis_names=['embed', 'heads', 'kv'],
          name='value',
      )(inputs_kv)
    # TODO: should we fuse/slice along depth or head dim?
    elif self.kernels_to_fuse == 'qkv':
      if inputs_q is not inputs_kv:
        raise ValueError(
            'qkv fusion is only supported in self-attention mode '
            '(when inputs_q is inputs_kv).'
        )
      # 'qkv' fusion mode implies self-attention
      qkv = make_dense(
          kernel_init=qkv_kernel_init,
          features=(3, self.num_heads, head_dim),
          kernel_axis_names=['embed', 'stack', 'heads', 'kv'],
          name='qkv_fused',
      )(inputs_q)
      query = jnp.squeeze(lax.dynamic_slice_in_dim(qkv, 0, 1, -3), -3)
      key = jnp.squeeze(lax.dynamic_slice_in_dim(qkv, 1, 1, -3), -3)
      value = jnp.squeeze(lax.dynamic_slice_in_dim(qkv, 2, 1, -3), -3)
    elif self.kernels_to_fuse == 'kv':
      query = make_dense(
          kernel_init=query_init,
          features=(self.num_heads, head_dim),
          kernel_axis_names=['embed', 'heads', 'kv'],
          name='query',
      )(inputs_q)
      kv = make_dense(
          kernel_init=kv_kernel_init,
          features=(2, self.num_heads, head_dim),
          kernel_axis_names=['embed', 'stack', 'heads', 'kv'],
          name='kv_fused',
      )(inputs_kv)
      key = jnp.squeeze(lax.dynamic_slice_in_dim(kv, 0, 1, -3), -3)
      value = jnp.squeeze(lax.dynamic_slice_in_dim(kv, 1, 1, -3), -3)
    else:
      raise ValueError('Incorrect kernel fusion mode specified.')

    # Multi Dconv Head Attention options:
    if self.q_conv is not None:
      query = self.q_conv(  # pylint: disable=not-callable
          query, decode=decode, prefill=prefill, prefill_lengths=prefill_lengths
      )
    if self.k_conv is not None:
      key = self.k_conv(  # pylint: disable=not-callable
          key, decode=decode, prefill=prefill, prefill_lengths=prefill_lengths
      )
    if self.v_conv is not None:
      value = self.v_conv(  # pylint: disable=not-callable
          value, decode=decode, prefill=prefill, prefill_lengths=prefill_lengths
      )

    if self.sharding_over_head_dimension:
      # Note: We don't use `activation_partitioning.with_sharding_migration`
      # here because we do often want this 2D sharded. However, if rules are
      # valid, they should result in 2D sharding. We don't need to raise errors
      # if both result in 2D sharding (which with_sharding_migration does).
      if flax_partitioning.get_axis_rules():
        query = flax_partitioning.with_sharding_constraint(
            query, ('batch', 'length', 'heads', 'kv')
        )
        key = flax_partitioning.with_sharding_constraint(
            key, ('batch', 'length', 'heads', 'kv')
        )
        value = flax_partitioning.with_sharding_constraint(
            value, ('batch', 'length', 'heads', 'kv')
        )
      else:
        query = activation_partitioning.with_sharding(query, 2)
        key = activation_partitioning.with_sharding(key, 2)
        value = activation_partitioning.with_sharding(value, 2)

    query: Array = query  # hint to quiet pytype.
    key: Array = key
    value: Array = value

    if prefill and decode:
      raise ValueError(
          'prefill and decode cannot both be true at the same'
          'time. If you are using a prefix LM with bidirectional '
          'attention on the inputs, please make a call with '
          'prefill=True that includes an attention mask that '
          'covers your inputs first and then make your decoding '
          'calls.'
      )
    if prefill or decode:
      # Detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      # The key and value have dimension
      # [batch..., length, num_heads, features_per_head], but we cache them as
      # [batch..., num_heads, features_per_head, length] as a TPU fusion
      # optimization. This also enable the "scatter via one-hot broadcast"
      # trick, which means we do a one-hot broadcast instead of a scatter/gather
      # operations, which gives a 3-4x speedup in practice.
      swap_dims = lambda x: x[:-3] + tuple(x[i] for i in [-2, -1, -3])
      cached_key = self.variable(
          'cache', 'cached_key', jnp.zeros, swap_dims(key.shape), key.dtype
      )
      cached_value = self.variable(
          'cache',
          'cached_value',
          jnp.zeros,
          swap_dims(value.shape),
          value.dtype,
      )
      cache_index = self.variable(
          'cache', 'cache_index', lambda: jnp.array(0, dtype=jnp.int32)
      )
      rotary_index = cache_index.value
      if is_initialized:
        # Here we are in "apply()".
        *batch_dims, num_heads, features_per_head, length = (
            cached_key.value.shape
        )
        if prefill:
          if prefill_lengths is None:
            # Figure out how far each element in the batch fills the cache based
            # on the mask. We index each element in the batch, the first head
            # dim (because this is always set to one), and the first query
            # vector. If there is any prefix at all, the first element in the
            # prefix would be part of it.
            prefill_lengths = jnp.sum(mask[:, 0, 0, :], axis=-1).astype(
                cache_index.value.dtype
            )
          (
              key,
              value,
              cur_index,
              cached_key_value,
              cached_value_value,
              cache_index_value,
          ) = self.update_cache_prefill(
              key, value, cached_key, cached_value, cache_index, prefill_lengths
          )
        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        elif decode:
          # Check the shape of the cached key against the input query.
          expected_shape = tuple(batch_dims) + (1, num_heads, features_per_head)
          if expected_shape != query.shape:
            raise ValueError(
                'Autoregressive cache shape error, '
                'expected query shape %s instead got %s.'
                % (expected_shape, query.shape)
            )
          (
              key,
              value,
              cur_index,
              cached_key_value,
              cached_value_value,
              cache_index_value,
          ) = self.update_cache_decode(
              key, value, cached_key, cached_value, cache_index
          )
          # Enforcing the Causal mask over previous positions and selecting only
          # the bias value for the current index is only needed during decode
          # mode where a single example is feed at a time. In prefill mode we
          # uses these as provided, that same way it is done in a normal forward
          # pass, like when computing logits during training.

          # Causal mask for cached decoder self-attention: our single query
          # position should only attend to those key positions that have already
          # been generated and cached, not the remaining zero elements.

          # (1, 1, length) represent (head dim, query length, key length)
          # query length is 1 because during decoding we deal with one
          # index.
          # The same mask is applied to all batch elements and heads.
          #
          # Add trailing dims to the current index so it can either
          # broadcast over the batch dim or it can just be batch size.
          mask = combine_masks(
              mask,
              jnp.broadcast_to(
                  jnp.arange(length), tuple(batch_dims) + (1, 1, length)
              )
              <= jnp.reshape(cur_index, (-1, 1, 1, 1)),
          )
          # Grab the correct relative attention bias during decoding. This is
          # only required during single step decoding.
          if bias is not None:
            # The bias is a full attention matrix, but during decoding we only
            # have to take a slice of it.
            # This is equivalent to bias[..., cur_index:cur_index+1, :].
            # If we are doing prefix decoding where cur index is a vector the
            # result will be [batch, heads, 1, :]. If cur_index is a scalar
            # like in encdec decoding, the result will be [1, heads, 1, :].
            # We use a one-hot einsum rather than a slice to avoid introducing
            # a Gather op that is currently lowered poorly by SPMD passes,
            # adding expensive all-reduce and all-gather operations.

            bias = jnp.einsum(
                'bq, bhqk->bhk',
                common_utils.onehot(cur_index, num_classes=length),
                bias,
            )
            bias = jnp.expand_dims(bias, 2)

        # Currently, updating a variable inside of a method is not handled
        # in flax, so we return the actual values and assign them in the main
        # compacted call for now.
        # TODO: Move variable assignment inside of the
        # cache update functions once variable references are tracked across
        # transform boundaries.
        cache_index.value = cache_index_value
        cached_key.value = cached_key_value
        cached_value.value = cached_value_value

    # Convert the boolean attention mask to an attention bias.
    if mask is not None:
      # attention mask in the form of attention bias
      attention_bias = lax.select(
          mask > 0,
          jnp.full(mask.shape, 0.0).astype(self.dtype),
          jnp.full(mask.shape, -1e10).astype(self.dtype),
      )
    else:
      attention_bias = None

    # Add provided bias term (e.g. relative position embedding).
    if bias is not None:
      attention_bias = combine_biases(attention_bias, bias)

    dropout_rng = None
    if enable_dropout and self.dropout_rate > 0.0:
      dropout_rng = self.make_rng('dropout')

    if self.use_rotary_embedding:
      # use rotary embeddings before attention
      # https://arxiv.org/abs/2104.09864
      # TODO: Put it in a new class
      dim = query.shape[-1]
      max_length = max(query.shape[1], key.shape[1])
      sin, cos = embedding.generate_fixed_pos_embedding(
          dim, max_length, max_timescale=self.rotary_embedding_max_timescale
      )
      sin = sin.astype(self.dtype)
      cos = cos.astype(self.dtype)
      query, key = embedding.apply_rotary_embedding(
          query, key, cos, sin, decode=decode, rotary_index=rotary_index
      )

    x = self.attention_fn(
        query,
        key,
        value,
        bias=attention_bias,
        broadcast_dropout=self.broadcast_dropout,
        rescale_logits=self.rescale_logits,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        enable_dropout=enable_dropout,
        dtype=self.dtype,
        precision=self.precision,
        use_extra_logit=self.use_extra_logit,
        float32_logits=self.float32_logits,
    )  # pytype: disable=wrong-keyword-args

    # Compute attention.
    if not self.output_projection:
      return x

    # Back to the original inputs dimensions.
    out = self.dense_general_factory(
        features=features,
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        precision=self.precision,
        reshape_kernel=not self.split_head_kernel,
        kernel_axis_names=['heads', 'kv', 'embed'],
        name='out',
    )(
        x
    )  # pytype: disable=wrong-keyword-args
    return out


class MultiQueryDotProductAttention(nn.Module, DenseAttention):
  """Multi-query dot-product attention.

  This is a variant of the MultiHeadDotProductAttention. The key and the value
  have 1 head whereas query has 1 or more heads. This variant, called
  "multi-query" attention, was introduced in Shazeer 2019
  (https://arxiv.org/abs/1911.02150).

  Attributes:
    num_heads: number of attention heads for query. Features (i.e.
      inputs_q.shape[-1]) should be divisible by the number of heads.
    use_bias: bool: whether pointwise QKVO dense transforms use bias.
    dtype: the dtype of the computation (default: float32)
    qkv_features: dimension of the key, query, and value.
    head_dim: dimension of each head. If unspecified, it defaults to
      qkv_features // num_heads.
    out_features: dimension of the last projection
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rate: dropout rate
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the kernel of the Dense layers.
    q_kernel_init: optional initializer for the query (q) kernel. If None,
      kernel_init will be used instead.
    bias_init: initializer for the bias of the Dense layers.
    attention_fn: dot_product_attention or compatible function. Accepts query,
      key, value, and returns output of shape `[bs, dim1, dim2, ..., dimN,,
      num_heads, value_channels]``
    use_extra_logit: whether to use a virtual extra logit equal to zero.
    float32_logits: bool, if True then compute logits in float32 to avoid
      numerical issues with bfloat16.
    use_rotary_embedding: whether to use RoPE embeddings.
    use_aqt: whether to use aqt quantization.
    weight_params: Parameters for weight quantization.
    act_params: Parameters for acitvation quantization.
  """

  num_heads: int
  use_bias: bool
  dtype: DType = jnp.float32
  qkv_features: Optional[int] = None
  head_dim: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.0
  precision: Optional[lax.Precision] = None
  kernel_init: Initializer = (
      default_kernel_init  # pytype: disable=annotation-type-mismatch  # jax-types
  )
  q_kernel_init: Optional[Initializer] = None
  bias_init: Initializer = initializers.zeros
  rescale_logits: bool = False
  attention_fn: Callable[[Array, Array, Array], Array] = staticmethod(
      dot_product_attention_multiquery
  )
  use_extra_logit: bool = False
  float32_logits: bool = False
  use_rotary_embedding: bool = False
  rotary_embedding_max_timescale: float = 1e4
  split_head_kernel: bool = False
  q_conv: Optional[nn.Module] = None
  k_conv: Optional[nn.Module] = None
  v_conv: Optional[nn.Module] = None
  use_aqt: Optional[bool] = False
  weight_params: Optional[aqt.QuantOps.WeightParams] = None
  act_params: Optional[aqt.QuantOps.ActHParams] = None
  possibly_use_quantized_vars: bool = False

  def update_cache_prefill(
      self,
      key: Array,
      value: Array,
      cached_key: variables.Variable,
      cached_value: variables.Variable,
      cache_index: variables.Variable,
      prefill_lengths: Array,
  ) -> Tuple[
      Array,
      Array,
      Array,
      variables.Variable,
      variables.Variable,
      variables.Variable,
  ]:
    """Update the autoregressive cache for multiple timesteps at once.

    This is useful for things like a prefix-lm where the encoder section of the
    input is visible bidirectionally. The key and value for this section need to
    be computed in a single shot, as a step by step approach would result in
    causal attention.

    Args:
      key: The calculated key used in attention. [batch..., length,
        features_per_head]
      value: The calculated value used in attention. [batch..., length,
        features_per_head]
      cached_key: The cache of previous keys. [batch..., features_per_head,
        length]
      cached_value: The cache of previous values. [batch..., features_per_head,
        length]
      cache_index: The timestep that we are currently calculating the key and
        value for. [batch]
      prefill_lengths: The number of timesteps we should fill in the cache.
        [batch]

    Returns:
      The key, value, and the last timestep we just filled in the cache.
    """
    cache_index.value = prefill_lengths
    # Make a reference to the data underlaying the variable for ease of
    # use.
    cur_index = cache_index.value
    # Move the sequence dimension to the end to match the cache shapes.
    key_cached = jnp.moveaxis(key, -2, -1)
    value_cached = jnp.moveaxis(value, -2, -1)
    # Reshape the index so the batch is at the beginning, default
    # broadcasting behavior is to add singleton dims to the front but
    # we need them at the end.
    batch_first_index = jnp.reshape(
        cur_index, (-1,) + tuple(1 for _ in range(cached_key.value.ndim - 1))
    )
    # Calculate a mask that will set any position past the prefix to zero
    # when applied to the key.
    key_mask = (
        lax.broadcasted_iota(
            jnp.int32, cached_key.value.shape, cached_key.value.ndim - 1
        )
        < batch_first_index
    )
    value_mask = (
        lax.broadcasted_iota(
            jnp.int32, cached_value.value.shape, cached_value.value.ndim - 1
        )
        < batch_first_index
    )
    # Set the caches with the calculated key and values but hide anything
    # past the prefix.
    cached_key_value = key_cached * key_mask
    cached_value_value = value_cached * value_mask
    return (  # pytype: disable=bad-return-type  # jax-ndarray
        key,
        value,
        cur_index,
        cached_key_value,
        cached_value_value,
        prefill_lengths,
    )

  def update_cache_decode(
      self,
      key: Array,
      value: Array,
      cached_key: variables.Variable,
      cached_value: variables.Variable,
      cache_index: variables.Variable,
  ) -> Tuple[
      Array,
      Array,
      Array,
      variables.Variable,
      variables.Variable,
      variables.Variable,
  ]:
    """Update the next timestep in the autoregressive cache.

    This is used during step by step decoding where each key and value we get
    are a single (the next) timestep.

    Args:
      key: The calculated key used in attention. [batch..., 1,
        features_per_head]
      value: The calculated value used in attention. [batch..., 1,
        features_per_head]
      cached_key: The cache of previous keys. [batch..., features_per_head,
        length]
      cached_value: The cache of previous values. [batch..., features_per_head,
        length]
      cache_index: The timestep that we are currently calculating the key and
        value for. [batch]

    Returns:
      The key, value, and the last timestep we just filled in the cache. Note:
      this index is the last timestep we just fill, the actual value of the
      `cache_index` is already increased to point to the next timestep to fill.
    """
    cache_length = cached_key.value.shape[-1]
    # Create a OHE of the current index.
    # NOTE: the index is increased below.
    cur_index = jnp.reshape(cache_index.value, (-1,))
    one_hot_indices = jax.nn.one_hot(cur_index, cache_length, dtype=key.dtype)
    # In order to update the key, value caches with the current key and
    # value, we move the length axis to the back, similar to what we did
    # for the cached ones above.
    # Note these are currently the key and value of a single position,
    # since we feed one position at a time.
    # [batch..., length, features_per_head] -> [batch...,
    # features_per_head, length]
    one_token_key = jnp.moveaxis(key, -2, -1)
    one_token_value = jnp.moveaxis(value, -2, -1)
    # The one hot indices are now either [1, length] for a scalar index or
    # [batch size, length] for examples where there are different lengths
    # of prefixes. We need to add dims for and num_features as
    # broadcasting doesn't work for the batched version.
    one_hot_indices = jnp.expand_dims(one_hot_indices, axis=1)
    # Update key, value caches with our new 1d spatial slices.
    # We implement an efficient scatter into the cache via one-hot
    # broadcast and addition.
    key = cached_key.value + one_token_key * one_hot_indices
    value = cached_value.value + one_token_value * one_hot_indices
    cached_key_value = key
    cached_value_value = value
    cache_index_value = cache_index.value + 1
    # Move the keys and values back to their original shapes.
    key = jnp.moveaxis(key, -1, -2)
    value = jnp.moveaxis(value, -1, -2)
    return (
        key,
        value,
        cur_index,
        cached_key_value,
        cached_value_value,
        cache_index_value,
    )

  @nn.compact
  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      mask: Optional[Array] = None,
      bias: Optional[Array] = None,
      *,
      precomputed_qkv=None,
      decode: bool = False,
      enable_dropout: bool = True,
      prefill: bool = False,
      prefill_lengths: Optional[Array] = None,
  ) -> Array:
    """Applies multi-query dot product attention on the input data.

    Projects the inputs into multi-headed query and single-headed key and value
    vectors, applies dot-product attention and project the results to an output
    vector.

    Args:
      inputs_q: input queries of shape `[batch_sizes..., q_length, q_features]`.
      inputs_kv: key/values of shape `[batch_sizes..., kv_length, kv_features]`.
      mask: attention mask of shape `[batch_sizes..., num_heads, q_length,
        kv_length]`.
      bias: attention bias of shape `[batch_sizes..., num_heads, q_length,
        kv_length]`.
      precomputed_qkv: 3-tuple of precomputed query, key, value arrays, only
        used for parallel, fused-parameter optimizations.
      decode: Whether to prepare and use an autoregressive cache.
      enable_dropout: Enables dropout if set to True.
      prefill: Whether to run a partial sequence to prefill the cache.
      prefill_lengths: The length of each partial sequence we are filling in the
        cache, lengths are inferred from the mask if not provided.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    validate_dense_attention_call_parameter_shapes(
        inputs_q, inputs_kv, mask, bias, self.num_heads
    )
    q_kernel_init = (
        self.q_kernel_init
        if self.q_kernel_init is not None
        else self.kernel_init
    )

    rotary_index = None
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    if self.head_dim is None:
      head_dim = qkv_features // self.num_heads
    else:
      head_dim = self.head_dim

    # Is attention logit rescaling explicit or folded into initializer?
    if self.rescale_logits:
      query_init = q_kernel_init
    else:
      depth_scaling = jnp.sqrt(head_dim).astype(self.dtype)
      query_init = lambda *args: q_kernel_init(*args) / depth_scaling

    def dense_output(
        features,
        axis,
        kernel_init,
        kernel_axis_names,
        name,
        inputs,
        reshape_kernel=True,
    ):
      if self.use_aqt:
        if self.weight_params is None and self.act_params is None:
          raise ValueError(
              'If use_aqt is True, either of weights or acts quantization need '
              'to be specified using arguments `weight_params` or `act_params`.'
          )
        # TODO: Push the "quantized vs not" decision down into
        # the AQT library. Currently we make that decision here, because the AQT
        # library doesn't support DenseGeneral.
        aqt_context = aqt_config.DynamicContext(
            update_bounds=False, collect_acts_stats=False
        )
        weight_prec = self.weight_params.prec if self.weight_params else None
        half_shift = (
            self.weight_params.half_shift if self.weight_params else False
        )
        aqt_hparams = aqt_flax_layers.DenseAqt.HParams(
            weight_prec=weight_prec,
            weight_half_shift=half_shift,
            quant_act=self.act_params,  # currently supports fixed bounds only.
            quant_type=aqt.QuantType.AQT,
            weight_quant_granularity=aqt_config.QuantGranularity.PER_CHANNEL,
        )
        return aqt_flax_layers.DenseAqt(
            features=features,
            hparams=aqt_hparams,
            train=enable_dropout,
            dynamic_context=aqt_context,
            paxis_name=None,
            # No "cross-replica" reduction expressed in the XLA graph at this
            # stage. Will be imposed later, automatically, by XLA SPMD.
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            kernel_axis_names=kernel_axis_names,
            # we do not have reshape kernel option here but we explicitly
            # reshape kernel.
            precision=self.precision,
            possibly_use_quantized_vars=self.possibly_use_quantized_vars,
            name=name,
        )(inputs, padding_mask=None)
      else:
        return dense.DenseGeneral(
            axis=axis,
            features=features,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            dtype=self.dtype,
            kernel_init=kernel_init,
            precision=self.precision,
            kernel_axis_names=kernel_axis_names,
            reshape_kernel=reshape_kernel,
            name=name,
        )(inputs)

    # Project inputs_q to multi-headed q and single-headed k and v
    # query dimension is then [batch..., length, num_heads, features_per_head]
    # key and value dimensions are [batch..., length, features_per_head].
    if precomputed_qkv is None:
      query = dense_output(
          features=(self.num_heads, head_dim),
          axis=-1,
          kernel_init=query_init,
          kernel_axis_names=['embed', 'heads', 'kv'],
          name='query',
          inputs=inputs_q,
          reshape_kernel=not self.split_head_kernel,
      )

      key = dense_output(
          features=head_dim,
          axis=-1,
          kernel_init=self.kernel_init,
          kernel_axis_names=['embed', 'kv'],
          name='key',
          inputs=inputs_kv,
      )
      value = dense_output(
          features=head_dim,
          axis=-1,
          kernel_init=self.kernel_init,
          kernel_axis_names=['embed', 'kv'],
          name='value',
          inputs=inputs_kv,
      )
    else:
      query, key, value = precomputed_qkv

    # Multi Dconv Head Attention options:
    if self.q_conv is not None:
      query = self.q_conv(  # pylint: disable=not-callable
          query, decode=decode, prefill=prefill, prefill_lengths=prefill_lengths
      )
    if self.k_conv is not None:
      key = self.k_conv(  # pylint: disable=not-callable
          key, decode=decode, prefill=prefill, prefill_lengths=prefill_lengths
      )
    if self.v_conv is not None:
      value = self.v_conv(  # pylint: disable=not-callable
          value, decode=decode, prefill=prefill, prefill_lengths=prefill_lengths
      )

    sharding_prefix = 'attn_decode' if decode else 'attn_encode'

    bias_sharding = (
        f'{sharding_prefix}_batch',
        f'{sharding_prefix}_heads',
        f'{sharding_prefix}_q_length',
        f'{sharding_prefix}_kv_length',
    )

    # Note: We don't use `activation_partitioning.with_sharding_migration` here
    # because we do often want this 2D sharded. However, if rules are valid,
    # they should result in 2D sharding. We don't need to raise errors if both
    # result in 2D sharding (which with_sharding_migration does).
    if flax_partitioning.get_axis_rules():
      query = flax_partitioning.with_sharding_constraint(
          query, ('batch', 'length', 'heads', 'kv')
      )
    else:
      query = activation_partitioning.with_sharding(query, 2)

    if prefill and decode:
      raise ValueError(
          'prefill and decode cannot both be true at the same'
          'time. If you are using a prefix LM with bidirectional '
          'attention on the inputs, please make a call with '
          'prefill=True that includes an attention mask that '
          'covers your inputs first and then make your decoding '
          'calls.'
      )
    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    if prefill or decode:
      # Detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      # The key and value have dimension
      # [batch..., length, features_per_head], but we cache them as
      # [batch..., features_per_head, length] as a TPU fusion
      # optimization. This also enable the "scatter via one-hot broadcast"
      # trick, which means we do a one-hot broadcast instead of a scatter/gather
      # operations, which gives a 3-4x speedup in practice.
      swap_dims = lambda x: x[:-2] + tuple(x[i] for i in [-1, -2])
      cached_key = flax_partitioning.variable_with_axes(
          'cache',
          'cached_key',
          jnp.zeros,
          swap_dims(key.shape),
          key.dtype,
          axes=('cache_batch', 'cache_kv', 'cache_length'),
          fallback=RulesFallback.NO_CONSTRAINT,
      )
      cached_value = flax_partitioning.variable_with_axes(
          'cache',
          'cached_value',
          jnp.zeros,
          swap_dims(value.shape),
          value.dtype,
          axes=('cache_batch', 'cache_kv', 'cache_length'),
          fallback=RulesFallback.NO_CONSTRAINT,
      )
      cache_index = flax_partitioning.variable_with_axes(
          'cache',
          'cache_index',
          jnp.zeros,
          query.shape[0],
          jnp.int32,
          axes=('cache_batch',),
          fallback=RulesFallback.NO_CONSTRAINT,
      )
      rotary_index = cache_index.value

      if is_initialized:
        # Here we are in "apply()".
        *batch_dims, features_per_head, length = cached_key.value.shape
        if prefill:
          # Figure out how far each element in the batch fills the cache based
          # on the mask. We index each element in the batch, the first head
          # dim (because this is always set to one), and the first query
          # vector. If there is any prefix at all, the first element in the
          # prefix would be part of it. Note, the cache index is now a vector
          # of batch size so that each example can start just after it's
          # prefix which can be different lengths for different examples.
          if prefill_lengths is None:
            prefill_lengths = jnp.sum(mask[:, 0, 0, :], axis=-1).astype(
                cache_index.value.dtype
            )
          (
              key,
              value,
              cur_index,
              cached_key_value,
              cached_value_value,
              cache_index_value,
          ) = self.update_cache_prefill(
              key, value, cached_key, cached_value, cache_index, prefill_lengths
          )

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        elif decode:
          # Check the shape of the cached key against the input query.
          expected_query_shape = tuple(batch_dims) + (
              1,
              self.num_heads,
              features_per_head,
          )
          if expected_query_shape != query.shape:
            raise ValueError(
                'Autoregressive cache shape error, '
                'expected query shape %s instead got %s.'
                % (expected_query_shape, query.shape)
            )

          expected_key_shape = tuple(batch_dims) + (1, features_per_head)
          if expected_key_shape != key.shape:
            raise ValueError(
                'Autoregressive cache shape error, '
                'expected key shape %s instead got %s.'
                % (expected_key_shape, key.shape)
            )

          # value and key should have the same shape.
          if expected_key_shape != value.shape:
            raise ValueError(
                'Autoregressive cache shape error, '
                'expected value shape %s instead got %s.'
                % (expected_key_shape, value.shape)
            )
          (
              key,
              value,
              cur_index,
              cached_key_value,
              cached_value_value,
              cache_index_value,
          ) = self.update_cache_decode(
              key, value, cached_key, cached_value, cache_index
          )
          # Enforcing the Causal mask over previous positions and selecting only
          # the bias value for the current index is only needed during decode
          # mode where a single example is feed at a time. In prefill mode we
          # uses these as provided, that same way it is done in a normal forward
          # pass, like when computing logits during training.

          # Causal mask for cached decoder self-attention: our single query
          # position should only attend to those key positions that have already
          # been generated and cached, not the remaining zero elements.
          #
          # (1, 1, length) represent (head dim, query length, key length)
          # query length is 1 because during decoding we deal with one
          # index.
          # The same mask is applied to all batch elements and heads.
          #
          # Add trailing dims to the current index so it can either
          # broadcast over the batch dim or it can just be batch size.
          mask = combine_masks(
              mask,
              jnp.broadcast_to(
                  jnp.arange(length), tuple(batch_dims) + (1, 1, length)
              )
              <= jnp.reshape(cur_index, (-1, 1, 1, 1)),
          )

          mask = flax_partitioning.with_sharding_constraint(
              mask,
              (f'{sharding_prefix}_batch', None, None, None),
              fallback=RulesFallback.NO_CONSTRAINT,
          )

          # Grab the correct relative attention bias during decoding.
          if bias is not None:
            # The bias is a full attention matrix, but during decoding we only
            # have to take a slice of it.
            # This is equivalent to bias[..., cur_index:cur_index+1, :].
            # If we are doing prefix decoding where cur index is a vector the
            # result will be [batch, heads, 1, :]. If cur_index is a scalar
            # like in encdec decoding, the result will be [1, heads, 1, :]
            # We use a one-hot einsum rather than a slice to avoid introducing
            # a Gather op that is currently lowered poorly by SPMD passes,
            # adding expensive all-reduce and all-gather operations.

            bias = jnp.einsum(
                'bq, bhqk->bhk',
                common_utils.onehot(cur_index, num_classes=length),
                bias,
            )
            bias = jnp.expand_dims(bias, 2)
            bias = flax_partitioning.with_sharding_constraint(
                bias, bias_sharding, fallback=RulesFallback.NO_CONSTRAINT
            )

        # Currently, updating a variable inside of a method is not handled
        # in flax, so we return the actual values and assign them in the main
        # compacted call for now.
        # TODO: Move variable assignment inside of the
        # cache update functions once variable references are tracked across
        # transform boundaries.
        cache_index.value = cache_index_value
        cached_key.value = cached_key_value
        cached_value.value = cached_value_value

    # Convert the boolean attention mask to an attention bias.
    if mask is not None:
      # attention mask in the form of attention bias
      attention_bias = lax.select(
          mask > 0,
          jnp.full(mask.shape, 0.0).astype(self.dtype),
          jnp.full(mask.shape, -1e10).astype(self.dtype),
      )
      attention_bias = flax_partitioning.with_sharding_constraint(
          attention_bias, bias_sharding, fallback=RulesFallback.NO_CONSTRAINT
      )
    else:
      attention_bias = None

    # Add provided bias term (e.g. relative position embedding).
    if bias is not None:
      attention_bias = combine_biases(attention_bias, bias)
      attention_bias = flax_partitioning.with_sharding_constraint(
          attention_bias, bias_sharding, fallback=RulesFallback.NO_CONSTRAINT
      )

    dropout_rng = None
    if enable_dropout and self.dropout_rate > 0.0:
      dropout_rng = self.make_rng('dropout')

    # During decode we typically want to reshard at this point from sharding by
    # by head to sharding by batch. Give new names to the sharding axes to allow
    # this reshard.
    query = flax_partitioning.with_sharding_constraint(
        query,
        (
            f'{sharding_prefix}_batch',
            f'{sharding_prefix}_q_length',
            f'{sharding_prefix}_heads',
            'kv',
        ),
        fallback=RulesFallback.NO_CONSTRAINT,
    )
    key = flax_partitioning.with_sharding_constraint(
        key,
        (f'{sharding_prefix}_batch', f'{sharding_prefix}_kv_length', 'kv'),
        fallback=RulesFallback.NO_CONSTRAINT,
    )
    value = flax_partitioning.with_sharding_constraint(
        value,
        (f'{sharding_prefix}_batch', f'{sharding_prefix}_kv_length', 'kv'),
        fallback=RulesFallback.NO_CONSTRAINT,
    )

    if self.use_rotary_embedding:
      # use rotary embeddings before attention
      # https://arxiv.org/abs/2104.09864
      # TODO: Figure out if this should be put in a new class.
      dim = query.shape[-1]
      max_length = max(query.shape[1], key.shape[1])
      sin, cos = embedding.generate_fixed_pos_embedding(
          dim, max_length, max_timescale=self.rotary_embedding_max_timescale
      )
      sin = sin.astype(self.dtype)
      cos = cos.astype(self.dtype)
      query, key = embedding.apply_rotary_embedding(
          query, key, cos, sin, decode=decode, rotary_index=rotary_index
      )

    # Apply attention.
    x = self.attention_fn(
        query,
        key,
        value,
        bias=attention_bias,
        broadcast_dropout=self.broadcast_dropout,
        rescale_logits=self.rescale_logits,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        enable_dropout=enable_dropout,
        dtype=self.dtype,
        precision=self.precision,
        use_extra_logit=self.use_extra_logit,
        float32_logits=self.float32_logits,
    )  # pytype: disable=wrong-keyword-args

    # During decode we typically want to reshard at this point from sharding by
    # batch to sharding by head. Return to the old names of the sharding axes to
    # allow this reshard.
    x = flax_partitioning.with_sharding_constraint(
        x,
        (
            f'{sharding_prefix}_batch',
            f'{sharding_prefix}_q_length',
            f'{sharding_prefix}_heads',
            'kv',
        ),
        fallback=RulesFallback.NO_CONSTRAINT,
    )
    x = flax_partitioning.with_sharding_constraint(
        x,
        ('batch', 'length', 'heads', 'kv'),
        fallback=RulesFallback.NO_CONSTRAINT,
    )

    if precomputed_qkv is None:
      kernel_axis_names = ['heads', 'kv', 'embed']
      # TODO: activation quantization support is unimplemented
      # here.
      if self.use_aqt and self.weight_params is not None:
        weight_prec = self.weight_params.prec if self.weight_params else None
        half_shift = (
            self.weight_params.half_shift if self.weight_params else False
        )
        aqt_hparams = aqt_flax_layers.DenseGeneralAqt.HParams(
            weight_prec=weight_prec,
            weight_half_shift=half_shift,
            quant_act=None,  # currently supports fixed bounds only.
            weight_quant_granularity=aqt_config.QuantGranularity.PER_CHANNEL,
        )
        out = aqt_flax_layers.DenseGeneralAqt(
            hparams=aqt_hparams,
            train=enable_dropout,
            possibly_use_quantized_vars=self.possibly_use_quantized_vars,
            features=features,
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            dtype=self.dtype,
            precision=self.precision,
            kernel_axis_names=kernel_axis_names,
            reshape_kernel=not self.split_head_kernel,
            name='out',
        )(  # pytype: disable=wrong-arg-types
            x
        )
      else:
        # Back to the original inputs dimensions.
        out = dense.DenseGeneral(
            features=features,
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            dtype=self.dtype,
            precision=self.precision,
            kernel_axis_names=kernel_axis_names,
            reshape_kernel=not self.split_head_kernel,
            name='out',
        )(  # pytype: disable=wrong-arg-types
            x
        )
    else:
      # in fused parallel layer, fused outer dense operation is external
      out = x

    return out


class LocalAttentionLayer(nn.Module, DenseAttention):
  """Performs attention on separate chunks and concatenates the results.

  This attention is local in that attention happens only within each chunk
  (as defined by the length of the strides).

  For example usage, see CANINE, that uses this to attend over character chunks.

  Attributes:
    localized_attention: The attention layer that will be applied to each chunk.
      Typically this is a `MultiHeadDotProductAttention`.
    q_chunk_width: The width of each chunk in `inputs_q`. A value of 128 is
      optimal for TPUs.
    q_chunk_stride: The number of elements to skip when moving to the next chunk
      in `inputs_q`. Typically this is equal to `q_chunk_width`.
    kv_chunk_width: The width of each chunk in `inputs_kv`. A value of 128 is
      optimal for TPUs.
    kv_chunk_stride: The number of elements to skip when moving to the next
      chunk in `inputs_kv`. Typically this is equal to `kv_chunk_width`.
    always_attend_to_first_position: Should all chunks be able to attend to the
      `inputs_kv`'s first position (e.g. a [CLS] position)?
    first_position_attends_to_all: Should the `inputs_q`'s first position be
      able to attend to all positions within the `inputs_q`?
  """

  localized_attention: DenseAttention
  q_chunk_width: int
  q_chunk_stride: int
  kv_chunk_width: int
  kv_chunk_stride: int
  always_attend_to_first_position: bool = False
  first_position_attends_to_all: bool = False

  def setup(self) -> None:
    if self.q_chunk_stride > self.q_chunk_width:
      raise ValueError(
          '`q_chunk_stride` < `q_chunk_width` '
          'would cause `inputs_q` positions to get skipped.'
      )
    if self.kv_chunk_stride > self.kv_chunk_width:
      raise ValueError(
          '`kv_chunk_stride` > `kv_chunk_width` '
          'would cause `inputs_kv` positions to get skipped.'
      )

  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      mask: Optional[Array] = None,
      bias: Optional[Array] = None,
      *,
      decode: bool = False,
      enable_dropout: bool = True,
      precomputed_qkv: Optional[Array] = None,
  ) -> Array:
    """Applies local attention.

    Args:
      inputs_q: <float>[batch_sizes..., q_len, q_features].
      inputs_kv: <float>[batch_sizes..., kv_len, kv_features].
      mask: <int32>[batch_sizes..., num_heads, q_len, kv_len]. The values should
        be 1 or 0. The attention scores will effectively be set to -∞ for any
        positions in the mask that are 0, and will be unchanged for positions
        that are 1.
      bias: Unsupported.
      decode: Unsupported.
      enable_dropout: Enables dropout if set to True.
      precomputed_qkv: Precomputed QKV arrays. Not currently supported.

    Returns:
      An array with the same shape as would be expected from calling
      `localized_attention` directly.
    """
    chex.assert_shape(inputs_q, (..., None, None))
    num_batch_dims = inputs_q.ndim - 2
    batch_sizes = inputs_q.shape[:num_batch_dims]
    chex.assert_shape(inputs_kv, (*batch_sizes, None, None))
    q_len, _ = inputs_q.shape[num_batch_dims:]
    kv_len, _ = inputs_kv.shape[num_batch_dims:]
    if mask is not None:
      chex.assert_shape(mask, (*batch_sizes, None, q_len, kv_len))
    if bias is not None:
      chex.assert_shape(
          bias, (*({b, 1} for b in batch_sizes), None, q_len, kv_len)
      )

    if decode:
      raise ValueError(f'{type(self).__name__} does not support decoding mode')

    # Determine the chunks that we will attend *from*.
    q_chunks = []
    if self.first_position_attends_to_all:
      q_chunks.append((0, 1))
      # We must skip this first position so that our output sequence is the
      # correct length (this matters in the *from* sequence only).
      q_start = 1
    else:
      q_start = 0
    for chunk_start in range(q_start, q_len, self.q_chunk_stride):
      chunk_end = min(q_len, chunk_start + self.q_chunk_width)
      q_chunks.append((chunk_start, chunk_end))

    # Determine the chunks that we will attend *to*.
    kv_chunks = []
    if self.first_position_attends_to_all:
      kv_chunks.append((0, kv_len))
    for chunk_start in range(0, kv_len, self.kv_chunk_stride):
      chunk_end = min(kv_len, chunk_start + self.kv_chunk_width)
      kv_chunks.append((chunk_start, chunk_end))

    if len(q_chunks) != len(kv_chunks):
      raise ValueError(
          f'Expected to have same number of `q_chunks` ({q_chunks}) and '
          f'`kv_chunks` ({kv_chunks}). Check strides.'
      )

    # TODO: Can we save a bit of extra compute by slicing the Q/K/V
    # projected versions of these instead of recomputing those projections?
    # This only helps when the Q stride isn't the same as the K/V stride.
    # Length of `attention_output_chunks` and therefore `attention_output` is
    # determined by `q_chunks` to ensure correctness. The correspondence with
    # `kv_chunks` is somewhat best effort. We need to do more to enforce this.
    attention_output_chunks = []
    for q_chunk, kv_chunk in zip(q_chunks, kv_chunks):
      q_start, q_end = q_chunk
      kv_start, kv_end = kv_chunk

      inputs_q_chunk = inputs_q[..., q_start:q_end, :]
      inputs_kv_chunk = inputs_kv[..., kv_start:kv_end, :]
      if mask is not None:
        mask_chunk = mask[..., q_start:q_end, kv_start:kv_end]
      if bias is not None:
        bias_chunk = bias[..., q_start:q_end, kv_start:kv_end]

      if self.always_attend_to_first_position:
        if mask is not None:
          cls_mask = mask[..., q_start:q_end, 0:1]
          mask_chunk = jnp.concatenate([cls_mask, mask_chunk], axis=-1)
        if bias is not None:
          cls_bias = bias[..., q_start:q_end, 0:1]
          bias_chunk = jnp.concatenate([cls_bias, bias_chunk], axis=-1)

        kv_cls = inputs_kv[..., 0:1, :]
        inputs_kv_chunk = jnp.concatenate([kv_cls, inputs_kv_chunk], axis=-2)

      attention_output_chunk = self.localized_attention(
          inputs_q=inputs_q_chunk,
          inputs_kv=inputs_kv_chunk,
          mask=mask_chunk if mask is not None else None,
          bias=bias_chunk if bias is not None else None,
          enable_dropout=enable_dropout,
      )
      chex.assert_shape(
          attention_output_chunk, (*inputs_q_chunk.shape[:-1], ...)
      )
      attention_output_chunks.append(attention_output_chunk)

    # Concatenate along the length dim (which directly follows the batch dims).
    return jnp.concatenate(attention_output_chunks, axis=num_batch_dims)


# ------------------------------------------------------------------------------
# Mask-making utility functions.
# ------------------------------------------------------------------------------
def make_attention_mask(
    query_input: Array,
    key_input: Array,
    pairwise_fn: Callable = jnp.multiply,
    extra_batch_dims: int = 0,
    dtype: DType = jnp.float32,
) -> Array:
  """Mask-making helper for attention weights.

  In case of 1d inputs (i.e., `[batch..., len_q]`, `[batch..., len_kv]`, the
  attention weights will be `[batch..., heads, len_q, len_kv]` and this
  function will produce `[batch..., 1, len_q, len_kv]`.

  Args:
    query_input: a batched, flat input of query_length size
    key_input: a batched, flat input of key_length size
    pairwise_fn: broadcasting elementwise comparison function
    extra_batch_dims: number of extra batch dims to add singleton axes for, none
      by default
    dtype: mask return dtype

  Returns:
    A `[batch..., 1, len_q, len_kv]` shaped mask for 1d attention.
  """
  # [batch..., len_q, len_kv]
  mask = pairwise_fn(
      # [batch..., len_q] -> [batch..., len_q, 1]
      jnp.expand_dims(query_input, axis=-1),
      # [batch..., len_q] -> [batch..., 1, len_kv]
      jnp.expand_dims(key_input, axis=-2),
  )

  # [batch..., 1, len_q, len_kv]. This creates the head dim.
  mask = jnp.expand_dims(mask, axis=-3)
  mask = jnp.expand_dims(mask, axis=tuple(range(extra_batch_dims)))
  return mask.astype(dtype)


def make_causal_mask(
    x: Array, extra_batch_dims: int = 0, dtype: DType = jnp.float32
) -> Array:
  """Make a causal mask for self-attention.

  In case of 1d inputs (i.e., `[batch..., len]`, the self-attention weights
  will be `[batch..., heads, len, len]` and this function will produce a
  causal mask of shape `[batch..., 1, len, len]`.

  Note that a causal mask does not depend on the values of x; it only depends on
  the shape. If x has padding elements, they will not be treated in a special
  manner.

  Args:
    x: input array of shape `[batch..., len]`
    extra_batch_dims: number of batch dims to add singleton axes for, none by
      default
    dtype: mask return dtype

  Returns:
    A `[batch..., 1, len, len]` shaped causal mask for 1d attention.
  """
  idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
  return make_attention_mask(
      idxs,
      idxs,
      jnp.greater_equal,
      extra_batch_dims=extra_batch_dims,
      dtype=dtype,
  )


def combine_masks(*masks: Optional[Array], dtype: DType = jnp.float32):
  """Combine attention masks.

  Args:
    *masks: set of attention mask arguments to combine, some can be None.
    dtype: final mask dtype

  Returns:
    Combined mask, reduced by logical and, returns None if no masks given.
  """
  masks = [m for m in masks if m is not None]
  if not masks:
    return None
  assert all(
      map(lambda x: x.ndim == masks[0].ndim, masks)
  ), f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}'
  mask, *other_masks = masks
  for other_mask in other_masks:
    mask = jnp.logical_and(mask, other_mask)
  return mask.astype(dtype)


def combine_biases(*masks: Optional[Array]):
  """Combine attention biases.

  Args:
    *masks: set of attention bias arguments to combine, some can be None.

  Returns:
    Combined mask, reduced by summation, returns None if no masks given.
  """
  masks = [m for m in masks if m is not None]
  if not masks:
    return None
  assert all(
      map(lambda x: x.ndim == masks[0].ndim, masks)
  ), f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}'
  mask, *other_masks = masks
  for other_mask in other_masks:
    mask = mask + other_mask
  return mask


def make_decoder_mask(
    decoder_target_tokens: Array,
    dtype: DType,
    decoder_causal_attention: Optional[Array] = None,
    decoder_segment_ids: Optional[Array] = None,
) -> Array:
  """Compute the self-attention mask for a decoder.

  Decoder mask is formed by combining a causal mask, a padding mask and an
  optional packing mask. If decoder_causal_attention is passed, it makes the
  masking non-causal for positions that have value of 1.

  A prefix LM is applied to a dataset which has a notion of "inputs" and
  "targets", e.g., a machine translation task. The inputs and targets are
  concatenated to form a new target. `decoder_target_tokens` is the concatenated
  decoder output tokens.

  The "inputs" portion of the concatenated sequence can attend to other "inputs"
  tokens even for those at a later time steps. In order to control this
  behavior, `decoder_causal_attention` is necessary. This is a binary mask with
  a value of 1 indicating that the position belonged to "inputs" portion of the
  original dataset.

  Example:

    Suppose we have a dataset with two examples.

    ds = [{"inputs": [6, 7], "targets": [8]},
          {"inputs": [3, 4], "targets": [5]}]

    After the data preprocessing with packing, the two examples are packed into
    one example with the following three fields (some fields are skipped for
    simplicity).

       decoder_target_tokens = [[6, 7, 8, 3, 4, 5, 0]]
         decoder_segment_ids = [[1, 1, 1, 2, 2, 2, 0]]
    decoder_causal_attention = [[1, 1, 0, 1, 1, 0, 0]]

    where each array has [batch, length] shape with batch size being 1. Then,
    this function computes the following mask.

                      mask = [[[[1, 1, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 0, 1, 1, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0]]]]

    mask[b, 1, :, :] represents the mask for the example `b` in the batch.
    Because mask is for a self-attention layer, the mask's shape is a square of
    shape [query length, key length].

    mask[b, 1, i, j] = 1 means that the query token at position i can attend to
    the key token at position j.

  Args:
    decoder_target_tokens: decoder output tokens. [batch..., length]
    dtype: dtype of the output mask.
    decoder_causal_attention: a binary mask indicating which position should
      only attend to earlier positions in the sequence. Others will attend
      bidirectionally. [batch..., length]
    decoder_segment_ids: decoder segmentation info for packed examples.
      [batch..., length]

  Returns:
    the combined decoder mask.
  """
  masks = []
  # The same mask is applied to all attention heads. So the head dimension is 1,
  # i.e., the mask will be broadcast along the heads dim.
  # [batch..., 1, length, length]
  causal_mask = make_causal_mask(decoder_target_tokens, dtype=dtype)

  # Positions with value 1 in `decoder_causal_attention` can attend
  # bidirectionally.
  if decoder_causal_attention is not None:
    # [batch..., 1, length, length]
    inputs_mask = make_attention_mask(
        decoder_causal_attention,
        decoder_causal_attention,
        jnp.logical_and,
        dtype=dtype,
    )
    masks.append(jnp.logical_or(causal_mask, inputs_mask).astype(dtype))
  else:
    masks.append(causal_mask)

  # Padding mask.
  masks.append(
      make_attention_mask(
          decoder_target_tokens > 0, decoder_target_tokens > 0, dtype=dtype
      )
  )

  # Packing mask
  if decoder_segment_ids is not None:
    masks.append(
        make_attention_mask(
            decoder_segment_ids, decoder_segment_ids, jnp.equal, dtype=dtype
        )
    )

  return combine_masks(*masks, dtype=dtype)  # pytype: disable=bad-return-type  # jax-ndarray


def validate_dense_attention_call_parameter_shapes(
    inputs_q: Array,
    inputs_kv: Array,
    mask: Optional[Array],
    bias: Optional[Array],
    num_heads: Optional[int],
):
  """Validates the shapes of parameters to DenseAttention call methods."""
  if inputs_q.ndim != inputs_kv.ndim:
    raise ValueError(
        'Mismatched inputs rank: expected '
        f'inputs_q.ndim ({inputs_q.ndim}) == '
        f'inputs_kv.ndim ({inputs_kv.ndim})'
    )
  if inputs_q.ndim < 3:
    raise ValueError(f'Expected rank of inputs >= 3, was {inputs_q.ndim}')
  if inputs_q.shape[:-3] != inputs_kv.shape[:-3]:
    raise ValueError(
        'Mismatched batch dims: expected '
        f'inputs_q.shape[:-3] ({inputs_q.shape[:-3]}) == '
        f'inputs_kv.shape[:-3] ({inputs_kv.shape[:-3]})'
    )
  if mask is not None:
    if mask.ndim != inputs_q.ndim + 1:
      raise ValueError(
          'Mismatched ranks: expected '
          f'mask.ndim ({mask.ndim}) to be one more than '
          f'inputs_q.ndim ({inputs_q.ndim})'
      )
    if num_heads is not None:
      if mask.shape[-3] not in (1, num_heads):
        raise ValueError(
            'Mismatched num_heads: expected '
            f'mask.shape[-3] ({mask.shape[-3]}) == '
            f'num_heads ({num_heads}), or 1'
        )
    else:
      num_heads = mask.shape[-3]
    if mask.shape[-2] not in (1, inputs_q.shape[-2]):
      raise ValueError(
          'Mismatched q_length: expected '
          f'mask.shape[-2] ({mask.shape[-2]}) == '
          f'inputs_q.shape[-2] ({inputs_q.shape[-2]}), or 1'
      )
    if mask.shape[-1] != inputs_kv.shape[-2]:
      raise ValueError(
          'Mismatched kv_length: expected '
          f'mask.shape[-1] ({mask.shape[-1]}) == '
          f'inputs_kv.shape[-2] ({inputs_kv.shape[-2]})'
      )
  if bias is not None:
    if bias.ndim != inputs_q.ndim + 1:
      raise ValueError(
          'Mismatched ranks: expected '
          f'bias.ndim ({bias.ndim}) to be one less than '
          f'inputs_q.ndim ({inputs_q.ndim})'
      )
    if num_heads is not None:
      if bias.shape[-3] not in (1, num_heads):
        raise ValueError(
            'Mismatched num_heads: expected '
            f'bias.shape[-3] ({bias.shape[-3]}) == '
            f'num_heads ({num_heads}), or 1'
        )
    else:
      num_heads = bias.shape[-3]
    if bias.shape[-2] != inputs_q.shape[-2]:
      if inputs_q.shape[-2] != 1:  # TODO: Remove this exception?
        raise ValueError(
            'Mismatched q_length: expected '
            f'bias.shape[-2] ({bias.shape[-2]}) == '
            f'inputs_q.shape[-2] ({inputs_q.shape[-2]})'
        )
    if bias.shape[-1] != inputs_kv.shape[-2]:
      if inputs_kv.shape[-2] != 1:  # TODO: Remove this exception?
        raise ValueError(
            'Mismatched kv_length: expected '
            f'bias.shape[-1] ({bias.shape[-1]}) == '
            f'inputs_kv.shape[-2] ({inputs_kv.shape[-2]})'
        )


def shift_left(x, axis=-1):
  """Shift the input array to the left by one unit and pad by 0 to the right."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (0, 1)
  padded = jnp.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(0)
  )
  return jax.lax.slice_in_dim(padded, 1, x.shape[axis] + 1, axis=axis)


def get_decoder_logit_mask(decoder_input_tokens, dtype):
  """Gets a mask that zeros-out the padding tokens in the attention logits.

  This function creates a mask that can be used to zero out the logits
  corresponding to the padding tokens. Typically, a large negative number is
  added to the attention logits so that those tokens are not attended. If the
  magnitude of the attention logits of the non-padding tokens is on the same
  order as the large negative number, the attention pattern gets affected and
  results in an unexpected behavior. Empirically, this can happen for large
  models and using this mask can help stabilizing the training.

  In T5 models, the token id of 0 serves as the padding id as well as the BOS id
  in the decoder. This requires a special handling of the BOS token ids.
  Furthermore, the input sequences can be packed where each sequences.

  TODO: Make T5 models use a separate BOS ID.

  ```
  Example: Here booleans are represented as 1 and 0 for simplicity and the batch
           dimension is skipped.

      decoder_input_tokens = [0, 3, 9, 0, 4, 8, 0, 0]
  ---------------------------------------------------
    decoder_input_tokens_0 = [1, 0, 0, 0, 0, 0, 0, 0]
               shifted > 0 = [1, 1, 0, 1, 1, 0, 0, 0]
  decoder_input_tokens > 0 = [0, 1, 1, 0, 1, 1, 0, 0]
  ---------------------------------------------------
                logit_mask = [1, 1, 1, 1, 1, 1, 0, 0]
  ```

  NB: this function assumes that the each sequence in the packed
  `decoder_input_tokens` has exactly one 0-BOS token.

  Args:
    decoder_input_tokens: input_tokens to the decoder with one 0-BOS id per
      sequence.
    dtype: output (logit mask) data type.

  Returns:
    logit mask with zeros and ones with a shape [batch, length, 1]
  """
  # We don't want to mask the initial shifted '0-BOS' logit of decoder inputs.
  decoder_input_tokens_0 = (
      jax.lax.broadcasted_iota(
          jnp.int32, decoder_input_tokens.shape, decoder_input_tokens.ndim - 1
      )
      == 0
  )

  return jnp.expand_dims(
      jnp.array(
          (decoder_input_tokens > 0)
          | (shift_left(decoder_input_tokens) > 0)
          | decoder_input_tokens_0,
          dtype=dtype,
      ),
      axis=-1,
  )
