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

"""Extending components.attention.dense_attention to allow cache propagation."""

# pylint: disable=attribute-defined-outside-init,g-bare-generic

import functools
from typing import Optional

from flax import linen as nn
from flax.linen import partitioning as flax_partitioning
from flax.training import common_utils
from jax import lax
import jax.numpy as jnp

from flaxformer import activation_partitioning
from flaxformer.components import dense
from flaxformer.components import embedding
from flaxformer.components.attention import dense_attention
from flaxformer.types import Array


class MultiHeadDotProductAttention(dense_attention.MultiHeadDotProductAttention
                                  ):
  """Extends Multi-head dot-product attention class to enable cache propagation.

  Cache propagation is enabled by setting only_propagate_state in the __call__
  function. Used for early exiting, passing the last hidden state to the skipped
  layers.
  """

  @nn.compact
  def __call__(self,
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
               only_propagate_state: bool = False) -> Array:
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
      only_propagate_state: Whether to stop after the projected key-values are
        computed and stored in the auto-regressive cache. If true, the rest of
        the computation is skipped, and nothing is returned.

    Returns:
      If output_projection is True, then output of shape
      `[batch_sizes..., length, out_features]`, where out_features is set to
      features if not provided. If output_projection is False, then output of
      shape `[batch_sizes..., length, num_heads, head_dim]`.
    """
    dense_attention.validate_dense_attention_call_parameter_shapes(
        inputs_q, inputs_kv, mask, bias, self.num_heads)

    qkv_kernel_init = (
        self.qkv_kernel_init
        if self.qkv_kernel_init is not None else self.kernel_init)
    kv_kernel_init = (
        self.kv_kernel_init
        if self.kv_kernel_init is not None else self.kernel_init)
    q_kernel_init = (
        self.q_kernel_init
        if self.q_kernel_init is not None else self.kernel_init)

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
      raise ValueError('Un-reshaped kernels are required when using QKV fused '
                       'kernel optimization.')

    # Is attention logit rescaling explicit or folded into initializer?
    if self.rescale_logits:
      query_init = q_kernel_init
    else:
      if self.kernels_to_fuse:
        raise ValueError('Cannot fold in logit normalization to query '
                         'initializer when using fused kernels.')
      depth_scaling = jnp.sqrt(head_dim).astype(self.dtype)
      query_init = lambda *args: q_kernel_init(*args) / depth_scaling

    make_dense = functools.partial(
        dense.DenseGeneral,
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
          name='query')(
              inputs_q)
      key = make_dense(
          kernel_init=self.kernel_init,
          features=(self.num_heads, head_dim),
          kernel_axis_names=['embed', 'heads', 'kv'],
          name='key')(
              inputs_kv)
      value = make_dense(
          kernel_init=self.kernel_init,
          features=(self.num_heads, head_dim),
          kernel_axis_names=['embed', 'heads', 'kv'],
          name='value')(
              inputs_kv)
    # TODO: should we fuse/slice along depth or head dim?
    elif self.kernels_to_fuse == 'qkv':
      if inputs_q is not inputs_kv:
        raise ValueError('qkv fusion is only supported in self-attention mode '
                         '(when inputs_q is inputs_kv).')
      # 'qkv' fusion mode implies self-attention
      qkv = make_dense(
          kernel_init=qkv_kernel_init,
          features=(3, self.num_heads, head_dim),
          kernel_axis_names=['embed', 'stack', 'heads', 'kv'],
          name='qkv_fused')(
              inputs_q)
      query = jnp.squeeze(lax.dynamic_slice_in_dim(qkv, 0, 1, -3), -3)
      key = jnp.squeeze(lax.dynamic_slice_in_dim(qkv, 1, 1, -3), -3)
      value = jnp.squeeze(lax.dynamic_slice_in_dim(qkv, 2, 1, -3), -3)
    elif self.kernels_to_fuse == 'kv':
      query = make_dense(
          kernel_init=query_init,
          features=(self.num_heads, head_dim),
          kernel_axis_names=['embed', 'heads', 'kv'],
          name='query')(
              inputs_q)
      kv = make_dense(
          kernel_init=kv_kernel_init,
          features=(2, self.num_heads, head_dim),
          kernel_axis_names=['embed', 'stack', 'heads', 'kv'],
          name='kv_fused')(
              inputs_kv)
      key = jnp.squeeze(lax.dynamic_slice_in_dim(kv, 0, 1, -3), -3)
      value = jnp.squeeze(lax.dynamic_slice_in_dim(kv, 1, 1, -3), -3)
    else:
      raise ValueError('Incorrect kernel fusion mode specified.')

    # Multi Dconv Head Attention options:
    if self.q_conv is not None:
      query = self.q_conv(  # pylint: disable=not-callable
          query,
          decode=decode,
          prefill=prefill,
          prefill_lengths=prefill_lengths)
    if self.k_conv is not None:
      key = self.k_conv(  # pylint: disable=not-callable
          key,
          decode=decode,
          prefill=prefill,
          prefill_lengths=prefill_lengths)
    if self.v_conv is not None:
      value = self.v_conv(  # pylint: disable=not-callable
          value,
          decode=decode,
          prefill=prefill,
          prefill_lengths=prefill_lengths)

    if self.sharding_over_head_dimension:
      # Note: We don't use `activation_partitioning.with_sharding_migration`
      # here because we do often want this 2D sharded. However, if rules are
      # valid, they should result in 2D sharding. We don't need to raise errors
      # if both result in 2D sharding (which with_sharding_migration does).
      if flax_partitioning.get_axis_rules():
        query = flax_partitioning.with_sharding_constraint(
            query, ('batch', 'length', 'heads', 'kv'))
        key = flax_partitioning.with_sharding_constraint(
            key, ('batch', 'length', 'heads', 'kv'))
        value = flax_partitioning.with_sharding_constraint(
            value, ('batch', 'length', 'heads', 'kv'))
      else:
        query = activation_partitioning.with_sharding(query, 2)
        key = activation_partitioning.with_sharding(key, 2)
        value = activation_partitioning.with_sharding(value, 2)

    query: Array = query  # hint to quiet pytype.
    key: Array = key
    value: Array = value

    if prefill and decode:
      raise ValueError('prefill and decode cannot both be true at the same'
                       'time. If you are using a prefix LM with bidirectional '
                       'attention on the inputs, please make a call with '
                       'prefill=True that includes an attention mask that '
                       'covers your inputs first and then make your decoding '
                       'calls.')
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
      cached_key = self.variable('cache', 'cached_key', jnp.zeros,
                                 swap_dims(key.shape), key.dtype)
      cached_value = self.variable('cache', 'cached_value', jnp.zeros,
                                   swap_dims(value.shape), value.dtype)
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.int32))
      rotary_index = cache_index.value
      if is_initialized:
        # Here we are in "apply()".
        *batch_dims, num_heads, features_per_head, length = (
            cached_key.value.shape)
        if prefill:
          if prefill_lengths is None:
            # Figure out how far each element in the batch fills the cache based
            # on the mask. We index each element in the batch, the first head
            # dim (because this is always set to one), and the first query
            # vector. If there is any prefix at all, the first element in the
            # prefix would be part of it.
            prefill_lengths = jnp.sum(
                mask[:, 0, 0, :], axis=-1).astype(cache_index.value.dtype)
          (key, value, cur_index, cached_key_value, cached_value_value,
           cache_index_value) = self.update_cache_prefill(
               key, value, cached_key, cached_value, cache_index,
               prefill_lengths)
        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        elif decode:
          # Check the shape of the cached key against the input query.
          expected_shape = tuple(batch_dims) + (1, num_heads, features_per_head)
          if expected_shape != query.shape:
            raise ValueError('Autoregressive cache shape error, '
                             'expected query shape %s instead got %s.' %
                             (expected_shape, query.shape))
          (key, value, cur_index, cached_key_value, cached_value_value,
           cache_index_value) = self.update_cache_decode(
               key, value, cached_key, cached_value, cache_index)
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
          mask = dense_attention.combine_masks(
              mask,
              jnp.broadcast_to(
                  jnp.arange(length),
                  tuple(batch_dims) +
                  (1, 1, length)) <= jnp.reshape(cur_index, (-1, 1, 1, 1)))
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
                common_utils.onehot(cur_index, num_classes=length), bias)
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

    if only_propagate_state:
      return  # pytype: disable=bad-return-type  # jax-ndarray

    # Convert the boolean attention mask to an attention bias.
    if mask is not None:
      # attention mask in the form of attention bias
      attention_bias = lax.select(
          mask > 0,
          jnp.full(mask.shape, 0.).astype(self.dtype),
          jnp.full(mask.shape, -1e10).astype(self.dtype))
    else:
      attention_bias = None

    # Add provided bias term (e.g. relative position embedding).
    if bias is not None:
      attention_bias = dense_attention.combine_biases(attention_bias, bias)

    dropout_rng = None
    if enable_dropout and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    if self.use_rotary_embedding:
      # use rotary embeddings before attention
      # https://arxiv.org/abs/2104.09864
      # TODO: Put it in a new class
      dim = query.shape[-1]
      max_length = max(query.shape[1], key.shape[1])
      sin, cos = embedding.generate_fixed_pos_embedding(
          dim, max_length, max_timescale=self.rotary_embedding_max_timescale)
      query, key = embedding.apply_rotary_embedding(
          query, key, cos, sin, decode=decode, rotary_index=rotary_index)

    # Compute attention.
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

    if not self.output_projection:
      return x

    # Back to the original inputs dimensions.
    out = dense.DenseGeneral(
        features=features,
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        precision=self.precision,
        reshape_kernel=not self.split_head_kernel,
        kernel_axis_names=['heads', 'kv', 'embed'],
        name='out')(  # pytype: disable=wrong-arg-types
            x)
    return out
