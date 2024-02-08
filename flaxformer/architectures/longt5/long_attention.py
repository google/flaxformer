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

"""Long attention classes and mask/weighting functions."""


import abc
import functools
from typing import Any, Callable, Optional, Tuple, Union
from flax import linen as nn

from flax.linen import initializers
from flax.linen import partitioning
from flax.linen.linear import default_kernel_init
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
from flaxformer.architectures.longt5 import relative_position_biases_general
from flaxformer.architectures.longt5 import tensor_utils
from flaxformer.components import dense
from flaxformer.components import embedding
from flaxformer.components import layer_norm
from flaxformer.components.attention import dense_attention  # GOOGLE-INTERNAL # pylint: disable=line-too-long
from flaxformer.types import Array
from flaxformer.types import DType
from flaxformer.types import Initializer
from flaxformer.types import PRNGKey


RelativePositionBiasesGeneral = (
    relative_position_biases_general.RelativePositionBiasesGeneral)


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
# Long attention layers.
# ------------------------------------------------------------------------------


class LongSelfAttention(abc.ABC):
  """API for long self-attention classes.

  These should be nn.Module instances also.
  """

  @abc.abstractmethod
  def __call__(self,
               inputs: Array,
               inputs_mask: Array,
               *,
               segment_ids: Optional[Array] = None,
               positions: Optional[Array] = None,
               enable_dropout: bool = True) -> Array:
    """Calls the attention layer.

    See the following for an example of how packed inputs are represented
    by `segment_ids` and `positions`:
    https://github.com/google/seqio/blob/main/seqio/utils.py#L292

    Args:
      inputs: <float>[batch, length, emb_dim] array of embeddings to self-attend
        over.
      inputs_mask: <bool>[batch, length] array indicating True for non-padding
        tokens and False for padding.
      segment_ids: Optional <int32>[batch, length] encoder input segmentation
        info for packed examples.
      positions: Optional <int32>[batch, length] encoder input subsequence
        positions for packed examples.
      enable_dropout: Enables dropout if set to True.

    Returns:
      <float>[batch, length, out_dim] result of self attention.
    """
    raise NotImplementedError




class EncoderLocalSelfAttention(nn.Module, LongSelfAttention):
  """Local bidirectional sliding window self attention.

  This implements self-attention analogous to `MultiHeadDotProductAttention`
  but only applied to a local window of `local_radius` tokens to the left and to
  the right of each token.  Unlike a "blocked" approach, this is a "sliding
  window" approach that can grow the receptive field with multiple stacks.

  Attributes:
    num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    local_radius: how many tokens to the left/right for each token to locally
      self-attend to. For example, a value of 1 would allow each token to only
      attend to 1 token to the left and 1 token to the right of it. TPU-friendly
      values include 84, 127, 169, 255, with 127 being the LongT5 default.
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
    bias_init: initializer for the bias of the Dense layers.
    use_bias: bool: whether pointwise QKVO dense transforms use bias.
    rescale_logits: bool. Whether to rescale `query` logits by 1/sqrt(depth_kq).
    use_extra_logit: whether to include a virtual extra logit equal to zero.
    float32_logits: bool, if True then compute logits in float32 to avoid
      numerical issues with bfloat16.
    output_projection: Project the output of `attention_fn` to `out_features`.
      If False, returns the output of `attention_fn` without a projection.
    split_head_kernel: whether to store QKVO variables with a split head
      dimension.
    kernels_to_fuse: Which kernels to fuse, if any.
    concat_3_blocks_implementation: Optional string specifying an alternative
      (but functionally equivalanet) local sparsity implementation.  Leave as
      `None` to use the default implementation.  The only current alternative is
      'onehot', which is more efficient when training with `scan`.
    relpos_bias: `RelativePositionBiasesGeneral` module to use for relative
      attention.
  """
  num_heads: int
  local_radius: int = 127  # TPU-friendly values include 84, 127, 169, 255
  dtype: DType = jnp.float32
  qkv_features: Optional[int] = None
  head_dim: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.
  precision: Any = None
  kernel_init: Initializer = default_kernel_init  # pytype: disable=annotation-type-mismatch  # jax-types
  bias_init: Initializer = initializers.zeros
  use_bias: bool = True
  rescale_logits: bool = False
  use_extra_logit: bool = False
  float32_logits: bool = False
  output_projection: bool = True
  split_head_kernel: bool = False
  kernels_to_fuse: Optional[str] = None  # Only 'qkv' is supported.
  use_rotary_embedding: bool = False
  rotary_embedding_max_timescale: float = 1e4
  concat_3_blocks_implementation: Optional[str] = None
  relpos_bias: Optional[RelativePositionBiasesGeneral] = None

  @nn.compact
  def __call__(self,
               inputs: Array,
               inputs_mask: Array,
               *,
               segment_ids: Optional[Array] = None,
               positions: Optional[Array] = None,
               enable_dropout: bool = True) -> Array:
    """Calls the attention layer (see `LongSelfAttention`)."""
    validate_long_attention_call_parameter_shapes(inputs, inputs_mask,
                                                  segment_ids, positions)

    block_len = self.local_radius + 1

    # [batch, num_blocks, 1, block_len, 3 * block_len] shape.
    mask = tensor_utils.make_3block_local_att_mask(
        block_len, inputs_mask, segment_ids)[:, :, jnp.newaxis, :, :]
    attention_bias = mask_to_bias(mask, self.dtype)

    if self.relpos_bias:
      # [block_len, 3 * block_len]
      relative_position = tensor_utils.make_3block_relative_position(block_len)
      rp_bucket = RelativePositionBiasesGeneral.relative_position_bucket(
          relative_position,
          bidirectional=True,
          num_buckets=self.relpos_bias.num_buckets,
          max_distance=self.relpos_bias.max_distance)

      # [1, 1, num_heads, block_len, 3 * block_len]
      bias = self.relpos_bias(rp_bucket)[jnp.newaxis, ...]  # pylint: disable=not-callable
      attention_bias += bias

    features = self.out_features or inputs.shape[-1]
    qkv_features = self.qkv_features or inputs.shape[-1]
    if self.head_dim is None:
      head_dim = qkv_features // self.num_heads
    else:
      head_dim = self.head_dim

    if self.kernels_to_fuse and not self.split_head_kernel:
      raise ValueError('Un-reshaped kernels are required when using QKV fused '
                       'kernel optimization.')

    # Is attention logit rescaling explicit or folded into initializer?
    if self.rescale_logits:
      query_init = self.kernel_init
    else:
      if self.kernels_to_fuse:
        raise ValueError('Cannot fold in logit normalization to query '
                         'initializer when using fused kernels.')
      depth_scaling = jnp.sqrt(head_dim).astype(self.dtype)
      query_init = lambda *args: self.kernel_init(*args) / depth_scaling

    make_dense = functools.partial(
        dense.DenseGeneral,
        axis=-1,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        precision=self.precision,
        reshape_kernel=not self.split_head_kernel,
    )

    # Project inputs to multi-headed q/k/v
    # dimensions are then [batch..., length, num_heads, features_per_head]
    if self.kernels_to_fuse is None:
      query = make_dense(
          kernel_init=query_init,
          features=(self.num_heads, head_dim),
          kernel_axis_names=['embed', 'heads', 'kv'],
          name='query')(
              inputs)
      key = make_dense(
          kernel_init=self.kernel_init,
          features=(self.num_heads, head_dim),
          kernel_axis_names=['embed', 'heads', 'kv'],
          name='key')(
              inputs)
      value = make_dense(
          kernel_init=self.kernel_init,
          features=(self.num_heads, head_dim),
          kernel_axis_names=['embed', 'heads', 'kv'],
          name='value')(
              inputs)
    elif self.kernels_to_fuse == 'qkv':
      qkv = make_dense(
          kernel_init=self.kernel_init,
          features=(3, self.num_heads, head_dim),
          kernel_axis_names=['embed', 'stack', 'heads', 'kv'],
          name='qkv_fused')(
              inputs)
      query = jnp.squeeze(lax.dynamic_slice_in_dim(qkv, 0, 1, -3), -3)
      key = jnp.squeeze(lax.dynamic_slice_in_dim(qkv, 1, 1, -3), -3)
      value = jnp.squeeze(lax.dynamic_slice_in_dim(qkv, 2, 1, -3), -3)
    else:
      raise ValueError(
          f'Unsupported kernel fusion mode: "{self.kernels_to_fuse}"')

    dropout_rng = None
    if enable_dropout and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    if self.use_rotary_embedding:
      length = inputs.shape[-2]
      sin, cos = embedding.generate_fixed_pos_embedding(
          head_dim, length, max_timescale=self.rotary_embedding_max_timescale
      )
      query, key = embedding.apply_rotary_embedding(
          query, key, cos, sin, decode=False, rotary_index=None
      )

    # Apply attention.
    x = _local_self_attention(
        query,
        key,
        value,
        local_radius=self.local_radius,
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
        concat_3_blocks_implementation=self.concat_3_blocks_implementation)  # pytype: disable=wrong-keyword-args

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


def _local_self_attention(query: Array,
                          key: Array,
                          value: Array,
                          local_radius: int,
                          bias: Optional[Array] = None,
                          broadcast_dropout: bool = True,
                          rescale_logits: bool = False,
                          dropout_rng: Optional[PRNGKey] = None,
                          dropout_rate: float = 0.,
                          enable_dropout: bool = True,
                          dtype: DType = jnp.float32,
                          precision: Optional[lax.Precision] = None,
                          use_extra_logit: bool = False,
                          float32_logits: bool = False,
                          concat_3_blocks_implementation: Optional[str] = None):
  """Sliding window local self attention.

  This is analogous to `dot_product_attention` but only permits attention
  between tokens that are within `local_radius` of each other.  This reduces
  length-dependent complexity from O(N^2) to O(NR), where N is the sequence
  length and R is `local_radius`.  Only self-attention is supported, not
  cross attention.

  The current implementation mirrors the original implementation used in ETC
  (https://arxiv.org/abs/2004.08483).

  Args:
    query: queries for calculating attention with shape of `[batch..., length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch..., length,
      num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of `[batch..., length,
      num_heads, v_depth_per_head]`.
    local_radius: How many tokens to the left/right for input tokens to locally
      self-attend to.  For example, a value of 1 would allow each token to only
      attend to 1 token to the left and 1 token to the right of it. TPU-friendly
      values include 84, 127, 169, and 255 since the internal `block_len` will
      be `local_radius + 1`.
    bias: bias for the attention weights.  This should be broadcastable to the
      shape `[batch..., num_blocks, num_heads, block_len, 3 * block_len]`. This
      can be used for incorporating causal masks, padding masks, proximity bias,
      etc.  Note that `bias` must be responsible for enforcing that tokens do
      not attend beyond `local_radius` since the 3-block approach technically
      permits attention to tokens up to `2 * local_radius + 1` away.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    rescale_logits: bool. Whether to rescale `query` logits by 1/sqrt(depth_kq).
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    enable_dropout: bool, enable_dropout or not (to apply dropout)
    dtype: the dtype of the computation (default: float32)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    use_extra_logit: whether to include a virtual extra logit equal to zero.
    float32_logits: bool, if True then compute logits in float32 to avoid
      numerical issues with bfloat16.
    concat_3_blocks_implementation: Optional string specifying an alternative
      implementation to use.  Leave as `None` to use the default implementation.
      The only current alternative is 'onehot', which is more efficient when
      training with `scan`.

  Returns:
    Output of shape `[batch..., length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
      'q, k, v batch dims must match.')
  assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
      'q, k, v num_heads must match.')
  assert query.shape[-3] == key.shape[-3] == value.shape[-3], (
      'q, k, v lengths must match.')
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  concat_3_blocks = _get_concat_3_blocks_implementation(
      concat_3_blocks_implementation)

  # calculate attention matrix
  # NOTE: T5 does not explicitly rescale the attention logits by
  #       1/sqrt(depth_kq)!  This is folded into the initializers of the
  #       linear transformations, which is equivalent under Adafactor.
  if rescale_logits:
    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)

  # split into blocks
  seq_len = query.shape[-3]
  block_len = local_radius + 1
  # [batch..., num_blocks, block_len, num_heads, *_depth_per_head] shape
  query = tensor_utils.split_into_blocks(query, block_len, axis=-3)
  key = tensor_utils.split_into_blocks(key, block_len, axis=-3)
  value = tensor_utils.split_into_blocks(value, block_len, axis=-3)

  # concatenate 3 blocks for keys and values
  # [batch..., num_blocks, 3 * block_len, num_heads, *_depth_per_head] shape
  key = concat_3_blocks(key, block_axis=-4, seq_axis=-3)
  value = concat_3_blocks(value, block_axis=-4, seq_axis=-3)

  # Casting logits and softmax computation for float32 for model stability.
  if float32_logits:
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)

  # [batch..., num_blocks, num_heads, block_len, 3 * block_len] shape
  attn_weights = jnp.einsum(
      '...qhd,...khd->...hqk', query, key, precision=precision)

  # apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias.astype(attn_weights.dtype)

  # normalize the attention weights
  attn_weights = (_softmax_with_extra_logit if use_extra_logit else
                  jax.nn.softmax)(attn_weights).astype(dtype)

  # apply attention dropout
  if enable_dropout and dropout_rate > 0.:
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
    multiplier = (
        keep.astype(attn_weights.dtype) / jnp.asarray(keep_prob, dtype=dtype))
    attn_weights = attn_weights * multiplier

  # compute weighted sum over values for each query position
  # [batch..., num_blocks, block_len, num_heads, v_depth_per_head] shape
  y = jnp.einsum(
      '...hqk,...khd->...qhd', attn_weights, value, precision=precision)

  # undo blocking and return results
  unblocked_output = y.reshape(y.shape[:-4] + (-1,) + y.shape[-2:])
  return unblocked_output[..., :seq_len, :, :]


def _get_concat_3_blocks_implementation(name: Optional[str]):
  if name is None:
    return tensor_utils.concat_3_blocks
  elif name == 'onehot':
    return tensor_utils.concat_3_blocks_one_hot
  else:
    raise ValueError(f'Unknown concat_3_blocks implementation: {name}')




class EtcTransientGlobalSelfAttention(nn.Module, LongSelfAttention):
  """ETC-like self-attention with transient globals only.

  This augments `EncoderLocalSelfAttention` with transiently constructed
  global tokens as side inputs to attend to in addition to local self-attention.
  The transient "global tokens" are computed as averages of the long input
  tokens in a "fixed blocks" pattern.  These block-average global tokens are
  computed at each layer and thrown away after the attention operation,
  allowing simpler drop-in replacement in the Transformer API since there
  isn't a separate "global input" array carried along from layer to layer.
  The configuration can be thought of as something like ETC without g2l and g2g
  components, only l2l and l2g.  (See https://arxiv.org/abs/2004.08483 for
  more about ETC.)

  Attributes:
    num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    tokens_per_block: positive integer number of tokens per transient global
      token. Typical values are 16 or 32.
    local_radius: how many tokens to the left/right for each token to locally
      self-attend to. For example, a value of 1 would allow each token to only
      attend to 1 token to the left and 1 token to the right of it. TPU-friendly
      values include 84, 127, 169, 255, with 127 being the LongT5 default.
    causal: bool. Whether to causally mask attention. Default false.
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
    bias_init: initializer for the bias of the Dense layers.
    use_bias: bool: whether pointwise QKVO dense transforms use bias.
    rescale_logits: bool. Whether to rescale `query` logits by 1/sqrt(depth_kq).
    use_extra_logit: whether to include a virtual extra logit equal to zero.
    float32_logits: bool, if True then compute logits in float32 to avoid
      numerical issues with bfloat16.
    output_projection: Project the output of `attention_fn` to `out_features`.
      If False, returns the output of `attention_fn` without a projection.
    split_head_kernel: whether to store QKVO variables with a split head
      dimension.
    kernels_to_fuse: Which kernels to fuse, if any.
    concat_3_blocks_implementation: Optional string specifying an alternative
      (but functionally equivalanet) local sparsity implementation.  Leave as
      `None` to use the default implementation.  The only current alternative is
      'onehot', which is more efficient when training with `scan`.
    relpos_bias: `RelativePositionBiasesGeneral` module to use for relative
      attention between input tokens (local).
    side_relpos_bias: `RelativePositionBiasesGeneral` module to use for relative
      attention from input tokens to transient globals.
  """
  num_heads: int
  tokens_per_block: int  # Typical values are 16 or 32.
  local_radius: int = 127  # TPU-friendly values include 84, 127, 169, 255.
  causal: bool = False
  dtype: DType = jnp.float32
  qkv_features: Optional[int] = None
  head_dim: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.
  precision: Any = None
  kernel_init: Initializer = default_kernel_init  # pytype: disable=annotation-type-mismatch  # jax-types
  bias_init: Initializer = initializers.zeros
  use_bias: bool = True
  rescale_logits: bool = False
  use_extra_logit: bool = False
  float32_logits: bool = False
  output_projection: bool = True
  split_head_kernel: bool = False
  kernels_to_fuse: Optional[str] = None  # Only 'kv' is supported.
  concat_3_blocks_implementation: Optional[str] = None
  relpos_bias: Optional[RelativePositionBiasesGeneral] = None
  side_relpos_bias: Optional[RelativePositionBiasesGeneral] = None

  @nn.compact
  def __call__(self,
               inputs: Array,
               inputs_mask: Array,
               *,
               segment_ids: Optional[Array] = None,
               positions: Optional[Array] = None,
               enable_dropout: bool = True) -> Array:
    """Calls the attention layer (see `LongSelfAttention`)."""
    validate_long_attention_call_parameter_shapes(inputs, inputs_mask,
                                                  segment_ids, positions)

    block_len = self.local_radius + 1

    # [batch, num_blocks, 1, block_len, 3 * block_len] shape.
    mask = tensor_utils.make_3block_local_att_mask(
        block_len, inputs_mask, segment_ids,
        use_causal_mask=self.causal)[:, :, jnp.newaxis, :, :]
    attention_bias = mask_to_bias(mask, self.dtype)

    if self.relpos_bias:
      # [block_len, 3 * block_len] shape.
      relative_position = tensor_utils.make_3block_relative_position(block_len)
      rp_bucket = RelativePositionBiasesGeneral.relative_position_bucket(
          relative_position,
          bidirectional=not self.causal,
          num_buckets=self.relpos_bias.num_buckets,
          max_distance=self.relpos_bias.max_distance)

      # [1, 1, num_heads, block_len, 3 * block_len] shape.
      bias = self.relpos_bias(rp_bucket)[jnp.newaxis, ...]  # pylint: disable=not-callable
      attention_bias += bias

    # Create side attention bias.
    block_ids, global_segment_ids = make_etc_fixed_block_ids(
        self.tokens_per_block,
        inputs_mask,
        segment_ids,
        positions,
        adopt_orphan_tokens=not self.causal)
    global_seq_len = global_segment_ids.shape[-1]
    if segment_ids is None:
      segment_ids = jnp.asarray(inputs_mask, jnp.int32)
    # [batch, seq_len, global_seq_len] shape.
    side_mask = jnp.equal(segment_ids[..., jnp.newaxis],
                          global_segment_ids[..., jnp.newaxis, :])
    # [batch, 1, seq_len, global_seq_len] shape.
    side_mask = side_mask[..., jnp.newaxis, :, :]
    attention_side_bias = mask_to_bias(side_mask, self.dtype)

    global_positions = jnp.arange(global_seq_len)
    if self.causal:
      orphans = identify_orphan_tokens(self.tokens_per_block, inputs_mask,
                                       positions)

      # Below is a slight hack to ensure that orphan tokens can attend to the
      # global tokens. By definition, orphan tokens can attend to all global
      # tokens in their segment; so, we set their "effective" block_id to be
      # global_seq_len, as this is greater than all global_positions and will
      # thus always satisfy the causality condition.
      effective_block_ids = block_ids * (1 - orphans) + global_seq_len * orphans
      causal_side_mask = jnp.less(global_positions,
                                  effective_block_ids[..., :, jnp.newaxis])
      causal_side_mask = causal_side_mask[..., jnp.newaxis, :, :]
      causal_side_bias = mask_to_bias(causal_side_mask, self.dtype)
      attention_side_bias += causal_side_bias

    if self.side_relpos_bias is None:
      raise ValueError('`side_relpos_bias` must be given.')

    side_relative_position = _make_side_relpos(
        self.tokens_per_block,
        inputs_mask,
        segment_ids,
        positions,
        adopt_orphan_tokens=not self.causal)

    side_rp_bucket = RelativePositionBiasesGeneral.relative_position_bucket(
        side_relative_position,
        bidirectional=not self.causal,
        num_buckets=self.side_relpos_bias.num_buckets,
        max_distance=self.side_relpos_bias.max_distance)
    # [1, num_heads, batch, seq_len, global_seq_len] shape.
    side_bias = self.side_relpos_bias(side_rp_bucket)  # pylint: disable=not-callable
    # [batch, num_heads, seq_len, global_seq_len] shape.
    side_bias = jnp.swapaxes(side_bias[0], -4, -3)
    attention_side_bias += side_bias

    features = self.out_features or inputs.shape[-1]
    qkv_features = self.qkv_features or inputs.shape[-1]
    if self.head_dim is None:
      head_dim = qkv_features // self.num_heads
    else:
      head_dim = self.head_dim

    if self.kernels_to_fuse and not self.split_head_kernel:
      raise ValueError('Un-reshaped kernels are required when using QKV fused '
                       'kernel optimization.')

    # Is attention logit rescaling explicit or folded into initializer?
    if self.rescale_logits:
      query_init = self.kernel_init
    else:
      if self.kernels_to_fuse:
        raise ValueError('Cannot fold in logit normalization to query '
                         'initializer when using fused kernels.')
      depth_scaling = jnp.sqrt(head_dim).astype(self.dtype)
      query_init = lambda *args: self.kernel_init(*args) / depth_scaling

    # Create global aggregates.
    global_inputs = _create_global_aggregates(inputs, block_ids, global_seq_len)
    global_inputs = layer_norm.T5LayerNorm(dtype=self.dtype)(global_inputs)

    make_dense = functools.partial(
        dense.DenseGeneral,
        axis=-1,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        precision=self.precision,
        reshape_kernel=not self.split_head_kernel,
    )

    # Project inputs to multi-headed q/k/v
    # dimensions are then [batch..., length, num_heads, features_per_head]
    if self.kernels_to_fuse is None:
      query = make_dense(
          kernel_init=query_init,
          features=(self.num_heads, head_dim),
          kernel_axis_names=['embed', 'heads', 'kv'],
          name='query')(
              inputs)
      key_dense = make_dense(
          kernel_init=self.kernel_init,
          features=(self.num_heads, head_dim),
          kernel_axis_names=['embed', 'heads', 'kv'],
          name='key')
      value_dense = make_dense(
          kernel_init=self.kernel_init,
          features=(self.num_heads, head_dim),
          kernel_axis_names=['embed', 'heads', 'kv'],
          name='value')
      key = key_dense(inputs)
      value = value_dense(inputs)
      # Share global key/value projections with long input for now.
      side_key = key_dense(global_inputs)
      side_value = value_dense(global_inputs)
    elif self.kernels_to_fuse == 'kv':
      query = make_dense(
          kernel_init=query_init,
          features=(self.num_heads, head_dim),
          kernel_axis_names=['embed', 'heads', 'kv'],
          name='query')(
              inputs)
      kv_dense = make_dense(
          kernel_init=self.kernel_init,
          features=(2, self.num_heads, head_dim),
          kernel_axis_names=['embed', 'stack', 'heads', 'kv'],
          name='kv_fused')
      kv = kv_dense(inputs)
      key = jnp.squeeze(lax.dynamic_slice_in_dim(kv, 0, 1, -3), -3)
      value = jnp.squeeze(lax.dynamic_slice_in_dim(kv, 1, 1, -3), -3)
      # Share global key/value projections with long input for now.
      side_kv = kv_dense(global_inputs)
      side_key = jnp.squeeze(lax.dynamic_slice_in_dim(side_kv, 0, 1, -3), -3)
      side_value = jnp.squeeze(lax.dynamic_slice_in_dim(side_kv, 1, 1, -3), -3)
    else:
      raise ValueError(
          f'Unsupported kernel fusion mode: "{self.kernels_to_fuse}"')

    dropout_rng = None
    if enable_dropout and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    # Apply attention.
    x = _local_plus_side_attention(
        query,
        key,
        value,
        side_key,
        side_value,
        local_radius=self.local_radius,
        bias=attention_bias,
        side_bias=attention_side_bias,
        broadcast_dropout=self.broadcast_dropout,
        rescale_logits=self.rescale_logits,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        enable_dropout=enable_dropout,
        dtype=self.dtype,
        precision=self.precision,
        use_extra_logit=self.use_extra_logit,
        float32_logits=self.float32_logits,
        concat_3_blocks_implementation=self.concat_3_blocks_implementation)  # pytype: disable=wrong-keyword-args

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


def make_etc_fixed_block_ids(
    tokens_per_block: int,
    inputs_mask: Array,
    segment_ids: Optional[Array] = None,
    positions: Optional[Array] = None,
    adopt_orphan_tokens: bool = True) -> Tuple[Array, Array]:
  """Returns the "fixed block" global id corresponding to each long token.

  The array arguments follow `LongSelfAttention`.

  Args:
    tokens_per_block: Integer number of input tokens assigned to each "block"
      corresponding to a global token.  Note that "blocks" in this sense have no
      connection with the internal "blocks" used for implementing sliding window
      local self-attention.
    inputs_mask: <bool>[batch, seq_len] shaped Array.
    segment_ids: Optional <int32>[batch, seq_len] shaped Array.
    positions: Optional <int32>[batch, seq_len] shaped Array.
    adopt_orphan_tokens: bool, determining the behavior when sequence lengths in
      the input do not evenly divide by tokens_per_block. See 'Note on orphan
      tokens' in the docstring of the helper function
      _make_etc_fixed_block_ids_1d().

  Returns:
    (block_ids, global_segment_ids) Tuple:
      block_ids: <int32>[batch, seq_len] shaped Array of global token ids for
        each (long) input token.  Long input tokens that aren't assigned any
        global tokens will have id `-1` (which will be the case for any examples
        that have fewer than `tokens_per_block` tokens).
      global_segment_ids: <int32>[batch, global_seq_len] shaped Array of the
        "segment id" (i.e. example id) each global token belongs to.
        `global_seq_len` is inferred as `seq_len // tokens_per_block`.
  """
  inputs_mask = jnp.asarray(inputs_mask)
  segment_ids = (
      inputs_mask.astype(jnp.int32)
      if segment_ids is None else jnp.asarray(segment_ids))
  positions = (
      jnp.arange(inputs_mask.shape[-1]) *
      inputs_mask if positions is None else jnp.asarray(positions))

  vmapped_fn = jax.vmap(
      _make_etc_fixed_block_ids_1d, in_axes=(None, 0, 0, 0, None), out_axes=0)
  return vmapped_fn(tokens_per_block, inputs_mask, segment_ids, positions,
                    adopt_orphan_tokens)


def _make_etc_fixed_block_ids_1d(
    tokens_per_block: int,
    inputs_mask: Array,
    segment_ids: Array,
    positions: Array,
    adopt_orphan_tokens: bool = True) -> Tuple[Array, Array]:
  """Helper for `make_etc_fixed_block_ids` applied to a single example.

  See the following for an example of what packed inputs look like:
  https://github.com/google/seqio/blob/main/seqio/utils.py#L292

  Args:
    tokens_per_block: Positive integer.
    inputs_mask: <bool>[seq_len] shaped Array.
    segment_ids: <int32>[seq_len] shaped Array.
    positions: <int32>[seq_len] shaped Array.
    adopt_orphan_tokens: bool, determining the behavior when sequence lengths in
      the input do not evenly divide by tokens_per_block. See 'Note on orphan
      tokens' below

  Returns:
    (block_ids, global_segment_ids) Tuple:
      block_ids: <int32>[seq_len] shaped Array of global token ids for each
        (long) input token.  Long tokens that aren't assigned any global
        tokens will have id `-1`.
      global_segment_ids: <int32>[global_seq_len] shaped Array of the "segment
        id" (i.e. example id) each global token belongs to.  `global_seq_len`
        is inferred as `seq_len // tokens_per_block`.

  Note on orphan tokens:
     If a sequence in the provided input has a length which does not evenly
     divide by tokens_per_block, the final tokens which do not correspond
     naturally to a block are known as orphan tokens.  There are two ways this
     function may assign a block number to these orphans, depending on the value
     of the (bool) adopt_orphan_tokens argument.
     If adopt_orphan_tokens == True,
       the orphan tokens will be assigned to the same block as the final
       non-orphan tokens. E.g., if tokens_per_block == 2, and the inputs_mask is
       jnp.array([1,1,1,0]), the block_ids returned by this function will be
       jnp.array([0,0,0,-1]), indicated that the orphan token in position 2 is
       effectively a member of the same block as the non-orphan tokens in pos 0
       and 1.
    If adopt_orphan_tokens == False,
       the orphan tokens will not be assigned to a global-token block.  In this
       case, if tokens_per_block == 2, and the inputs_mask is
       jnp.array([1,1,1,0]), the block_ids returned by this function will be
       jnp.array([0,0,-1,-1]), indicating that the orphan token in position 2 is
       NOT a member of the same block as the non-orphan tokens in pos 0 and 1.
  """
  assert 1 == inputs_mask.ndim == segment_ids.ndim == positions.ndim
  assert inputs_mask.shape[0] == segment_ids.shape[0] == positions.shape[0]

  seq_len = inputs_mask.shape[0]
  num_globals = seq_len // tokens_per_block

  position_mod = positions % tokens_per_block
  start_marker = position_mod == 0
  end_marker = position_mod == tokens_per_block - 1
  candidate_blocks = jnp.cumsum(start_marker, axis=-1) * inputs_mask - 1

  positions_start_end = positions * jnp.logical_or(start_marker, end_marker)
  candidate_block_sums = jax.ops.segment_sum(
      positions_start_end, candidate_blocks, num_segments=seq_len)

  blocks_with_starts = candidate_block_sums % tokens_per_block != 0
  global_start = jnp.logical_and(start_marker,
                                 blocks_with_starts[candidate_blocks])

  blocks_without_globals = candidate_block_sums == 0
  token_without_global = blocks_without_globals[candidate_blocks]
  token_without_global = jnp.logical_or(token_without_global,
                                        jnp.logical_not(inputs_mask))

  block_ids = jnp.cumsum(global_start) * jnp.logical_not(
      token_without_global) - 1
  global_segment_ids = jax.ops.segment_sum(
      segment_ids * global_start, block_ids, num_segments=num_globals)

  if not adopt_orphan_tokens:
    orphan_tokens = _identify_orphan_tokens(tokens_per_block, inputs_mask,
                                            positions)
    not_orphan_tokens = 1 - orphan_tokens

    orphan_indicator = -1
    block_ids = not_orphan_tokens * block_ids + orphan_tokens * orphan_indicator
    block_ids = block_ids.astype(jnp.int32)

  return block_ids, global_segment_ids


def identify_orphan_tokens(tokens_per_block: int,
                           inputs_mask: Array,
                           positions: Optional[Array] = None) -> Array:
  """Returns an Array with 1s in places corresponding to "orphan" tokens.

  The array arguments follow `LongSelfAttention`, with the exception of
  segment_ids, which is not needed.

  Args:
    tokens_per_block: Integer number of input tokens assigned to each "block"
      corresponding to a global token.  Note that "blocks" in this sense have no
      connection with the internal "blocks" used for implementing sliding window
      local self-attention.
    inputs_mask: <bool>[batch, seq_len] shaped Array.
    positions: Optional <int32>[batch, seq_len] shaped Array.

  Returns:
    orphan_tokens: <int32>[batch, seq_len] shaped Array of orphan indicators for
        each (long) input token.  orphan_tokens has 1s in positions which
        correspond to orphan tokens, and 0s in all other positions. See note
        below.

  Note on orphan tokens:
     If a sequence in the provided input has a length which does not evenly
     divide by tokens_per_block, the final tokens which do not correspond
     naturally to a block are known as orphan tokens.
  """
  inputs_mask = jnp.asarray(inputs_mask)
  positions = (
      jnp.arange(inputs_mask.shape[-1]) *
      inputs_mask if positions is None else jnp.asarray(positions))

  vmapped_fn = jax.vmap(
      _identify_orphan_tokens, in_axes=(None, 0, 0), out_axes=0)
  return vmapped_fn(tokens_per_block, inputs_mask, positions)


def _identify_orphan_tokens(tokens_per_block: int, inputs_mask: Array,
                            positions: Array) -> Array:
  """Helper for `identify_orphan_tokens` applied to a single example.

  The array arguments follow `LongSelfAttention`.

  Args:
    tokens_per_block: Integer number of input tokens assigned to each "block"
      corresponding to a global token.  Note that "blocks" in this sense have no
      connection with the internal "blocks" used for implementing sliding window
      local self-attention.
    inputs_mask: <bool>[seq_len] shaped Array.
    positions: Optional <int32>[seq_len] shaped Array.

  Returns:
    orphan_tokens: <int32>[seq_len] shaped Array of orphan indicators for
        each (long) input token.  orphan_tokens has 1s in positions which
        correspond to orphan tokens, and 0s in all other positions. See note
        below.

  Note on orphan tokens:
     If a sequence in the provided input has a length which does not evenly
     divide by tokens_per_block, the final tokens which do not correspond
     naturally to a block are known as orphan tokens.
     For example, if the input mask is [1,1,1,1,1,1,1,1,0,0], and the number of
     tokens per global block is 3, the final two tokens in the sequence are
     orphans, and the returned value of orphan_tokens is [0,0,0,0,0,0,1,1,0,0].
  """

  position_mod = positions % tokens_per_block
  end_marker = position_mod == tokens_per_block - 1

  k = jnp.ones(tokens_per_block)
  x = 1 - jnp.correlate(end_marker, k, mode='full')[tokens_per_block - 1:]

  return jnp.logical_and(x.astype(jnp.int32), inputs_mask).astype(jnp.int32)


def _create_global_aggregates(inputs: Array, block_ids: Array,
                              global_seq_len: int) -> Array:
  """Computes global aggregates by summing embeddings in each block.

  Args:
    inputs: <float>[batch..., seq_len, hidden_size] array of token embeddings to
      aggregate over.
    block_ids: <int32>[batch..., seq_len] array indicating the block (i.e.
      global token) ids for each token in `inputs`.  Only ids in the range [0,
      global_seq_len] will be aggregated.
    global_seq_len: integer number of global tokens to return in the result.

  Returns:
    [batch..., global_seq_len, hidden_size] array of global aggregates taken
    by summing.
  """
  # [batch..., seq_len, global_seq_len] shape.
  one_hot_block_ids = jax.nn.one_hot(block_ids, global_seq_len)

  return jnp.einsum('...nd,...ng->...gd', inputs, one_hot_block_ids)


def _local_plus_side_attention(
    query: Array,
    key: Array,
    value: Array,
    side_key: Array,
    side_value: Array,
    local_radius: int,
    bias: Optional[Array] = None,
    side_bias: Optional[Array] = None,
    broadcast_dropout: bool = True,
    rescale_logits: bool = False,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.,
    enable_dropout: bool = True,
    dtype: DType = jnp.float32,
    precision: Optional[lax.Precision] = None,
    use_extra_logit: bool = False,
    float32_logits: bool = False,
    concat_3_blocks_implementation: Optional[str] = None):
  """Local self attention with side keys/values (e.g.

  from global memory).

  This is an extension to `_local_self_attention` that also attends to a side
  input in addition to the local window.

  Args:
    query: queries for calculating attention with shape of `[batch..., length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch..., length,
      num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of `[batch..., length,
      num_heads, v_depth_per_head]`.
    side_key: `[batch..., side_len, num_heads, qk_depth_per_head]` array of keys
      for the side input.
    side_value: `[batch..., side_len, num_heads, v_depth_per_head]` array of
      values for the side input.
    local_radius: How many tokens to the left/right for input tokens to locally
      self-attend to.  For example, a value of 1 would allow each token to only
      attend to 1 token to the left and 1 token to the right of it. TPU-friendly
      values include 84, 127, 169, and 255 since the internal `block_len` will
      be `local_radius + 1`.
    bias: bias for the attention weights.  This should be broadcastable to the
      shape `[batch..., num_blocks, num_heads, block_len, 3 * block_len]`. This
      can be used for incorporating causal masks, padding masks, proximity bias,
      etc.  Note that `bias` must be responsible for enforcing that tokens do
      not attend beyond `local_radius` since the 3-block approach technically
      permits attention to tokens up to `2 * local_radius + 1` away.
    side_bias: `[batch..., num_heads, length, side_len]` shaped array for side
      input attention bias.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    rescale_logits: bool. Whether to rescale `query` logits by 1/sqrt(depth_kq).
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    enable_dropout: bool, enable_dropout or not (to apply dropout)
    dtype: the dtype of the computation (default: float32)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    use_extra_logit: whether to include a virtual extra logit equal to zero.
    float32_logits: bool, if True then compute logits in float32 to avoid
      numerical issues with bfloat16.
    concat_3_blocks_implementation: Optional string specifying an alternative
      implementation to use.  Leave as `None` to use the default implementation.
      The only current alternative is 'onehot', which is more efficient when
      training with `scan`.

  Returns:
    Output of shape `[batch..., length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
      'q, k, v batch dims must match.')
  assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
      'q, k, v num_heads must match.')
  assert query.shape[-3] == key.shape[-3] == value.shape[-3], (
      'q, k, v lengths must match.')
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  assert side_key.ndim == side_value.ndim
  assert query.shape[:-3] == side_key.shape[:-3] == side_value.shape[:-3], (
      'side k, v batch dims must match q.')
  assert query.shape[-2] == side_key.shape[-2] == side_value.shape[-2], (
      'side k, v num_heads must match q.')
  assert side_key.shape[-3] == side_value.shape[-3], (
      'side k, v lengths must must match.')
  assert query.shape[-1] == side_key.shape[-1], 'side k depth must match q.'
  assert value.shape[-1] == side_value.shape[-1], 'side v depth must match v.'

  assert (bias is None) == (side_bias is None), (
      'bias and side_bias must be either both present or both None')

  concat_3_blocks = _get_concat_3_blocks_implementation(
      concat_3_blocks_implementation)

  # NOTE: T5 does not explicitly rescale the attention logits by
  #       1/sqrt(depth_kq)!  This is folded into the initializers of the
  #       linear transformations, which is equivalent under Adafactor.
  if rescale_logits:
    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)

  # split into blocks
  seq_len = query.shape[-3]
  block_len = local_radius + 1
  # [batch..., num_blocks, block_len, num_heads, *_depth_per_head] shape.
  query = tensor_utils.split_into_blocks(query, block_len, axis=-3)
  key = tensor_utils.split_into_blocks(key, block_len, axis=-3)
  value = tensor_utils.split_into_blocks(value, block_len, axis=-3)

  # concatenate 3 blocks for keys and values
  # [batch..., num_blocks, 3 * block_len, num_heads, *_depth_per_head] shape.
  key = concat_3_blocks(key, block_axis=-4, seq_axis=-3)
  value = concat_3_blocks(value, block_axis=-4, seq_axis=-3)

  # casting logits and softmax computation for float32 for model stability.
  if float32_logits:
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    side_key = side_key.astype(jnp.float32)

  # tile side inputs across blocks
  num_blocks = query.shape[-4]
  reps = [1] * (side_key.ndim + 1)
  reps[-4] = num_blocks
  # [batch..., num_blocks, side_len, num_heads, *_depth_per_head] shape.
  tiled_side_key = jnp.tile(side_key[..., jnp.newaxis, :, :, :], reps)
  tiled_side_value = jnp.tile(side_value[..., jnp.newaxis, :, :, :], reps)

  # [batch..., num_blocks, 3 * block_len + side_len, num_heads,
  #   *_depth_per_head] shape.
  key = jnp.concatenate((key, tiled_side_key), axis=-3)
  value = jnp.concatenate((value, tiled_side_value), axis=-3)

  # [batch..., num_blocks, num_heads, block_len, 3 * block_len + side_len] shape
  attn_weights = jnp.einsum(
      '...qhd,...khd->...hqk', query, key, precision=precision)

  # apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    num_heads = query.shape[-2]

    # [batch..., num_blocks, num_heads, block_len, 3 * block_len] shape.
    bias = jnp.broadcast_to(
        bias, attn_weights.shape[:-4] +
        (num_blocks, num_heads, block_len, 3 * block_len))

    # [batch..., num_heads, num_blocks, block_len, side_len] shape.
    side_bias = tensor_utils.split_into_blocks(side_bias, block_len, axis=-2)

    # [batch..., num_blocks, num_heads, block_len, side_len] shape.
    side_bias = jnp.swapaxes(side_bias, -4, -3)

    attn_weights += jnp.concatenate((bias, side_bias),
                                    axis=-1).astype(attn_weights.dtype)

  # normalize the attention weights
  attn_weights = (_softmax_with_extra_logit if use_extra_logit else
                  jax.nn.softmax)(attn_weights).astype(dtype)

  # apply attention dropout
  if enable_dropout and dropout_rate > 0.:
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
    multiplier = (
        keep.astype(attn_weights.dtype) / jnp.asarray(keep_prob, dtype=dtype))
    attn_weights = attn_weights * multiplier

  # compute weighted sum over values for each query position
  # [batch..., num_blocks, block_len, num_heads, v_depth_per_head] shape.
  y = jnp.einsum(
      '...hqk,...khd->...qhd', attn_weights, value, precision=precision)

  # undo blocking and return results
  unblocked_output = y.reshape(y.shape[:-4] + (-1,) + y.shape[-2:])
  return unblocked_output[..., :seq_len, :, :]




def mask_to_bias(mask: Array, dtype: jnp.dtype) -> Array:
  """Converts a mask to a bias-like Array suitable for adding to other biases.

  Arguments:
    mask: <bool> array of arbitrary shape
    dtype: jnp.dtype, desired dtype of the returned array

  Returns:
    bias: <bool> array of the same shape as the input, with 0 in place of truthy
          values and -1e10 in place of falsy values of mask
  """

  return lax.select(mask,
                    jnp.full(mask.shape, 0).astype(dtype),
                    jnp.full(mask.shape, -1e10).astype(dtype))


def _make_side_relpos(tokens_per_block: int,
                      inputs_mask: Array,
                      segment_ids: Optional[Array] = None,
                      positions: Optional[Array] = None,
                      adopt_orphan_tokens: bool = True) -> Array:
  """Makes the relative position tensor for local -> global attention.

  Args:
    tokens_per_block: Integer number of input tokens assigned to each "block"
      corresponding to a global token.  Note that "blocks" in this sense have no
      connection with the internal "blocks" used for implementing sliding window
      local self-attention.
    inputs_mask: <bool>[batch, seq_len] shaped Array.
    segment_ids: Optional <int32>[batch, seq_len] shaped Array.
    positions: Optional <int32>[batch, seq_len] shaped Array.
    adopt_orphan_tokens: bool, determining the behavior when sequence lengths in
      the input do not evenly divide by tokens_per_block. See 'Note on orphan
      tokens' in the docstring of the helper function
      _make_etc_fixed_block_ids_1d().

  Returns:
    side_relative_position: <int32>[batch, seq_len, global_seq_len] shaped Array
                           of relative positions between the local tokens and
                           the corresponding global tokens in the segment.
  """
  block_ids, global_segment_ids = make_etc_fixed_block_ids(
      tokens_per_block,
      inputs_mask,
      segment_ids,
      positions,
      adopt_orphan_tokens=True)
  if not adopt_orphan_tokens:
    orphan_locations = identify_orphan_tokens(tokens_per_block, inputs_mask,
                                              positions)
    block_ids = block_ids + orphan_locations
  global_seq_len = global_segment_ids.shape[-1]
  global_positions = jnp.arange(global_seq_len, dtype=jnp.int32)
  side_relative_position = global_positions - block_ids[..., jnp.newaxis]
  return side_relative_position.astype(jnp.int32)




def validate_long_attention_call_parameter_shapes(
    inputs: Array,
    inputs_mask: Array,
    segment_ids: Optional[Array],
    positions: Optional[Array],
    *,
    allow_positions_without_segment_ids: bool = False) -> None:
  """Validates the shapes of parameters to LongSelfAttention call methods.

  Args:
      inputs: <float>[batch, length, emb_dim] array of embeddings to self-attend
        over.
      inputs_mask: <bool>[batch, length] array indicating True for non-padding
        tokens and False for padding.
      segment_ids: Optional <int32>[batch, length] encoder input segmentation
        info for packed examples.
      positions: Optional <int32>[batch, length] encoder input subsequence
        positions for packed examples.
      allow_positions_without_segment_ids: If True, `segment_ids` can be None
        while `positions` is given.  This will be the case for example if
        packing is off but tokens appear in a non-sequential order.

  Raises:
    ValueError if any arrays fail validation.
  """
  if inputs.ndim < 3:
    raise ValueError(f'Expected rank of inputs >= 3, was {inputs.ndim}')
  if inputs_mask.ndim != inputs.ndim - 1:
    raise ValueError(f'Mismatched ranks: expected '
                     f'inputs_mask.ndim ({inputs_mask.ndim}) to be one less '
                     f'than inputs.ndim ({inputs.ndim})')
  if inputs.shape[:-2] != inputs_mask.shape[:-1]:
    raise ValueError(f'Mismatched batch dims: expected '
                     f'inputs.shape[:-2] ({inputs.shape[:-2]}) == '
                     f'inputs_mask.shape[:-1] ({inputs_mask[:-1]})')
  if inputs.shape[-2] != inputs_mask.shape[-1]:
    raise ValueError(f'Mismatched length dim: expected '
                     f'inputs.shape[-2] ({inputs.shape[-2]}) == '
                     f'inputs_mask.shape[-1] ({inputs_mask[-1]})')

  if allow_positions_without_segment_ids:
    if positions is None and segment_ids is not None:
      raise ValueError(
          '`positions` must not be None when `segment_ids` is given')
  elif (segment_ids is None) != (positions is None):
    raise ValueError(
        f'segment_ids and positions must either be both given or both None '
        f'but got `segment_ids is None`: {segment_ids is None}, '
        f'`positions is None`: {positions is None}')
  if segment_ids is not None and segment_ids.shape != inputs_mask.shape:
    raise ValueError(f'Mismatched shapes: expected '
                     f'segment_ids.shape ({segment_ids.shape}) to match '
                     f'inputs_mask.shape ({inputs_mask.shape})')
  if positions is not None and positions.shape != inputs_mask.shape:
    raise ValueError(f'Mismatched shapes: expected '
                     f'positions.shape ({positions.shape}) to match '
                     f'inputs_mask.shape ({inputs_mask.shape})')
