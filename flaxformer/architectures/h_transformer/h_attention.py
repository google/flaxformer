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

r"""Hierarchical attention classes."""
import abc
import enum
import functools
from typing import Any, Callable, Dict, Optional, Union, Tuple

from absl import logging

from flax import linen as nn
import gin
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as np
from flaxformer.architectures.h_transformer import hierarchical_relative_position_bias as h_rpb
from flaxformer.architectures.h_transformer import partitioning
from flaxformer.architectures.h_transformer import token_hierarchy as th
from flaxformer.components import dense
from flaxformer.types import Array
from flaxformer.types import Initializer
from flaxformer.types import PRNGKey

AXES = partitioning.AxisName


@gin.constants_from_enum
class MaxSimilarityMode(str, enum.Enum):
  """Names of the mode for finding max similarity."""
  SAMPLE_ANCHOR = 'sample_anchor'
  SCAN_ANCHOR = 'scan_anchor'
  SCAN_ALL = 'scan_all'


class HierarchicalAttention(nn.Module, metaclass=abc.ABCMeta):
  """Hierarchical attention base class.

  This computes hierarchical multi-head dot-product attention with linear
  complexity in memory usage and runtime.

  This class can be used for encoder-only or decoder-only by giving the same
  inputs_kv and inputs_q in the call parameters. The attribute causal_mask is
  to be used to separate these two cases.

  It can also be used for encoder-decoder cross attention  by giving different
  inputs_kv and inputs_q in the call parameters. Note that the code assumes
  that the Query and Key/Value have the same spatial size (after padding)
  and they share the same token hierarchy. Hence this cross attention can
  be applied to the tasks like machine translation, but not to the tasks
  like summarization.

  Attributes:
    num_heads: Number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    num_clusters: Number of clusters at each level in the hierarchy. At level=0,
      this is the diagonal block size in the attention matrix.
    causal_mask: This specifies whether to apply a causal mask on the attention
      weights. If True, the output at timestep `t` will not depend on inputs at
      timesteps strictly greater than `t`.
    dtype: The dtype of the computation.
    qkv_features: Feature dimension of the key, query, and value.
    out_features: Feature dimension of the output updated value.
    broadcast_dropout: This indicates if a broadcasted dropout for attention
      weights is used along the batch dimension.
    dropout_rate: Dropout rate.
    precision: Numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: Initializer for the kernel of the Dense layers.
    bias_init: Initializer for the bias of the Dense layers.
    use_bias: Whether the Dense layers use bias.
    split_head_kernel: whether to store QKVO projection kernels with a split
      head dimension. Default to False so the kernels are stored in 2D shape for
      the compatibility with Adafactor optimizer.
    rescale_logits: bool. Whether to  explicitly rescale `query` logits by
      1/sqrt(depth_kq). Default is to do this implicitly by folding the
      rescaling into query_kernel_init.
    sharding_over_head_dimension: Whether to shard over the head dimension.
      Setting this to False when the number of heads is not divisible your
      activation num_partitions.
    use_rpb: Whether the hierarchical relative position bias is used. Default to
      True because this setting delivers better results.
    use_multihead_rpb: Whether the hierarchical relative position bias is
      different among multihead. If False, the same relative position bias is
      shared among all heads. Default to True so the bias array is stored in 2D
      shape for the compatibility with Adafactor optimizer.
    conv_kernel_size: Convolution kernel size used for interpolation. This is
      not used during interpolation if the attribute
      interpolation_kernel_type=ConvKernelType.LINEAR since the kernel size is
      fixed at 3.
    interpolation_kernel_type: Type of interpolation convolution kernels.
    use_mxu: Indicates if MXU function einsum is used.
    max_similarity_mode: Name of the mode to find max similarity.
    max_similarity_factor: This is a buffer factor to amplify the max similariy
      found for the anchor similarity block. We need a larger-than-one factor to
      approximate the global maximum similarity. This factor can be adjusted
      upward if we get a NAN runtime error which usually indicates the overflow
      in attention=exp(similarity). But excessively large offset could lead to
      underflow.
    use_row_sum: Indicates if row sum is used to compute softmax partition.
    multihead_projection: Indicates if the multihead projection is performed. In
      unit tests, turning this off avoids randomness in projection.
    output_projection: Project the output of `attention_fn` to `out_features`.
      If False, returns the output of `attention_fn` without a projection.
    softmax_temperature: Temperature parameter in softmax. Default=1.0. A larger
      temperature smooths the output distribution of the softmax.
  """

  num_heads: int = 8
  num_clusters: int = 2
  causal_mask: bool = False
  dtype: jnp.dtype = jnp.float32
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.1
  precision: Optional[jax.lax.Precision] = None
  kernel_init: Initializer = nn.linear.default_kernel_init  # pytype: disable=annotation-type-mismatch  # jax-types
  bias_init: Initializer = nn.initializers.zeros
  use_bias: bool = True
  split_head_kernel: bool = False
  rescale_logits: bool = True
  sharding_over_head_dimension: bool = True
  use_rpb: bool = True
  use_multihead_rpb: bool = True
  conv_kernel_size: int = 2
  interpolation_kernel_type: th.ConvKernelType = th.ConvKernelType.CONST
  use_mxu: bool = True
  max_similarity_mode: MaxSimilarityMode = MaxSimilarityMode.SCAN_ALL
  max_similarity_factor: float = 3.
  use_row_sum: bool = False
  multihead_projection: bool = True
  output_projection: bool = True
  softmax_temperature: float = 1.0
  enable_param_axes: bool = True
  partitioner_factory: Callable[[], Any] = partitioning.Partitioner1D

  def setup(self):
    self.partitioner = self.partitioner_factory()

  @nn.compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               query_padding_mask: Optional[Array] = None,
               key_padding_mask: Optional[Array] = None,
               enable_dropout: Optional[bool] = False) -> Array:
    """Applies multi-head dot product hierarchical attention on input data.

    Args:
      inputs_q: Query, <float>[batch..., length, q_features].
      inputs_kv: Key/Value, <float>[batch..., length, kv_features].
      query_padding_mask: Query padding mask, <int>[batch..., length] or
        <int>[batch..., length, 1]. Zero entries mean the corresponding Query
        tokens are padding token.
      key_padding_mask: Key/Value padding mask, <int>[batch..., length] or
        <int>[batch..., length, 1]. Zero entries mean the corresponding
        Key/Value tokens are padding token.
      enable_dropout: Indicates if the attention weights are masked randomly
        with dropout.

    Returns:
      If output_projection is True, then output of shape
      `<float>[batch..., length, out_features]`, where out_features is set to
      features if not provided. If output_projection is False, then output of
      shape `<float>[batch..., length, num_heads, head_dim]`.
    """
    self._validate_call_parameters(inputs_q, inputs_kv, query_padding_mask,
                                   key_padding_mask)
    is_self_attention = inputs_q is inputs_kv

    inputs_q = self.partitioner.annotate_layer_activation(inputs_q)
    inputs_kv = self.partitioner.annotate_layer_activation(inputs_kv)

    # Applies padding_mask.
    if query_padding_mask is not None:
      if query_padding_mask.ndim == inputs_q.ndim - 1:
        query_padding_mask = query_padding_mask[..., None]
      inputs_q *= query_padding_mask
    if key_padding_mask is not None:
      if key_padding_mask.ndim == inputs_kv.ndim - 1:
        key_padding_mask = key_padding_mask[..., None]
      inputs_kv *= key_padding_mask

    # Performs Multihead projections.
    query, key, value = self._multihead_projection(inputs_q, inputs_kv)
    if self.sharding_over_head_dimension:
      query = self.partitioner.annotate_multihead_qkv(query)
      key = self.partitioner.annotate_multihead_qkv(key)
      value = self.partitioner.annotate_multihead_qkv(value)

    # Computes hierarchical attention and applies it to Value.
    dropout_rng = None
    if enable_dropout and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')
    updated_value = self._hierarchical_attention_fn(
        query,
        key,
        value,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
        dropout_rng=dropout_rng,
        is_self_attention=is_self_attention)
    updated_value = self.partitioner.annotate_layer_activation(updated_value)

    if self.output_projection:
      # The updated_value no longer has multihead shape due to interpolation.
      # So it is a simple 2D projection. This means reshape_kernel=False.
      kwargs = dict(
          features=self.out_features or inputs_q.shape[-1],
          axis=-1,
          kernel_init=self.kernel_init,
          bias_init=self.bias_init,
          use_bias=self.use_bias,
          dtype=self.dtype,
          precision=self.precision,
          name='out',
      )
      if self.enable_param_axes:
        kwargs['reshape_kernel'] = False
        kwargs['kernel_axis_names'] = [AXES.KV, AXES.EMBED]
        return dense.DenseGeneral(**kwargs)(updated_value)
      else:
        return nn.DenseGeneral(**kwargs)(updated_value)
    else:
      return updated_value

  def _validate_call_parameters(self,
                                inputs_q: Array,
                                inputs_kv: Array,
                                query_padding_mask: Optional[Array] = None,
                                key_padding_mask: Optional[Array] = None):
    """Validates the parameters to the call method.

    Args:
      inputs_q: Query, <float>[batch..., length, q_features].
      inputs_kv: Key/Value, <float>[batch..., length, kv_features].
      query_padding_mask: Query padding mask.
      key_padding_mask: Key/Value padding mask.

    Raises:
      ValueError: This is triggered if any of the parameters have a wrong shape.
    """

    def _validate_padding_mask_shape(padding_mask: Array, inputs: Array,
                                     mask_name: str):
      expected_shape = inputs.shape[:
                                    -1] if padding_mask.ndim == inputs.ndim - 1 else inputs.shape[:-1] + (
                                        1,)
      if padding_mask.shape != expected_shape:
        raise ValueError(f'{mask_name} must have shape {expected_shape}; '
                         f' instead got shape {padding_mask.shape}')

    if query_padding_mask is not None:
      _validate_padding_mask_shape(query_padding_mask, inputs_q,
                                   'query_padding_mask')
    if key_padding_mask is not None:
      _validate_padding_mask_shape(key_padding_mask, inputs_kv,
                                   'key_padding_mask')

    if inputs_kv is None:
      raise ValueError('inputs_kv is not given.')
    if inputs_q.ndim != inputs_kv.ndim:
      raise ValueError(f'Mismatched inputs rank: expected '
                       f'inputs_q.ndim ({inputs_q.ndim}) == '
                       f'inputs_kv.ndim ({inputs_kv.ndim})')
    if inputs_q.ndim < 3:
      raise ValueError(f'Expected rank of inputs >= 3, was {inputs_q.ndim}')
    if inputs_q.shape[:-1] != inputs_kv.shape[:-1]:
      raise ValueError(f'Mismatched inputs_kv and inputs_q shape: expected '
                       f'inputs_q.shape[:-1] ({inputs_q.shape[:-1]}) == '
                       f'inputs_kv.shape[:-1] ({inputs_kv.shape[:-1]})')

    qkv_features = self.qkv_features or inputs_q.shape[-1]
    if qkv_features % self.num_heads != 0:
      raise ValueError(
          f'The features dimension {qkv_features} is not divisible by number '
          f'of heads {self.num_heads}.'
      )

  @abc.abstractmethod
  def _setup_hierarchy(
      self,
      features: Union[int, Tuple[int, int]],
      for_self_attention: bool,
  ) -> th.TokenHierarchy:
    """Sets up token hierarchy.

    Args:
      features: Features dimension in inputs.
      for_self_attention: Indicating if this for the self attention.

    Returns:
       Instance of TokenHierarchy.
    """

  @abc.abstractmethod
  def _setup_position_bias(self,
                           hierarchy) -> h_rpb.HierarchicalRelativePositionBias:
    """Sets up hierarchical position bias.

    Args:
      hierarchy: Token hierarchy.

    Returns:
       Instance of HierarchicalRelativePositionBias.
    """

  def _multihead_projection(self, inputs_q, inputs_kv):
    """Project inputs_q/kv to multi-headed query, key and value.

    Args:
      inputs_q: Query, <float>[batch..., length, q_features].
      inputs_kv: Key/Value, <float>[batch..., length, kv_features].

    Returns:
      query: Array with shape <float>[batch..., length, num_head, head_dim]`.
      key: Array with shape <float>[batch..., length, num_head, head_dim]`.
      value: Array with shape <float>[batch..., length, num_head, head_dim]`.
    """
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    head_dim = qkv_features // self.num_heads
    if self.multihead_projection:
      kwargs = dict(
          axis=-1,
          features=(self.num_heads, head_dim),
          bias_init=self.bias_init,
          use_bias=self.use_bias,
          dtype=self.dtype,
          precision=self.precision,
      )
      if self.enable_param_axes:
        dense_module = dense.DenseGeneral
        kwargs['kernel_axis_names'] = [AXES.EMBED, AXES.HEADS, AXES.KV]
        kwargs['reshape_kernel'] = not self.split_head_kernel
      else:
        dense_module = nn.DenseGeneral

      make_dense = functools.partial(dense_module, **kwargs)
      key = make_dense(
          kernel_init=self.kernel_init, name='key_multihead_projection')(
              inputs_kv)
      value = make_dense(
          kernel_init=self.kernel_init, name='value_multihead_projection')(
              inputs_kv)
      depth_scaling = jnp.sqrt(head_dim).astype(self.dtype)
      if self.rescale_logits:
        # The rescaling is done explicitly. This takes more memory but is
        # numerically more stable.
        inputs_q /= depth_scaling
        query_kernel_init = self.kernel_init
      else:
        # This folds logit rescaling into initializer.
        query_kernel_init = (
            lambda *args: self.kernel_init(*args) / depth_scaling)
      query = make_dense(
          kernel_init=query_kernel_init, name='query_multihead_projection')(
              inputs_q)
    else:
      # This is only for unit tests. It avoids the randomness in the projection.
      projected_shape = inputs_q.shape[:-1] + tuple((self.num_heads, head_dim))
      if self.rescale_logits:
        # The rescaling is done explicitly.
        depth_scaling = jnp.sqrt(head_dim).astype(self.dtype)
        inputs_q /= depth_scaling
      query = inputs_q.reshape(projected_shape)
      key = inputs_kv.reshape(projected_shape)
      value = key

    return query, key, value

  def _hierarchical_attention_fn(
      self,  # pytype: disable=annotation-type-mismatch  # jax-ndarray
      query: Array,
      key: Array,
      value: Array,
      query_padding_mask: Optional[Array] = None,
      key_padding_mask: Optional[Array] = None,
      dropout_rng: PRNGKey = None,
      is_self_attention: bool = True,
  ) -> Array:
    r"""Applies hierarchical attention given query, key, and value.

    Args:
      query: Multihead Query with shape <float>[batch..., length,
        num_heads, head_dim].
      key: Multihead Key with shape <float>[batch..., length,
        num_heads, head_dim].
      value: Multihead Value with shape <float>[batch..., length,
        num_heads, head_dim].
      query_padding_mask: Original query padding mask.
      key_padding_mask:  Original key padding mask.
      dropout_rng: The key for generating random dropout.
      is_self_attention: Indicates if this is self-attention.

    Returns:
      Updated Value with shape <float>[batch..., length, features]`.
    """
    head_dim = query.shape[-1]
    if query.ndim == 5:
      seq_length_x = query.shape[-3]
      seq_length_y = query.shape[-4]
      hierarchy = self._setup_hierarchy(
          features=(seq_length_x, seq_length_y),
          for_self_attention=is_self_attention,
      )
    else:
      seq_length = query.shape[-3]
      hierarchy = self._setup_hierarchy(
          seq_length, for_self_attention=is_self_attention
      )
    coarse_query = hierarchy.hierarchical_coarsen(
        query,
        input_array_name=th.InputArrayName.QUERY,
        padding_mask=query_padding_mask)
    packed_coarse_q = coarse_query.packed_coarse_qkv
    if self.sharding_over_head_dimension:
      packed_coarse_q = self.partitioner.annotate_coarse_qkv(packed_coarse_q)
    coarse_key = hierarchy.hierarchical_coarsen(
        key,
        input_array_name=th.InputArrayName.KEY,
        padding_mask=key_padding_mask)
    aggregated_key_padding_mask = coarse_key.packed_aggregated_key_padding_mask
    packed_coarse_k = coarse_key.packed_coarse_qkv
    if self.sharding_over_head_dimension:
      packed_coarse_k = self.partitioner.annotate_coarse_qkv(packed_coarse_k)

    similarity = self._compute_hierarchical_similarity(
        packed_coarse_q, packed_coarse_k, hierarchy
    )
    if self.sharding_over_head_dimension:
      similarity = self.partitioner.annotate_similarity(similarity)
    attention = self._compute_hierarchical_attention(similarity, hierarchy)
    if self.sharding_over_head_dimension:
      attention = self.partitioner.annotate_attention(attention)
    softmax_partition = self._compute_softmax_partition(
        attention, hierarchy, head_dim, query_padding_mask,
        aggregated_key_padding_mask)
    softmax_partition = self.partitioner.annotate_softmax_partition(
        softmax_partition
    )

    # Note:
    # Attention dropout should follow the computation of softmax_partition,
    # according to the implementation in flax.nn.attention.
    if dropout_rng is not None:
      attention = self._attention_dropout(attention, hierarchy, dropout_rng)  # pytype: disable=wrong-arg-types  # jax-ndarray
    if self.sharding_over_head_dimension:
      attention = self.partitioner.annotate_attention(attention)

    coarse_value = hierarchy.hierarchical_coarsen(
        value, input_array_name=th.InputArrayName.VALUE)
    coarse_value = coarse_value.packed_coarse_qkv
    if self.sharding_over_head_dimension:
      coarse_value = self.partitioner.annotate_coarse_qkv(coarse_value)
    updated_value = self._multiply_attention_value(attention, coarse_value,
                                                   hierarchy)
    updated_value = self.partitioner.annotate_layer_activation(updated_value)
    updated_value /= softmax_partition

    if query_padding_mask is not None:
      # Attention matrix has a few rows with all zeros. These rows correspond to
      # the zeros in query_padding_mask. As a consequence, the entries
      # corresponding to these rows in updated_value should be zeros.
      updated_value *= query_padding_mask

    updated_value = self.partitioner.annotate_layer_activation(updated_value)
    return updated_value

  def _compute_hierarchical_similarity(
      self,
      query: Dict[th.TokenBlockName, Array],
      key: Dict[th.TokenBlockName, Array],
      hierarchy: th.TokenHierarchy,
  ) -> Dict[th.TokenBlockName, Array]:
    """Computes hierarchical similarity matrix.

    Args:
      query: Packed coarse Query, value array shape is <float>[batch,
        packed_dim, num_clusters, num_heads, head_dim].
      key: Packed coarse Key, value array shape is <float>[batch, packed_dim,
        num_clusters, num_heads, head_dim].
      hierarchy: Token hierarchy.

    Returns:
      Similarity arrays for all token block interaction.
    """

    def _matmult(query: Array, key: Array) -> Array:
      # einsum_str = 'bpqKhd, bpQkhd->bpqkh'
      if self.use_mxu:
        return jnp.einsum('bpqhd, bpkhd->bpqkh', query, key)
      else:
        return jnp.sum(
            query[..., None, :, :] * key[..., None, :, :, :], axis=-1)

    if self.use_rpb:
      batch_size = query[th.TokenBlockName.ANCHOR].shape[0]
      zero_block_mask = hierarchy.gen_packed_zero_block_mask(
          batch_size=batch_size, use_growth_factor=False, trailing_ndim=3)
      position_bias_fn = self._setup_position_bias(hierarchy)

    similarity = {}
    for block_name in hierarchy.block_names:
      similarity[block_name] = _matmult(query[block_name], key[block_name])
      if self.sharding_over_head_dimension:
        similarity[block_name] = self.partitioner.annotate_similarity(
            similarity[block_name]
        )
      if self.use_rpb:
        block_coord = hierarchy.block_coord[block_name]
        position_bias = position_bias_fn(block_coord)
        # Explicitly duplicates along batch. This is useful for model partition.
        batch = query[block_name].shape[0]
        position_bias = jnp.repeat(position_bias, batch, axis=0)
        if block_name == th.TokenBlockName.ANCHOR:
          similarity[block_name] += position_bias
        else:
          similarity[block_name] += (
              position_bias * zero_block_mask[block_name])

    # The normalization below is critical to avoid NaN error. I suspect that
    # float32 is not enough to support the potentially very large value in
    # exp(similarity). Hence we need to subtract a reasonable constant
    # to reduce the value of exp(similarity). This will not change the
    # attention weights.
    similarity_offset = self._find_similarity_offset(similarity, hierarchy)
    for block_name in hierarchy.block_names:
      similarity[block_name] -= lax.stop_gradient(similarity_offset)
    return similarity

  def _find_similarity_offset(
      self,
      similarity: Dict[th.TokenBlockName, Array],
      hierarchy: th.TokenHierarchy,
  ) -> Array:
    """Finds the offset to normalize the similarity array.

    Args:
      similarity: Similarity arrays, value array has shape <float>[batch,
        packed_dim, num_clusters, num_clusters, num_heads].
      hierarchy: Token hierarchy.

    Returns:
      Similarity offset with shape<float>[batch, 1, 1, 1, num_heads].
    """
    if self.max_similarity_mode == MaxSimilarityMode.SAMPLE_ANCHOR:
      max_axes = tuple((1,))
      similarity_offset = jnp.max(
          similarity[th.TokenBlockName.ANCHOR][:, :, 0, 0, :],
          axis=max_axes,
          keepdims=True)
      if self.max_similarity_factor > 1.:
        similarity_offset *= self.max_similarity_factor
      similarity_offset = similarity_offset[:, :, None, None, :]
    elif self.max_similarity_mode == MaxSimilarityMode.SCAN_ANCHOR:
      # The jnp.max reduces [batch, num_block, num_clusters, num_clusters,
      # num_heads] to [batch, 1, 1, 1, num_heads]. We need to find the maximum
      # for each head of the attention for each example in a batch separately.
      max_axes = tuple((1, 2, 3))
      similarity_offset = jnp.max(
          similarity[th.TokenBlockName.ANCHOR], axis=max_axes, keepdims=True)
      if self.max_similarity_factor > 1.:
        similarity_offset *= self.max_similarity_factor
    else:
      max_axes = tuple((1, 2, 3))
      max_similarity_list = []
      for block_name in hierarchy.block_names:
        max_similarity_list.append(
            jnp.max(similarity[block_name], axis=max_axes, keepdims=True))
      # The jnp.stack() adds axis=0 which is then removed by jnp.max().
      max_similarity_all = jnp.stack(max_similarity_list, axis=0)
      similarity_offset = jnp.max(max_similarity_all, axis=0, keepdims=False)

    if self.sharding_over_head_dimension:
      similarity_offset = self.partitioner.annotate_similarity(
          similarity_offset
      )
    return similarity_offset

  def _compute_hierarchical_attention(
      self, similarity: Dict[th.TokenBlockName, Array],
      hierarchy: th.TokenHierarchy) -> Dict[th.TokenBlockName, Array]:
    """Computes the scaled dot-product attention hierarchically.

    Args:
      similarity: Similarity arrays, value array has shape <float>[batch,
        packed_dim, num_clusters, num_clusters, num_heads].
      hierarchy: Token hierarchy.

    Returns:
      Attention arrays  for all token block interaction. It value array has the
      same shape as that of similarity.
    """
    attention = {}
    assert self.softmax_temperature > 1e-10, 'Softmax temperature too small.'
    for block_name in hierarchy.block_names:
      attention[block_name] = jnp.exp(
          similarity[block_name] / self.softmax_temperature
      )
    if self.sharding_over_head_dimension:
      attention = self.partitioner.annotate_attention(attention)

    # This is to correct the overlapping between attention blocks for the
    # adjacent levels
    if hierarchy.num_level > 1:
      logging.info('Applying correction_mask')
      correction_mask = self._gen_correction_mask(hierarchy)
      for block_name in hierarchy.neighbor_block_names:
        attention[block_name] *= correction_mask[block_name]

    # This is for the auto-regressive decoding. We only need to explicitly
    # mask the attention[th.TokenBlockName.ANCHOR] because we can
    # simply skip the use of non-causal attention blocks in the
    # hierarchical attention-value multiplication. So no need to explicitly
    # mask non-causal attention blocks.
    if self.causal_mask:
      logging.info('Applying causal_mask.')
      attention[th.TokenBlockName.ANCHOR] *= self._gen_causal_mask(hierarchy)

    return attention

  @abc.abstractmethod
  def _gen_correction_mask(
      self, hierarchy: th.TokenHierarchy) -> Dict[th.TokenBlockName, Array]:
    """Generates correction mask.

    Args:
      hierarchy: Token hierarchy.

    Returns:
      The correction mask.
    """

  def _gen_causal_mask(self, hierarchy: th.TokenHierarchy) -> Array:
    """Generates causal mask.

    Args:
      hierarchy: Token hierarchy.

    Returns:
      Causal mask with shape <float>[num_block_cluster, num_block_cluster, 1].
    """
    nc = hierarchy.num_block_cluster
    causal_mask = jnp.tril(jnp.ones((nc, nc), dtype=self.dtype), k=0)
    # Needs to add the last num_heads axis. The first 2 dimensions in
    # attention[TokenBlockName.ANCHOR] are handled by broadcast.
    return causal_mask[:, :, None]

  def _compute_softmax_partition(
      self,
      attention: Dict[th.TokenBlockName, Array],
      hierarchy: th.TokenHierarchy,
      head_dim: int,
      query_padding_mask: Optional[Array] = None,
      aggregated_key_padding_mask: Optional[Array] = None) -> Array:
    """Computes softmax partition.

    Args:
      attention: Attention arrays for all blocks, value array has shape
        <float>[batch, pack_dim, num_clusters, num_clusters, num_heads].
      hierarchy: Token hierarchy.
      head_dim: head dimension.
      query_padding_mask: Original query padding mask.
      aggregated_key_padding_mask: Packed aggregated key padding mask. Its value
        array has shape [batch, packed_dim, num_clusters, 1].

    Returns:
      softmax_partition: Partition for the softmax calculation. Array shape
        is [batch, length, 1].
    """
    if aggregated_key_padding_mask is not None:
      # Expands from [batch, packed_dim, num_clusters, 1] to
      # [batch, packed_dim, num_clusters, num_heads] in preparation
      # to compute attention*all_ones.
      all_ones = {}
      for block_name in hierarchy.block_names:
        all_ones[block_name] = jnp.repeat(
            aggregated_key_padding_mask[block_name], self.num_heads, axis=3)
      softmax_partition = self._multiply_attention_value(
          attention, all_ones, hierarchy)
    else:
      if self.use_row_sum:
        softmax_partition = self._row_sum(attention, hierarchy)
      else:
        (batch_size, packed_dim, num_cluster, _, num_heads) = (
            attention[th.TokenBlockName.ANCHOR].shape)
        all_ones = hierarchy.gen_packed_zero_block_mask(
            batch_size=batch_size, use_growth_factor=True, trailing_ndim=2)
        # Expands from [batch_size, packed_dim, 1, 1] to
        # [batch_size, packed_dim, num_clusters, num_heads] in preparation
        # to compute attention*all_ones.
        for block_name in hierarchy.neighbor_block_names:
          repeated_all_ones = jnp.repeat(
              all_ones[block_name], num_cluster, axis=2)
          repeated_all_ones = jnp.repeat(repeated_all_ones, num_heads, axis=3)
          all_ones[block_name] = repeated_all_ones
        # Special treatment for anchor since it is not created by the function
        # gen_packed_zero_block_mask().
        all_ones[th.TokenBlockName.ANCHOR] = jnp.ones(
            (batch_size, packed_dim, num_cluster, num_heads), dtype=self.dtype)
        softmax_partition = self._multiply_attention_value(
            attention, all_ones, hierarchy)

    # Sets entries corresponding to padding tokens to 1.
    if query_padding_mask is not None:
      softmax_partition = softmax_partition * query_padding_mask + (
          1. - query_padding_mask)

    # Filters out potentially very small entries which can lead to NaN.
    very_small_entry = 1e-6
    softmax_partition = lax.select(
        softmax_partition > very_small_entry,
        softmax_partition.astype(self.dtype),
        jnp.ones(softmax_partition.shape, dtype=self.dtype))
    return self._duplicate_heads(softmax_partition, head_dim)

  def _row_sum(self, attention: Dict[th.TokenBlockName, Array],
               hierarchy: th.TokenHierarchy) -> Array:
    """Computes softmax partition by summing attention matrix rows.

    If there is no padding_mask, simple row summation correctly
    computes softmax_partition. We need packed_zero_block_mask to
    account for the scaling factor 2^k in coarsening at level-k.

    Args:
      attention: Attention arrays for all blocks, value array has shape
        <float>[batch, pack_dim, num_clusters, num_clusters, num_heads].
      hierarchy: Token hierarchy.

    Returns:
      softmax_partition: Partition for the softmax calculation. Array shape
        is [batch, length, 1].
    """
    batch_size = attention[th.TokenBlockName.ANCHOR].shape[0]
    zero_block_mask = hierarchy.gen_packed_zero_block_mask(
        batch_size=batch_size, use_growth_factor=True, trailing_ndim=2)
    first_block = True
    for block_name in hierarchy.block_names:
      # Summing over cluster column indexes in each block.
      block_result = jnp.sum(attention[block_name], axis=-2)
      if block_name == th.TokenBlockName.ANCHOR:
        fine_partition = block_result
      else:
        block_result *= zero_block_mask[block_name]
        if first_block:
          coarse_partition = block_result
          first_block = False
        else:
          coarse_partition += block_result

    # Merges coarse_partition and fine_partition at level=0 since they have
    # the same shape.
    fine_partition += coarse_partition[:, :hierarchy.num_fine_block]
    softmax_partition = hierarchy.recover_input_shape(fine_partition, level=0)
    softmax_partition += hierarchy.interpolate_cumulative_sum(
        coarse_partition[:, hierarchy.num_fine_block:])

    return softmax_partition

  @abc.abstractmethod
  def _duplicate_heads(self, softmax_partition: Array, head_dim: int) -> Array:
    """Duplicates entries in softmax_partition by head_dim times.

    Args:
      softmax_partition: Partition for the softmax calculation. Array shape is
        [batch..., length, num_heads]. This array does not have head_dim axis
        which is to be added here.
      head_dim: The head dimension size.

    Returns:
      New softmax_partition with added duplicated entries.
    """

  def _multiply_attention_value(self, attention: Dict[th.TokenBlockName, Array],
                                value: Dict[th.TokenBlockName, Array],
                                hierarchy: th.TokenHierarchy) -> Array:
    """Compute y=attention*value using hierarchical attention.

    Args:
      attention: The attention weights for all token blocks. Its dict value
        shape is <float>[batch, packed_dim, num_clusters, num_clusters,
        num_heads].
      value: Packed coarse Value for all blocks. The dict value array shape is
        <float>[batch, packed_dim, num_clusters, num_heads, head_dim].
      hierarchy: Token hierarchy.

    Returns:
      Multiplication result of y = attention * value with shape
        <float>[batch..., length, features]
    """

    def _matmul(attention: Array, value: Array) -> Array:
      if value.ndim == 5:
        if self.use_mxu:
          result = jnp.einsum('bpqkh, bpkhd->bpqhd', attention, value)
        else:
          # einsum_str = 'bpqkhD, bpQkhd->bpqhd'
          result = jnp.sum(
              attention[..., None] * value[..., None, :, :, :], axis=3)
      else:
        if self.use_mxu:
          result = jnp.einsum('bpqkh, bpkh->bpqh', attention, value)
        else:
          # einsum_str = 'bpqkh, bpQkh->bpqh'
          result = jnp.sum(attention * value[..., None, :, :], axis=3)
      return result

    first_block = True
    for block_name in hierarchy.block_names:
      block_result = _matmul(attention[block_name], value[block_name])
      if block_name == th.TokenBlockName.ANCHOR:
        fine_y = block_result
      else:
        if first_block:
          coarse_y = block_result
          first_block = False
        else:
          coarse_y += block_result

    # Merge coarse_y and fine_y at level=0 since they have the same shape.
    fine_y += coarse_y[:, :hierarchy.num_fine_block]
    result_y = hierarchy.recover_input_shape(fine_y, level=0)
    result_y += hierarchy.interpolate_cumulative_sum(
        coarse_y[:, hierarchy.num_fine_block:])
    return result_y

  def _attention_dropout(self, attention: Array, hierarchy: th.TokenHierarchy,
                         dropout_rng: PRNGKey) -> Array:
    """Apply dropout to the hierarchical attention weights.

    Args:
      attention: Attention arrays for all blocks, value array has shape
        <float>[batch, pack_dim, num_clusters, num_clusters, num_heads].
      hierarchy: Token hierarchy.
      dropout_rng: The key for generating random dropout.

    Returns:
      New attention arrays for all blocks with randomly zeroed out entries.
    """

    def dropout_multiplier(attention_block, dropout_rng):
      keep_prob = 1.0 - self.dropout_rate
      if self.broadcast_dropout:
        (_, num_block, _, num_clusters, _) = attention_block.shape
        # The dropout is broadcast across batch and num_heads.
        dropout_shape = (1, num_block, num_clusters, num_clusters, 1)
        keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
      else:
        keep = random.bernoulli(dropout_rng, keep_prob, attention_block.shape)
      # This roughly preserves the raw sum of each attention row.
      multiplier = (
          keep.astype(self.dtype) / jnp.asarray(keep_prob, dtype=self.dtype))
      return multiplier

    for block_name in hierarchy.block_names:
      attention[block_name] *= dropout_multiplier(attention[block_name],
                                                  dropout_rng)
    return attention


class OneDimHierarchicalAttention(HierarchicalAttention):
  """One-dimensional hierarchical attention class for sequences.

  See arxiv.org/abs/2107.11906 for algorithm details.
  """

  def _setup_hierarchy(
      self, features: Union[int, Tuple[int, int]], for_self_attention: bool
  ) -> th.OneDimTokenHierarchy:
    return th.OneDimTokenHierarchy(
        features,
        num_cluster=self.num_clusters,
        for_self_attention=for_self_attention,
        causal_mask=self.causal_mask,
        conv_kernel_size=self.conv_kernel_size,
        interpolation_kernel_type=self.interpolation_kernel_type,
        dtype=self.dtype)

  def _setup_position_bias(
      self, hierarchy: th.OneDimTokenHierarchy
  ) -> h_rpb.OneDimHierarchicalRelativePositionBias:
    """Sets up hierarchical position bias.

    Args:
      hierarchy: OneDimTokenHierarchy.

    Returns:
       Instance of OneDimHierarchicalRelativePositionBias.
    """
    num_heads = self.num_heads if self.use_multihead_rpb else 1
    return h_rpb.OneDimHierarchicalRelativePositionBias(
        num_cluster=hierarchy.num_cluster,
        num_head=num_heads,
        enable_param_axes=self.enable_param_axes,
        name='1d_relative_position_bias',
    )

  def _gen_correction_mask(
      self, hierarchy: th.TokenHierarchy) -> Dict[th.TokenBlockName, Array]:
    """Generate correction mask.

    Args:
      hierarchy: Token hierarchy.

    Returns:
      The correction mask. Its dict value array has the shape
         <float>[packed_dim, num_clusters, num_clusters, 1]
    """
    nc = hierarchy.num_cluster
    half_nc = nc // 2
    num_fine_block = hierarchy.num_fine_block
    num_coarse_block = hierarchy.num_coarse_block
    all_ones_block = jnp.ones((num_fine_block, nc, nc, 1), dtype=self.dtype)
    right_block = np.ones((num_coarse_block, nc, nc, 1), dtype=self.dtype)
    right_block[:, half_nc:, :half_nc] = 0
    left_block = np.ones((num_coarse_block, nc, nc, 1), dtype=self.dtype)
    left_block[:, :half_nc, half_nc:] = 0
    correction_mask = {
        th.TokenBlockName.RIGHT: jnp.concatenate(
            (all_ones_block, right_block), axis=0
        ),
        th.TokenBlockName.LEFT: jnp.concatenate(
            (all_ones_block, left_block), axis=0
        ),
    }
    return correction_mask

  def _duplicate_heads(self, softmax_partition: Array, head_dim: int) -> Array:
    """Duplicates entries in softmax_partition by head_dim times.

    Args:
      softmax_partition: Partition for the softmax calculation. Array shape is
        [batch, length, num_heads]. This array does not have head_dim axis which
        is to be added here.
      head_dim: The head dimension size.

    Returns:
      New softmax_partition with added duplicated entries.
    """
    softmax_partition = jnp.repeat(
        softmax_partition[..., None], head_dim, axis=3)
    new_shape = softmax_partition.shape[:2] + (self.num_heads * head_dim,)
    return softmax_partition.reshape(new_shape)


class OneDimDecoderSelfAttention(OneDimHierarchicalAttention):
  """Decoder self-attention for one-dimension sequences."""

  causal_mask: bool = True

  def __call__(
      self,  # pytype: disable=signature-mismatch  # overriding-default-value-checks
      inputs: Array,
      padding_mask: Array,
      enable_dropout: Optional[bool] = False) -> Array:
    return super().__call__(
        inputs,
        inputs,
        query_padding_mask=padding_mask,
        key_padding_mask=padding_mask,
        enable_dropout=enable_dropout)


class OneDimEncoderSelfAttention(OneDimHierarchicalAttention):
  """Encoder self-attention for one-dimension sequences."""

  causal_mask: bool = False

  def __call__(
      self,  # pytype: disable=signature-mismatch  # overriding-default-value-checks
      inputs: Array,
      padding_mask: Optional[Array] = None,
      enable_dropout: Optional[bool] = False,
  ) -> Array:
    return super().__call__(
        inputs,
        inputs,
        query_padding_mask=padding_mask,
        key_padding_mask=padding_mask,
        enable_dropout=enable_dropout)


OneDimCrossAttention = OneDimHierarchicalAttention
