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

r"""Hierarchical attention classes."""
import abc
import functools
from typing import Dict, Optional

from absl import logging

from flax import linen as nn
from flax.linen import partitioning as flax_partitioning
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as np
from flaxformer import activation_partitioning
from flaxformer.architectures.h_transformer import hierarchical_relative_position_bias as h_rpb
from flaxformer.architectures.h_transformer import token_hierarchy as th
from flaxformer.components import dense
from flaxformer.types import Array
from flaxformer.types import Initializer
from flaxformer.types import PRNGKey


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
    output_projection: Project the output of `attention_fn` to `out_features`.
      If False, returns the output of `attention_fn` without a projection.
    split_head_kernel: whether to store QKVO projection kernels with a split
      head dimension. Default to False so the kernels are stored in 2D shape for
      the compatibility with Adafactor optimizer.
    rescale_logits: bool. Whether to  explicitly rescale `query` logits by
      1/sqrt(depth_kq). Default is to do this implicitly by folding the
      rescaling into query_kernel_init.
    sharding_over_head_dimension: Whether to shard over the head dimension.
      Setting this to False when the number of heads is not divisible your
      activation num_partitions.
    use_rpb: Whether the hierarchical relative position bias is used.
      Default to True because this setting delivers better results.
    use_multihead_rpb: Whether the hierarchical relative position bias is
      different among multihead. If False, the same relative position bias is
      shared among all heads. Default to True so the bias array is stored in 2D
      shape for the compatibility with Adafactor optimizer.
    conv_kernel_size: Convolution kernel size used for coarsening and
      interpolation. This is not used during coarsening if the attribute
      coarsening_kernel_type=ConvKernelType.LINEAR since the kernel size is
      fixed at 3. This is also the case for the interpolation.
    coarsening_kernel_type: Type of coarsening convolution kernels.
    interpolation_kernel_type: Type of interpolation convolution kernels.
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
  kernel_init: Initializer = nn.linear.default_kernel_init
  bias_init: Initializer = nn.initializers.zeros
  use_bias: bool = True
  output_projection: bool = True
  split_head_kernel: bool = False
  rescale_logits: bool = False
  sharding_over_head_dimension: bool = True
  use_rpb: bool = True
  use_multihead_rpb: bool = True
  conv_kernel_size: int = 2
  coarsening_kernel_type: th.ConvKernelType = th.ConvKernelType.CONST
  interpolation_kernel_type: th.ConvKernelType = th.ConvKernelType.CONST

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
    if query_padding_mask is not None and query_padding_mask.ndim == inputs_q.ndim - 1:
      query_padding_mask = query_padding_mask[..., None]
    if key_padding_mask is not None and key_padding_mask.ndim == inputs_q.ndim - 1:
      key_padding_mask = key_padding_mask[..., None]

    for_self_attention = inputs_q is inputs_kv
    hierarchy = self._setup_hierarchy(
        inputs_q.shape[1], for_self_attention=for_self_attention)
    coarse_results = hierarchy.hierarchical_coarsen(
        inputs_q,
        inputs_kv,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask)
    coarse_qkv = coarse_results.packed_coarse_qkv
    aggregated_key_padding_mask = coarse_results.packed_aggregated_key_padding_mask

    qkv_features = self.qkv_features or inputs_q.shape[-1]
    head_dim = qkv_features // self.num_heads
    if self.rescale_logits:
      # The logit rescaling will be done explicitly to query later.
      query_kernel_init = self.kernel_init
    else:
      # This folds logit rescaling into initializer.
      depth_scaling = jnp.sqrt(head_dim).astype(self.dtype)
      query_kernel_init = lambda *args: self.kernel_init(*args) / depth_scaling

    make_dense = functools.partial(
        dense.DenseGeneral,
        axis=-1,
        features=(self.num_heads, head_dim),
        kernel_axis_names=['embed', 'heads', 'kv'],
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        precision=self.precision,
        reshape_kernel=not self.split_head_kernel,
    )
    dense_q = make_dense(
        kernel_init=query_kernel_init, name='query_multihead_projection')
    dense_k = make_dense(
        kernel_init=self.kernel_init, name='key_multihead_projection')
    dense_v = make_dense(
        kernel_init=self.kernel_init, name='value_multihead_projection')

    query: Dict[th.TokenBlockName, Array] = {}
    key: Dict[th.TokenBlockName, Array] = {}
    value: Dict[th.TokenBlockName, Array] = {}
    for block_name in hierarchy.block_names:
      query[block_name] = dense_q(
          coarse_qkv[th.InputArrayName.QUERY][block_name])
      key[block_name] = dense_k(coarse_qkv[th.InputArrayName.KEY][block_name])
      value[block_name] = dense_v(
          coarse_qkv[th.InputArrayName.VALUE][block_name])
      if self.sharding_over_head_dimension:
        query[block_name] = self._shard_over_head_dimension(query[block_name])
        key[block_name] = self._shard_over_head_dimension(key[block_name])
        value[block_name] = self._shard_over_head_dimension(value[block_name])

    dropout_rng = None
    if enable_dropout and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')
    updated_value = self._hierarchical_attention_fn(
        query,
        key,
        value,
        hierarchy,
        query_padding_mask=query_padding_mask,
        aggregated_key_padding_mask=aggregated_key_padding_mask,
        dropout_rng=dropout_rng)

    if self.output_projection:
      # The updated_value no longer has multihead shape due to interpolation.
      # So it is a simple 2D projection. This means reshape_kernel=False.
      return dense.DenseGeneral(
          features=self.out_features or inputs_q.shape[-1],
          axis=-1,
          kernel_init=self.kernel_init,
          bias_init=self.bias_init,
          use_bias=self.use_bias,
          dtype=self.dtype,
          precision=self.precision,
          reshape_kernel=False,
          kernel_axis_names=['kv', 'embed'],
          name='out')(  # pytype: disable=wrong-arg-types
              updated_value)
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
          f'The features dimension {qkv_features} is not divisible by number of '
          f'heads {self.num_heads}.')

  @abc.abstractmethod
  def _setup_hierarchy(self, features: int,
                       for_self_attention: bool) -> th.TokenHierarchy:
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

  def _shard_over_head_dimension(self, qkv: Array) -> Array:
    """Shards multihead projections.

    Args:
      qkv: Projected Query/Key/Value  with shape <float>[batch, packed_dim,
        num_clusters, num_heads, head_dim].

    Returns:
       Sharded projection of Query/Key/Value.
    """
    if flax_partitioning.get_axis_rules():
      qkv = flax_partitioning.with_sharding_constraint(
          qkv, ('batch', 'block', 'cluster', 'heads', 'kv'))
    else:
      # Only partitions along data axis because the partitioner does not handle
      # array with ndim=5 for both data and model partition. This is less
      # desirable. So callers should try to make the axis rules valid so the
      # if-branch above is taken. The default t5x setting guarantees this.
      qkv = activation_partitioning.with_sharding(qkv, 1)
    return qkv

  def _hierarchical_attention_fn(
      self,
      query: Dict[th.TokenBlockName, Array],
      key: Dict[th.TokenBlockName, Array],
      value: Dict[th.TokenBlockName, Array],
      hierarchy: th.TokenHierarchy,
      query_padding_mask: Optional[Dict[th.TokenBlockName, Array]] = None,
      aggregated_key_padding_mask: Optional[Dict[th.TokenBlockName,
                                                 Array]] = None,
      dropout_rng: PRNGKey = None) -> Array:
    r"""Applies hierarchical attention given query, key, and value.

    Args:
      query: Packed coarse Query, value array shape is <float>[batch,
        packed_dim, num_clusters, num_heads, head_dim].
      key: Packed coarse Key, value array shape is <float>[batch, packed_dim,
        num_clusters, num_heads, head_dim].
      value: Packed coarse Value, dict value array shape is <float>[batch,
        packed_dim, num_clusters, num_heads, head_dim].
      hierarchy: Token hierarchy.
      query_padding_mask: Original query padding mask.
      aggregated_key_padding_mask: Packed aggregated key padding mask.
      dropout_rng: The key for generating random dropout.

    Returns:
      Updated Value with shape <float>[batch..., length, features]`.
    """
    similarity = self._compute_hierarchical_similarity(query, key, hierarchy)
    attention = self._compute_hierarchical_attention(similarity, hierarchy)
    head_dim = query[th.TokenBlockName.ANCHOR].shape[-1]
    softmax_partition = self._compute_softmax_partition(
        attention, hierarchy, head_dim, query_padding_mask,
        aggregated_key_padding_mask)

    # Note:
    # Attention dropout should follow the computation of softmax_partition,
    # according to the implementation in flax.nn.attention.
    if dropout_rng is not None:
      attention = self._attention_dropout(attention, hierarchy, dropout_rng)

    updated_value = self._multiply_attention_value(attention, value, hierarchy)
    updated_value /= softmax_partition

    if query_padding_mask is not None:
      # Attention matrix has a few rows with all zeros. These rows correspond to
      # the zeros in query_padding_mask. As a consequence, the entries
      # corresponding to these rows in updated_value should be zeros.
      updated_value *= query_padding_mask

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
      return jnp.sum(query[..., None, :, :] * key[..., None, :, :, :], axis=-1)

    if self.use_rpb:
      zero_block_mask = hierarchy.gen_packed_zero_block_mask(
          use_growth_factor=False)
      position_bias_fn = self._setup_position_bias(hierarchy)

    if self.rescale_logits:
      head_dim = query[th.TokenBlockName.ANCHOR].shape[-1]
      depth_scaling = jnp.sqrt(head_dim).astype(self.dtype)
    similarity = {}
    for block_name in hierarchy.block_names:
      if self.rescale_logits:
        query[block_name] /= depth_scaling
      similarity[block_name] = _matmult(query[block_name], key[block_name])
      if self.use_rpb:
        block_coord = hierarchy.block_coord[block_name]
        position_bias = position_bias_fn(block_coord)
        if block_name == th.TokenBlockName.ANCHOR:
          similarity[block_name] += position_bias
        else:
          similarity[block_name] += (
              position_bias * zero_block_mask[block_name][..., None])

    # The normalization below is critical to avoid NaN error. I suspect that
    # float32 is not enough to support the potentially very large value in
    # exp(similarity). Hence we need to subtract a reasonable constant
    # to reduce the value of exp(similarity). This will not change the
    # attention weights.
    max_similarity = self._find_max_similarity(similarity, hierarchy)
    for block_name in hierarchy.block_names:
      similarity[block_name] -= max_similarity
    return similarity

  def _find_max_similarity(self, similarity: Dict[th.TokenBlockName, Array],
                           hierarchy: th.TokenHierarchy) -> Array:
    """Finds the maximum entries in the similarity array.

    Args:
      similarity: Similarity arrays, value array has shape <float>[batch,
        packed_dim, num_clusters, num_clusters, num_heads].
      hierarchy: Token hierarchy.

    Returns:
      Maximum similarity with shape<float>[batch, 1, 1, 1, num_heads].
    """
    # The jnp.max reduces [batch, num_block, num_clusters, num_clusters,
    # num_heads] to [batch, 1, 1, 1, num_heads]. We need to find the maximum
    # for each head of the attention for each example in a batch separately.
    max_axes = tuple((1, 2, 3))
    max_similarity_list = []
    for block_name in hierarchy.block_names:
      max_similarity_list.append(
          jnp.max(similarity[block_name], axis=max_axes, keepdims=True))
    # The jnp.stack() adds axis=0 which is then removed by jnp.max().
    max_similarity_all = jnp.stack(max_similarity_list, axis=0)
    return jnp.max(max_similarity_all, axis=0, keepdims=False)

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
    for block_name in hierarchy.block_names:
      attention[block_name] = jnp.exp(similarity[block_name])

    # This is to correct the overlapping between attention blocks for the
    # adjacent levels
    if hierarchy.num_level > 1:
      logging.info('Applying correction_mask')
      correction_mask = self._gen_correction_mask(hierarchy)
      for block_name in hierarchy.block_names:
        if block_name == th.TokenBlockName.ANCHOR:
          continue
        attention[block_name] *= correction_mask[block_name]

    # This is for the auto-regressive decoding. We only need to explcitly
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

  @abc.abstractmethod
  def _gen_causal_mask(self, hierarchy: th.TokenHierarchy) -> Array:
    """Generates causal mask.

    Args:
      hierarchy: Token hierarchy.

    Returns:
      The causal mask.
    """

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
      softmax_partition = self._column_sum(attention, hierarchy)

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

  def _column_sum(self, attention: Dict[th.TokenBlockName, Array],
                  hierarchy: th.TokenHierarchy) -> Array:
    """Computes softmax partition by summing attention matrix columns.

    If there is no padding_mask, simple column summation correctly
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
    zero_block_mask = hierarchy.gen_packed_zero_block_mask(
        use_growth_factor=True)
    first_block = True
    for block_name in hierarchy.block_names:
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
        # einsum_str = 'bpqkhD, bpQkhd->bpqhd'
        result = jnp.sum(
            attention[..., None] * value[..., None, :, :, :], axis=3)
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
      keep_prob = jax.lax.tie_in(attention_block, 1.0 - self.dropout_rate)
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

  def _setup_hierarchy(self, features: int,
                       for_self_attention: bool) -> th.OneDimTokenHierarchy:
    return th.OneDimTokenHierarchy(
        features,
        num_cluster=self.num_clusters,
        for_self_attention=for_self_attention,
        causal_mask=self.causal_mask,
        conv_kernel_size=self.conv_kernel_size,
        coarsening_kernel_type=self.coarsening_kernel_type,
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
        name='1d_relative_position_bias')

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
    right_block = np.ones((1, nc, nc, 1), dtype=self.dtype)
    right_block[:, half_nc:, :half_nc] = 0
    right_blocks = jnp.repeat(right_block, num_coarse_block, axis=0)
    left_block = np.ones((1, nc, nc, 1), dtype=self.dtype)
    left_block[:, :half_nc, half_nc:] = 0
    left_blocks = jnp.repeat(left_block, num_coarse_block, axis=0)
    return {
        th.TokenBlockName.RIGHT:
            jnp.concatenate((all_ones_block, right_blocks), axis=0),
        th.TokenBlockName.LEFT:
            jnp.concatenate((all_ones_block, left_blocks), axis=0),
    }

  def _gen_causal_mask(self, hierarchy: th.TokenHierarchy) -> Array:
    """Generate causal mask.

    Args:
      hierarchy: Token hierarchy.

    Returns:
      The causal mask with shape <float>[num_clusters, num_clusters, num_heads]
    """
    causal_mask = jnp.tril(
        jnp.ones((hierarchy.num_cluster, hierarchy.num_cluster),
                 dtype=self.dtype),
        k=0)
    # Needs to add the last num_heads axis. The first 2 dimensions in
    # attention[TokenBlockName.ANCHOR] are handled by broadcast.
    return causal_mask[:, :, None]

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

  def __call__(self,
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

  def __call__(self,
               inputs: Array,
               padding_mask: Array,
               enable_dropout: Optional[bool] = False) -> Array:
    return super().__call__(
        inputs,
        inputs,
        query_padding_mask=padding_mask,
        key_padding_mask=padding_mask,
        enable_dropout=enable_dropout)


OneDimCrossAttention = OneDimHierarchicalAttention
