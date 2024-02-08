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

r"""Token Hierarchy classes for the h-attention algorithm."""

import abc
import collections
import dataclasses
import enum
from typing import Dict, List, Optional, OrderedDict
from absl import logging
import gin
from jax import lax
import jax.numpy as jnp
import numpy as np
from flaxformer.types import Array
from flaxformer.types import DType


@enum.unique
class InputArrayName(enum.Enum):
  """Input array names."""

  QUERY = enum.auto()
  KEY = enum.auto()
  VALUE = enum.auto()


@enum.unique
class TokenBlockName(enum.Enum):
  """Token block names.

  At each level in the token hierarchy, the tokens are partitioned
  into equal-sized blocks. The attention among blocks corresponds
  to the attention matrix structure.
  """

  ANCHOR = enum.auto()
  # For one-dimension token sequences
  LEFT = enum.auto()
  RIGHT = enum.auto()


@dataclasses.dataclass()
class HierarchicalCoarsenResults:
  packed_coarse_qkv: Dict[TokenBlockName, Array]
  packed_aggregated_key_padding_mask: Optional[Dict[TokenBlockName, Array]]


@dataclasses.dataclass()
class CoarsenPaddingMaskResults:
  """The results of coarsening the padding mask.

  Attributes:
    aggregated_padding_mask: The aggregated padding mask where each entry at
      level=k is the sum of a sub set of mask tokens at level=k-1.
    denominator: Each entry at level=k records the number of effective mask
      tokens at level=k-1 that contributes to the corresponding mask token in
      aggregated_padding_mask. For instance, if a sub set of mask tokens at
      level=k-1 is [1, 1, 0, 0], then the corresponding  entry in denominator at
      level=k should be 2.
  """

  aggregated_padding_mask: Dict[int, Array]
  denominator: Optional[Dict[int, Array]]


@gin.constants_from_enum
class TokenCoarseningMethod(str, enum.Enum):
  """Names of the coarsening method."""

  SAMPLE = 'sample'
  SUM = 'sum'
  CONST_AVERAGE = 'const_average'

  def __format__(self, format_spec: str) -> str:
    return self.value.__format__(format_spec)


@gin.constants_from_enum
class ConvKernelType(str, enum.Enum):
  """Names of the convolution kernel type."""

  CONST = 'const'
  LINEAR = 'linear'

  def __format__(self, format_spec: str) -> str:
    return self.value.__format__(format_spec)


class OneDimTokenCoarsening:
  """Coarsening class for one-dimension sequence token hierarchy."""

  def __init__(
      self,
      method: TokenCoarseningMethod = TokenCoarseningMethod.SUM,
      coarsening_ratio: int = 2,
  ):
    """Initialize coarsening.

    Args:
      method: Coarsening method name.
      coarsening_ratio: The ratio of the token count at two adjacent levels. For
        instance, 2 means token count is reduced by a factor of 2 from level-k
        to level-(k+1) due to coarsening.
    """
    self._coarsening_ratio = coarsening_ratio
    self._method = method

  def __call__(self, inputs: Array) -> Array:
    """Coarsens or downscales sequence length by coarsening_ratio.

    Args:
      inputs: Input sequences, <float>[batch, seq_len, num_head_ head_dim].

    Returns:
      Coarsened sequences, <float>[batch, seq_len//coarsening_ratio,
        num_head_ head_dim].

    Raises:
      ValueError: This is triggered if sequence length does not divide
        coarsening_ratio.
    """
    (batch, seq_len, num_head, head_dim) = inputs.shape
    new_seq_len = seq_len // self._coarsening_ratio
    if new_seq_len * self._coarsening_ratio != seq_len:
      raise ValueError(
          f'The sequence length {seq_len} does not divide coarsening_ratio '
          f'{self._coarsening_ratio}.'
      )

    new_shape = (batch, new_seq_len, self._coarsening_ratio, num_head, head_dim)
    reshaped_inputs = inputs.reshape(new_shape)
    if self._method == TokenCoarseningMethod.SAMPLE:
      # For auto-regressive decoder, only sample the first position of
      # each block. This avoids leakage from future tokens.
      x = reshaped_inputs[:, :, 0, :, :]
    else:
      x = reshaped_inputs.sum(axis=2, keepdims=False)
      if self._method == TokenCoarseningMethod.CONST_AVERAGE:
        x /= self._coarsening_ratio
    return x


class OneDimTokenInterpolation:
  """Interpolation class for one-dimension sequence token hierarchy."""

  def __init__(
      self,
      conv_kernel_size: int = 2,
      conv_kernel_type: ConvKernelType = ConvKernelType.CONST,
      channel_dim: int = 1,
      interpolation_ratio: int = 2,
      dtype: DType = jnp.float32,
      use_edge_correction: bool = True,
  ):
    """Generates a static conv kernel for interpolation.

    Args:
      conv_kernel_size: Convolution kernel size used for interpolation.
      conv_kernel_type: Convolution kernel type for the interpolation.
      channel_dim: Size of Channel dimension.
      interpolation_ratio: The ratio of the token count at two adjacent levels.
        For instance, 2 means sequence length is increased by a factor of 2 from
        level-k to level-(k-1) due to interpolation.
      dtype: The dtype of the computation.
      use_edge_correction: Indicates if a correction is applied to the edge
        output entries.

    Raises:
      ValueError: This is triggered if method is not in the pre-determined set
        or coarsening_ratio is larger than conv_kernel_size.
    """
    if interpolation_ratio > conv_kernel_size:
      raise ValueError(
          f'interpolation_ratio {interpolation_ratio} is larger than '
          f'conv_kernel_size {conv_kernel_size}. This means some tokens '
          'will not be included in the interpolation and the final output will '
          'be wrong.'
      )
    self._interpolation_ratio = interpolation_ratio
    self._use_edge_correction = use_edge_correction
    self._conv_kernel_type = conv_kernel_type
    if conv_kernel_type == ConvKernelType.CONST:
      kernel = np.ones((conv_kernel_size,))
    elif conv_kernel_type == ConvKernelType.LINEAR:
      kernel = np.array([0.5, 1.0, 0.5])
    else:
      raise ValueError(f'Unsupported conv_kernel_type {conv_kernel_type}.')

    kernel = jnp.array(kernel[:, None, None], dtype=dtype)
    self._conv_kernel = (
        jnp.repeat(kernel, channel_dim, axis=2) if channel_dim > 1 else kernel
    )

  def __call__(self, inputs: Array) -> Array:
    """Interpolates or upscales sequence length by 2x.

    Args:
      inputs: Input sequences with shape <float>[batch, seq_len, channel_dim].

    Returns:
      Interpolated embeddings with shape <float>[batch, 2*seq_len, channel_dim].
    """
    dn = lax.conv_dimension_numbers(
        inputs.shape, self._conv_kernel.shape, ('NHC', 'HIO', 'NHC')
    )
    kernel_size = self._conv_kernel.shape[0]
    padding = ((kernel_size - 1, kernel_size - 1),)
    lhs_dilation = (self._interpolation_ratio,)
    y = lax.conv_general_dilated(
        inputs.astype(jnp.float32),
        self._conv_kernel.astype(jnp.float32),
        (1,),
        padding=padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=(1,),
        dimension_numbers=dn,
        feature_group_count=inputs.shape[-1],
    )

    if self._conv_kernel_type == ConvKernelType.LINEAR:
      # Linear conv_kernel can potentially improve accuracy. But it has two
      # issues:
      # 1) It expands the output sequence length by one at both left and right
      # end. This makes the output seq_len larger by one than necessary. So
      # we need to truncate the first token in the output.
      # 2) The original sequence is zero padded before the conv kernel is
      # applied. This makes the edge entries on both ends in the output less
      # accurate than those obtained with the constant conv_kernel. We can
      # compensate this artifact by simply multiplying the last entry by 2.
      # See the unit test for this effect.
      output_sequence_length = inputs.shape[1] * 2
      y = y[:, -output_sequence_length:]
      if self._use_edge_correction:
        correction = np.ones((output_sequence_length,))
        correction[-1] = 2
        correction = jnp.array(correction[None, :, None])
        y *= correction
    return y


class TokenHierarchy(metaclass=abc.ABCMeta):
  """Base class for the Token Hierarchy."""

  def __init__(
      self,
      seq_len: int,
      num_cluster: int = 2,
      conv_kernel_size: int = 2,
      interpolation_kernel_type: ConvKernelType = ConvKernelType.LINEAR,
      for_self_attention: bool = True,
      causal_mask: bool = False,
      token_ratio: int = 2,
      dtype: DType = jnp.float32,
  ):
    """Initializes class attributes.

    Args:
      seq_len: Sequence length.
      num_cluster: Number of clusters at each level in the hierarchy.
      conv_kernel_size: Size of convolution kernels.
      interpolation_kernel_type: Type of interpolation convolution kernels.
      for_self_attention: This indicates if this is for the self attention.
      causal_mask: This specifies whether to apply a causal mask on the
        attention weights. If True, the output at timestep `t` will not depend
        on inputs at timesteps strictly greater than `t`.
      token_ratio: The ratio of the token count at two adjacent levels. For
        instance, 2 means token count is reduced by a factor of 2 from level-k
        to level-(k+1) due to coarsening.
      dtype: The dtype of the computation.

    Raises:
      ValueError: This is triggered if num_cluster is not an even number or
          seq_len and num_cluster are incompatible.
    """
    self._conv_kernel_size = conv_kernel_size
    self._interpolation_kernel_type = interpolation_kernel_type
    self._for_self_attention = for_self_attention
    self._causal_mask = causal_mask
    self._token_ratio = token_ratio
    self._dtype = dtype

    if num_cluster % 2 != 0:
      raise ValueError('num_cluster must be an even number.')
    max_num_cluster = seq_len // 2
    if max_num_cluster <= 1:
      raise ValueError(
          'max_num_cluster must be larger than 1; instead got '
          f'max_num_cluster={max_num_cluster}. This is caused '
          'by a small input length = 2.'
      )
    self._num_cluster = min(num_cluster, max_num_cluster)
    if self._num_cluster != num_cluster:
      logging.info(
          'num_cluster is reset from %d to %d because max_num_cluster = %d',
          num_cluster,
          self._num_cluster,
          max_num_cluster,
      )

    self._num_block_leaf_level = seq_len // self._num_cluster
    if self._num_block_leaf_level * self._num_cluster != seq_len:
      raise ValueError(
          'seq_len must be divisible by num_cluster; instead got '
          f'(seq_len, num_cluster)={seq_len, self._num_cluster}.'
      )
    if self._num_block_leaf_level < 2:
      raise ValueError(
          'num_block=seq_len/num_cluster>=2 is required; '
          f'instead got num_block={self._num_block_leaf_level}'
      )

    self._num_level = int(np.log2(self._num_block_leaf_level))
    if np.exp2(self._num_level) != self._num_block_leaf_level:
      raise ValueError(
          'num_block=seq_len/num_cluster must be power of 2;'
          f' instead got num_block={self._num_block_leaf_level}'
      )

    self._num_block = [self._num_block_leaf_level] * self._num_level
    for level in range(1, self._num_level):
      self._num_block[level] = self._num_block[level - 1] // 2

    self._setup_block_hierarchy()
    logging.info('seq_len = %d', seq_len)
    logging.info('num_cluster = %d', self._num_cluster)
    logging.info('num_level = %d', self._num_level)
    logging.info('num_block = %s', self._num_block)
    logging.info('total_num_block = %d', self._total_num_block)
    logging.info('num_fine_block = %d', self._num_fine_block)
    logging.info('num_coarse_block = %d', self._num_coarse_block)

  def _setup_block_hierarchy(self):
    self._num_fine_block = 0
    self._num_coarse_block = 0
    self._total_num_block = 0
    self._level_end_coarse_block_idx = []
    self._block_coord = OrderedDict()
    self._causal_block_names = []
    self._neighbor_block_names = []

  @abc.abstractmethod
  def hierarchical_coarsen(
      self,
      inputs: Array,
      input_array_name: InputArrayName = InputArrayName.QUERY,
      padding_mask: Optional[Array] = None,
  ) -> HierarchicalCoarsenResults:
    """Hierarchically coarsens inputs level by level.

    Args:
      inputs: Input Query/Key/Value.
      input_array_name: The name of the inputs.
      padding_mask: Query/Key padding mask.

    Returns:
      Packed coarse Query/Key/Value and optional packed coarse padding mask.
    """

  @abc.abstractmethod
  def interpolate_cumulative_sum(self, coarse_y: Array) -> Array:
    """Interpolates and cumulatively sums over all levels.

    This function performs two tasks: 1) Interpolates from coarse grid to fine
    grid level-by-level, starting from the coarsest level. 2) Cumulatively
    sums the interpolated results at each level.

    Args:
      coarse_y: Packed and coarsened embeddings.

    Returns:
      Interpolated and cumulatively summed embeddings over all levels.
    """

  @abc.abstractmethod
  def recover_input_shape(self, packed_coarse_qkv: Array, level: int) -> Array:
    """Recovers from blockwise partitioned shape to the input shape.

    Args:
      packed_coarse_qkv: Packed coarse qkv at the specified level.
      level: The hierarchy level where the packed_coarse_y sits.

    Returns:
      Reshaped coarse_qkv.
    """

  @property
  def block_coord(self) -> OrderedDict[TokenBlockName, int]:
    return self._block_coord

  @property
  def block_names(self) -> List[TokenBlockName]:
    if self._causal_mask:
      return self._causal_block_names
    else:
      return list(self._block_coord.keys())

  @property
  def neighbor_block_names(self) -> List[TokenBlockName]:
    return self._neighbor_block_names

  @property
  def num_cluster(self) -> int:
    return self._num_cluster

  @property
  def num_block_cluster(self) -> int:
    return self._num_cluster

  @property
  def num_level(self) -> int:
    return self._num_level

  @property
  def num_block(self) -> List[int]:
    return self._num_block

  @property
  def num_fine_block(self) -> int:
    return self._num_fine_block

  @property
  def num_coarse_block(self) -> int:
    return self._num_coarse_block

  @property
  def total_num_block(self) -> int:
    return self._total_num_block

  @property
  def level_end_coarse_block_idx(self) -> List[int]:
    return self._level_end_coarse_block_idx

  @property
  def growth_factor(self) -> float:
    return self._token_ratio

  def gen_packed_zero_block_mask(
      self,
      batch_size: int,
      use_growth_factor: bool = False,
      trailing_ndim: int = 2,
  ) -> Dict[TokenBlockName, Array]:
    """Generates blockwise zero mask pattern in packed form.

    Args:
      batch_size: The batch size for training data.
      use_growth_factor: This indicates if the block entries at each level is
        enlarged by a factor exponential to the hierarchy level.
      trailing_ndim: Number of dimensions after the block dim. See notes below.

    Returns:
      Packed zero block mask: Dict with key=block_name, and value array has
        shape <float>[batch, packed_dim, num_cluster=1, num_head=1]
        or [batch, packed_dim, num_cluster=1, num_head=1, head_dim=1]
        or [batch, packed_dim, num_cluster=1, num_cluster=1, head_dim=1].
        The shapes are simply [batch, packed_dim, 1, 1] or
        [batch, packed_dim, 1, 1, 1].

    Note:
      This mask is used to zero out blocks in three types of arrays:
      1) The coarse token blocks with shape [batch, num_block, num_cluster,
      num_head, head_dim];
      2) The relative position bias with the shape [batch, num_block,
      num_cluster, num_cluster, num_head];
      3) The attention matrix row sum as softmax partition with the shape
      [batch, num_block, num_cluster, num_head].
      The mask shape should be compatible with them.
      The final shapes for the three types are different only in the number of
      trailing dimensions, 2 vs. 3. Hence we just need a flag to differentiate
      them.
    """
    growth_factor = self.growth_factor if use_growth_factor else 1
    packed_zero_block_mask = {}
    for block_name in self.block_names:
      if block_name == TokenBlockName.ANCHOR:
        continue
      block_mask_list = []
      scalar = 1.0
      for level in range(self.num_level):
        block_mask = self._gen_zero_block_mask(level, block_name, batch_size)
        if scalar > 1.0:
          block_mask *= scalar
        block_mask_list.append(block_mask)
        scalar *= growth_factor
      packed_block_mask = jnp.concatenate(block_mask_list, axis=1)
      if trailing_ndim == 2:
        packed_block_mask = packed_block_mask[:, :, None, None]
      else:
        packed_block_mask = packed_block_mask[:, :, None, None, None]
      packed_zero_block_mask[block_name] = packed_block_mask

    return packed_zero_block_mask

  @abc.abstractmethod
  def _gen_zero_block_mask(
      self,
      level: int,
      block_name: TokenBlockName,
      batch_size: int,
  ) -> Array:
    """Generates a block mask.

    Args:
      level: The specified level in the hierarchy.
      block_name: The generated mask is for the token/attention block with this
        name.
      batch_size: The batch size for training data.

    Returns:
      Generated zero block mask with shape <float32>[batch_size, num_block]
    """


def _shift_blocks_1d(input_array: Array, block_name: TokenBlockName) -> Array:
  """Shifts array blocks along axis=1.

  Args:
    input_array: Input array with shape <float>[batch, num_block, ...].
    block_name: The name of the token block.

  Returns:
     Shifted array with the same shape as that of input_array.
  """
  pad_shape = list(input_array.shape)
  pad_shape[1] = 1
  zero_pad = jnp.zeros(tuple(pad_shape), dtype=jnp.float32)
  if block_name == TokenBlockName.LEFT:
    # Shift top blocks downward and fill the vacancy with zero blocks.
    # This pushes out the bottom block.
    result = jnp.concatenate((zero_pad, input_array[:, :-1]), axis=1)
  elif block_name == TokenBlockName.RIGHT:
    # Shift bottom blocks upward and fill the vacancy with zero blocks.
    # This pushes out the top block.
    result = jnp.concatenate((input_array[:, 1:], zero_pad), axis=1)
  else:
    result = input_array
  return result


class OneDimTokenHierarchy(TokenHierarchy):
  """Token hierarchy for one-dimensional sequences.

  See arxiv.org/abs/2107.11906 for details on token hierarchy
  for one-dimensional sequences.
  """

  def _setup_block_hierarchy(self):
    """Sets up block hierarchy."""
    # These are for bookkeeping.
    self._num_fine_block = self._num_block[0]
    self._num_coarse_block = sum(self._num_block[1:])
    self._total_num_block = sum(self._num_block)

    # This is used to access h-attention blocks.
    self._block_coord = collections.OrderedDict({
        TokenBlockName.ANCHOR: 0,
        TokenBlockName.LEFT: -1,
        TokenBlockName.RIGHT: 1,
    })
    self._causal_block_names = [TokenBlockName.ANCHOR, TokenBlockName.LEFT]
    self._neighbor_block_names = list(self.block_names)
    self._neighbor_block_names.remove(TokenBlockName.ANCHOR)

    # This is used to unpack multilevel blocks.
    self._level_end_coarse_block_idx = [0] * self._num_level
    for level in range(1, self._num_level):
      self._level_end_coarse_block_idx[level] = (
          self._level_end_coarse_block_idx[level - 1] + self._num_block[level]
      )

  def hierarchical_coarsen(
      self,
      inputs: Array,
      input_array_name: InputArrayName = InputArrayName.QUERY,
      padding_mask: Optional[Array] = None,
  ) -> HierarchicalCoarsenResults:
    """Hierarchically coarsens inputs level by level.

    Args:
      inputs: Query/Key/Value, <float>[batch, seq_len, num_head, head_dim].
      input_array_name: The name of the inputs.
      padding_mask: padding mask, <float>[batch, seq_len, 1].

    Returns:
      packed_coarse_qkv: Packed and coarsened Query/Key/Value.
         key: TokenBlockName.ANCHOR
         value: <float>[batch, num_block[0], num_cluster, num_head, head_dim].
         key: TokenBlockName.LEFT
         value: <float>[batch, packed_dim, num_cluster, num_head, head_dim].
         key: TokenBlockName.RIGHT
         value: <float>[batch, packed_dim, num_cluster, num_head, head_dim].
     Packed aggregated Key padding mask.
       It has the same key-value pair as packed_coarse_qkv

    Raises:
      ValueError: This is triggered when inputs_q, query_padding_mask or
        key_padding_mask has the wrong rank.
    """
    if inputs.ndim != 4:
      raise ValueError(f'inputs rank={inputs.ndim}, it must be 4.')

    if padding_mask is not None:
      if padding_mask.ndim != 3:
        raise ValueError(
            f'padding_mask rank = {padding_mask.ndim}, it must be 3.'
        )
      padding_mask = padding_mask[..., None]

    decoder_only = self._for_self_attention and self._causal_mask
    encoder_only = self._for_self_attention and not self._causal_mask
    cross_attention = not self._for_self_attention

    aggregated_padding_mask = None
    if input_array_name == InputArrayName.QUERY:
      if encoder_only or cross_attention:
        # In both cases, q_denominator is needed for coarsening query.
        coarse_mask_results = self._coarsen_padding_mask(
            padding_mask, need_aggregation=False
        )
        q_denominator = coarse_mask_results.denominator
      else:
        q_denominator = None

      # For auto-regressive decoder, only sample the first position of each
      # coarsening cluster. This avoids the leakage from future positions.
      coarse_qkv = self._coarsen_query_or_key(
          inputs, denominator=q_denominator, use_sample=decoder_only
      )
    elif input_array_name == InputArrayName.KEY:
      coarse_mask_results = self._coarsen_padding_mask(
          padding_mask, need_aggregation=True
      )
      aggregated_padding_mask = coarse_mask_results.aggregated_padding_mask
      coarse_qkv = self._coarsen_query_or_key(
          inputs, denominator=coarse_mask_results.denominator, use_sample=False
      )
    else:
      coarse_qkv = self._coarsen_value(inputs)

    coarse_qkv = self._partition_sequences(coarse_qkv)
    packed_coarse_qkv = self._pack_coarse_qkv(
        coarse_qkv, input_name=input_array_name
    )
    if aggregated_padding_mask is not None:
      aggregated_padding_mask = self._partition_sequences(
          aggregated_padding_mask
      )
      packed_aggregated_padding_mask = self._pack_coarse_qkv(
          aggregated_padding_mask, input_name=InputArrayName.KEY
      )
    else:
      packed_aggregated_padding_mask = None

    return HierarchicalCoarsenResults(
        packed_coarse_qkv=packed_coarse_qkv,
        packed_aggregated_key_padding_mask=packed_aggregated_padding_mask,
    )

  def recover_input_shape(self, packed_coarse_qkv: Array, level: int) -> Array:
    """Recovers from blockwise partitioned shape to the input sequence shape.

    Args:
      packed_coarse_qkv: Packed coarse qkv with shape <float>[batch, num_block,
        num_cluster, features] or <float>[batch, num_block, num_cluster,
        num_head, head_dim].
      level: The hierarchy level where the packed_coarse_qkv sits.

    Returns:
      Reshaped coarse_qkv with shape <float>[batch, seq_len, features], where
      seq_len = num_block * num_cluster,
      features = num_head * head_dim
    """
    if packed_coarse_qkv.ndim == 4:
      (batch, _, num_cluster, features) = packed_coarse_qkv.shape
      channel_dim = features
    else:
      (batch, _, num_cluster, num_head, head_dim) = packed_coarse_qkv.shape
      channel_dim = num_head * head_dim

    num_block = self.num_block[level]
    new_shape = tuple((batch, num_block * num_cluster, channel_dim))
    return packed_coarse_qkv.reshape(new_shape)

  def interpolate_cumulative_sum(self, coarse_y: Array) -> Array:
    """Interpolates and cumulatively sums over all levels.

    Args:
      coarse_y: Packed and coarsened embeddings with shape <float>[batch,
        num_coarse_block, num_cluster, features] or <float>[batch,
        num_coarse_block, num_cluster, num_head, head_dim].

    Returns:
      Interpolated and cumulatively summed embeddings over all levels.
      Its shape is <float>[batch, seq_len, features], where
      seq_len = num_block[0] * num_cluster,
      features = num_head * head_dim
    """
    if coarse_y.ndim == 4:
      channel_dim = coarse_y.shape[-1]
    else:
      # Flatten the last two dims since conv_general() inside interpolation
      # only allows one channel_dim.
      (num_head, head_dim) = coarse_y.shape[-2:]
      channel_dim = num_head * head_dim

    interpolation_fn = OneDimTokenInterpolation(
        conv_kernel_size=self._conv_kernel_size,
        conv_kernel_type=self._interpolation_kernel_type,
        channel_dim=channel_dim,
        interpolation_ratio=self._token_ratio,
    )

    cumulative_sum = 0.0  # Default value in case num_level==1.
    for level in range(self.num_level - 1, 0, -1):
      level_start = self.level_end_coarse_block_idx[level - 1]
      level_end = self.level_end_coarse_block_idx[level]
      current_level_coarse_y = self.recover_input_shape(
          coarse_y[:, level_start:level_end], level
      )
      if level == self.num_level - 1:
        # This starts the cumsum. So only assignment is done.
        cumulative_sum = current_level_coarse_y
      else:
        cumulative_sum += current_level_coarse_y
      cumulative_sum = interpolation_fn(cumulative_sum)
    return cumulative_sum  # pytype: disable=bad-return-type  # jax-ndarray

  def _coarsen_padding_mask(
      self, padding_mask: Optional[Array], need_aggregation: bool = False
  ) -> CoarsenPaddingMaskResults:
    """Coarsens padding mask.

    Args:
      padding_mask: Query or Key/Value padding mask, <float>[batch, seq_len, 1].
      need_aggregation: This indicates if aggregated padding mask is needed.

    Returns:
      aggregated_padding_mask: Aggregated padding mask.
         key: level
         value: <float>[batch, seq_len[level], 1]
      denominator: The denominator to be used for normalization.
         key: level
         value: <float>[batch, seq_len[level], 1]
    """
    if padding_mask is None:
      return CoarsenPaddingMaskResults(  # pytype: disable=wrong-arg-types  # jax-ndarray
          aggregated_padding_mask=padding_mask, denominator=None
      )

    coarsening_fn = OneDimTokenCoarsening(
        method=TokenCoarseningMethod.SUM, coarsening_ratio=self._token_ratio
    )

    coarse_padding_mask = {0: padding_mask}
    if need_aggregation:
      aggregated_padding_mask = {0: coarse_padding_mask[0]}

    denominator = {}
    for level in range(1, self.num_level):
      coarse_padding_mask[level] = coarsening_fn(coarse_padding_mask[level - 1])

      # Sets zero entries to ones to avoid divide-by-zero later.
      # Note: No need for denominator[0] since it will not be used.
      denominator[level] = lax.select(
          coarse_padding_mask[level] > 0,
          coarse_padding_mask[level].astype(self._dtype),
          jnp.ones(coarse_padding_mask[level].shape, dtype=self._dtype),
      )

      if need_aggregation:
        if level == 1:
          aggregated_padding_mask[level] = coarse_padding_mask[level]
        else:
          aggregated_padding_mask[level] = coarsening_fn(
              aggregated_padding_mask[level - 1]
          )

      # Sets entries to 1/0 so that coarse_padding_mask at each level is still
      # a binary mask. This is important to get the correct denominator.
      # Note: No need to treat level-0 padding since it has not been aggregated.
      coarse_padding_mask[level] = lax.select(
          coarse_padding_mask[level] > 0,
          jnp.ones(coarse_padding_mask[level].shape, dtype=self._dtype),
          jnp.zeros(coarse_padding_mask[level].shape, dtype=self._dtype),
      )

    if not need_aggregation:
      aggregated_padding_mask = None

    return CoarsenPaddingMaskResults(
        aggregated_padding_mask=aggregated_padding_mask, denominator=denominator
    )

  def _coarsen_query_or_key(
      self,
      inputs_qk: Array,
      denominator: Optional[Array],
      use_sample: bool = False,
  ) -> Dict[int, Array]:
    """Coarsens Query or Key.

    Args:
      inputs_qk: Query or Key, <float32>[batch, seq_len, num_head, head_dim].
      denominator: <float32>[batch, seq_len, 1, 1].
      use_sample: bool, indicating if sampling is used to coarsen.

    Returns:
      coarse_qk: Coarsened Query or Key.
         key: level
         value: <float>[batch, seq_len[level], num_head, head_dim]
    """
    if use_sample:
      method = TokenCoarseningMethod.SAMPLE
    else:
      if denominator is None:
        method = TokenCoarseningMethod.CONST_AVERAGE
      else:
        method = TokenCoarseningMethod.SUM

    coarsening_fn = OneDimTokenCoarsening(
        method=method, coarsening_ratio=self._token_ratio
    )

    coarse_qk = {0: inputs_qk}
    for level in range(1, self.num_level):
      coarse_qk[level] = coarsening_fn(coarse_qk[level - 1])
      if not use_sample and denominator is not None:
        coarse_qk[level] /= denominator[level]

    return coarse_qk

  def _coarsen_value(self, inputs: Array) -> Dict[int, Array]:
    """Coarsens Value.

    Args:
      inputs: Value, <float32>[batch, seq_len, num_head, head_dim].

    Returns:
      coarse_v: Coarsened Value.
         key: level
         value: <float>[batch, seq_len[level], num_head, head_dim]
    """
    coarsening_fn = OneDimTokenCoarsening(
        method=TokenCoarseningMethod.SUM, coarsening_ratio=self._token_ratio
    )
    coarse_v = {0: inputs}
    for level in range(1, self.num_level):
      coarse_v[level] = coarsening_fn(coarse_v[level - 1])
    return coarse_v

  def _pack_coarse_qkv(
      self, coarse_qkv: Dict[int, Array], input_name: InputArrayName
  ) -> Dict[TokenBlockName, Array]:
    """Packs coarse Query/Key/Value.

    Args:
      coarse_qkv: Coarse Query/Key/Value at multiple levels in a dict, where key
        is a specific level, value is an array with shape <float>[batch,
        seq_len[level], feature_dim]
      input_name: Indicates if coarse_qkv is Query, Key or Value.

    Returns:
      packed_coarse_qkv: Packed coarse Query/Key/Value in a dict
          key: TokenBlockName.ANCHOR,
          value: Original Query/Key/Value at level=0. This is for computing
            or being multiplied by attention[TokenBlockName.ANCHOR]. Its shape
            is <float>[batch, num_block[0], num_cluster, feature_dim].
          key: TokenBlockName.LEFT,
          value: Coarsened Query/Key/Value at all levels. This is for
            computing or being multiplied by attention[TokenBlockName.LEFT].
            Its shape is <float>[batch, packed_dim, num_cluster, feature_dim].
        If causal_mask==False, there is one more (key, value) pair.
          key: TokenBlockName.RIGHT,
          value: Coarsened Query/Key/Value at all levels. This is for
            computing or being multiplied by attention[TokenBlockName.RIGHT].
            Its shape is <float>[batch, packed_dim, num_cluster, feature_dim].
    """
    to_replace = input_name == InputArrayName.QUERY
    if to_replace:
      batch_size = coarse_qkv[0].shape[0]
      packed_zero_block_mask = self.gen_packed_zero_block_mask(
          batch_size=batch_size, use_growth_factor=False, trailing_ndim=3
      )

    packed_coarse_qkv = {}
    for block_name in self.block_names:
      if block_name == TokenBlockName.ANCHOR:
        packed_coarse_qkv[TokenBlockName.ANCHOR] = coarse_qkv[0]
      else:
        packed_list = []
        for level in range(self.num_level):
          if to_replace:
            packed_list.append(coarse_qkv[level])
          else:
            packed_list.append(_shift_blocks_1d(coarse_qkv[level], block_name))
        packed_coarse_qkv[block_name] = jnp.concatenate(packed_list, axis=1)
        if to_replace:
          packed_coarse_qkv[block_name] *= packed_zero_block_mask[block_name]
    return packed_coarse_qkv

  def _partition_sequences(
      self, coarse_qkv: Dict[int, Array]
  ) -> Dict[int, Array]:
    """Partitions sequences at each level of the hierarchy into blockwise shape.

    Args:
      coarse_qkv: Coarse Query/Key/Value, where key=level, value=array with
        shape <float>[batch, seq_len[level], num_head, head_dim].

    Returns:
      Partitioned Coarse Query/Key/Value.
    """
    (batch, _, num_head, head_dim) = coarse_qkv[0].shape
    reshaped_coarse_qkv = {}
    for level in range(self.num_level):
      new_shape = (
          batch,
          self.num_block[level],
          self.num_cluster,
          num_head,
          head_dim,
      )
      reshaped_coarse_qkv[level] = coarse_qkv[level].reshape(new_shape)
    return reshaped_coarse_qkv

  def _gen_zero_block_mask(
      self,
      level: int,
      block_name: TokenBlockName,
      batch_size: int,
  ) -> Array:
    """Generates a block mask.

    Args:
      level: The specified level in the hierarchy.
      block_name: The generated mask is for the token/attention block with this
        name.
      batch_size: The batch size for training data.

    Returns:
      Generated zero block mask with shape <float32>[batch_size, num_block]
    """
    num_block = self.num_block[level]
    block_coord = self.block_coord[block_name]
    block_mask = np.ones((batch_size, num_block), dtype=self._dtype)
    if block_coord == -1:
      block_mask[:, 0] = 0.0
    elif block_coord == 1:
      block_mask[:, -1] = 0.0
    return jnp.array(block_mask)
