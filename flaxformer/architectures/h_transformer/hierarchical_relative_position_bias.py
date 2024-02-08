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

r"""Utility classes to compute the hierarchical relative position bias (h-RPB).

The notion of the relative position bias is the same as that used in T5.
The h-RPB is added to the hierarchical similarity matrix which is then used
to compute the hierarchical attention (h-attention) matrix.
But the h-RPB is tightly coupled with the token hierarchy established by
the h-attention algorithm. At each level in the hierarchy, each token block
only attends to its immediate neighboring token blocks left and right.
So the total number of relative positions at each level is independent of
the sequence length. Since the same h-RPB is shared by all levels in the
hierarchy, the memory footprint of the h-RPB is independent of the
sequence length. Experiments have shown that it adds very little overhead
to the overall model training memory usage and runtime.
"""

from flax import linen as nn
from flax.linen import partitioning
import jax.numpy as jnp
import numpy as np
from flaxformer.architectures.h_transformer.partitioning import AxisName
from flaxformer.types import Initializer


class HierarchicalRelativePositionBias(nn.Module):
  """Base class for the Hierarchical Relative Position Bias (h-RPB).

  Attributes:
    position_bias_init: Positional bias initializer.
    num_cluster: Number of clusters in h_attention.
    num_head: Number of heads with different h-RPB. Setting num_head=1 means all
      heads share the same h-RPB.
  """

  position_bias_init: Initializer = nn.initializers.normal(stddev=0.1)  # pytype: disable=annotation-type-mismatch  # jax-types
  num_cluster: int = 2
  num_head: int = 1
  enable_param_axes: bool = True

  def _create_1d_relative_position_bias(
      self, param_name: str = '1d_relative_position_bias') -> jnp.ndarray:
    """Creates a trainable one-dimensional relative position bias array.

    Args:
      param_name: Name for the trainable one-dimensional relative position bias
        array.

    Returns:
      Trainable one-dimensional relative position bias array.
        <float>[num_cluster, 3*num_cluster, num_head]

    Notes:
      A few small static arrays are calculated or allocated with numpy
      and get folded into program constants. This is more efficient and
      the memory foot print is as small as O(num_cluster).
    """
    # Key tokens sit at positions with coordinates in the range [0, 3*nc].
    key_length = 3 * self.num_cluster
    key_positions = np.arange(key_length)
    # Query tokens sit at positions with coordinates in the range [nc, 2*nc].
    query_length = self.num_cluster
    query_positions = np.arange(self.num_cluster, 2 * self.num_cluster)
    # Compute the relative positions between each query and key pair.
    relative_positions = key_positions.reshape(
        (1, key_length)) - query_positions.reshape((query_length, 1))
    # These indices are used by a bias lookup. So we shift the indices
    # such that the smallest index is zero.
    relative_positions -= np.min(relative_positions)
    total_positions = query_length + key_length - 1
    if self.enable_param_axes:
      bias_params = partitioning.param_with_axes(
          param_name,
          self.position_bias_init,
          (total_positions, self.num_head),
          jnp.float32,
          axes=(AxisName.RELPOS_BUCKETS, AxisName.HEADS),
      )
    else:
      bias_params = self.param(
          param_name, self.position_bias_init, (total_positions, self.num_head)
      )
    relative_pos_bias = jnp.take(bias_params, relative_positions, axis=0)
    return relative_pos_bias


class OneDimHierarchicalRelativePositionBias(HierarchicalRelativePositionBias):
  """Computes 1D Hierarchical Relative Position Bias."""

  def setup(self):
    # The resulting array has shape (nc, 3*nc, num_head).
    relative_position_bias = self._create_1d_relative_position_bias(
        '1d_relative_position_bias')
    # Split it into 3 blocks. They map to 3 key blocks, where the anchor
    # query block sitting at the center position-1.
    split_blocks = jnp.split(
        relative_position_bias, [self.num_cluster, 2 * self.num_cluster],
        axis=1)

    position_bias_blocks = {}
    for block_index, split_block in enumerate(split_blocks):
      # Add (batch=1, num_block=1) to axis=(0,1) to match the shape
      # (batch, num_block, num_cluster, num_cluster, num_head) for
      # the hierarchical similarity array in the h-attention algorithm.
      position_bias_blocks[str(block_index)] = jnp.expand_dims(
          split_block, axis=(0, 1))
    self.position_bias_blocks = position_bias_blocks

  def __call__(self, block_coord: int) -> jnp.ndarray:
    """Retrieve 1D HierarchicalRelativePositionBias.

    Args:
      block_coord: This is the position of the key block. Specifically, -1 for
        the left position, 1 for the right position, 0 for the center position
        where the query block also sits.

    Returns:
      The relative position bias block for the specified block_coord.
         <float>[batch=1, num_block=1, num_cluster, num_cluster, num_head].
    """
    block_index = str(block_coord + 1)
    return self.position_bias_blocks[block_index]
