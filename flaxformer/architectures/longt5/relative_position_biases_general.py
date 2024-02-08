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

"""Class for relative position biases generalized to long inputs."""
from typing import Any, Callable

from flax import linen as nn
from flax.linen import partitioning
from jax import lax
import jax.numpy as jnp
import numpy as np

from flaxformer.types import Array


class RelativePositionBiasesGeneral(nn.Module):
  """Adds T5-style relative positional embeddings to the attention logits.

  This generalizes the original `RelativePositionBiases` implementation to
  accept an `rp_bucket` input of any shape, avoiding construction of
  an O(N^2) tensor for long inputs of length N.  The original full attention
  `rp_bucket` can be retrieved with `full_att_rp_bucket()`.

  T5 uses a form of relative attention which biases the attention matrix, so
  each head effectively attends to things at different scales, irrespective of
  the contents of keys and queries.

  In the future, this class may be unified with classes which take into account
  key and query contents, like the original relative position embeddings of Shaw
  et al. and new proposals. However, this will rely on XLA to recover efficiency
  for this class (especially when, as in the original T5, the same bias matrix
  is shared for all layers).

  Attributes:
    num_buckets: Number of buckets to bucket distances between key and query
      positions into.
    max_distance: Maximum distance before everything is lumped into the last
      distance bucket.
    num_heads: Number of heads in the attention layer. Each head will get a
      different relative position weighting.
    dtype: Type of arrays through this module.
    embedding_init: initializer for relative embedding table.
  """
  num_buckets: int
  max_distance: int
  num_heads: int
  dtype: Any
  embedding_init: Callable[..., Array] = nn.linear.default_embed_init

  @staticmethod
  def relative_position_bucket(relative_position,
                               bidirectional=True,
                               num_buckets=32,
                               max_distance=128):
    """Translates relative position to a bucket number for relative attention.

    The relative position is defined as memory_position - query_position, i.e.
    the distance in tokens from the attending position to the attended-to
    position.  If bidirectional=False, then positive relative positions are
    invalid.
    We use smaller buckets for small absolute relative_position and larger
    buckets for larger absolute relative_positions.  All relative
    positions >=max_distance  map to the same bucket.  All relative
    positions <=-max_distance map to the same bucket.  This should allow for
    more graceful generalization to longer sequences than the model has been
    trained on.

    Args:
      relative_position: an int32 array
      bidirectional: a boolean - whether the attention is bidirectional
      num_buckets: an integer
      max_distance: an integer

    Returns:
      a Tensor with the same shape as relative_position, containing int32
        values in the range [0, num_buckets)
    """
    ret = 0
    n = -relative_position
    if bidirectional:
      num_buckets //= 2
      ret += (n < 0).astype(jnp.int32) * num_buckets
      n = jnp.abs(n)
    else:
      n = jnp.maximum(n, 0)
    # now n is in the range [0, inf)
    max_exact = num_buckets // 2
    is_small = (n < max_exact)
    val_if_large = max_exact + (
        jnp.log(n.astype(jnp.float32) / max_exact + jnp.finfo(jnp.float32).eps)
        / jnp.log(max_distance / max_exact) *
        (num_buckets - max_exact)).astype(jnp.int32)
    val_if_large = jnp.minimum(val_if_large, num_buckets - 1)
    ret += jnp.where(is_small, n, val_if_large)
    return ret

  def full_att_rp_bucket(self, qlen, klen, bidirectional=True):
    """Gets relative position buckets for full attention.

    Args:
      qlen: attention query length.
      klen: attention key length.
      bidirectional: a boolean - whether the attention is bidirectional

    Returns:
      int32 (qlen, klen) shaped array containing values in the range
      [0, num_buckets).
    """
    # TODO: should we be computing this w. numpy as a program
    # constant?
    context_position = np.arange(qlen, dtype=jnp.int32)[:, None]
    memory_position = np.arange(klen, dtype=jnp.int32)[None, :]
    relative_position = memory_position - context_position  # shape (qlen, klen)
    rp_bucket = self.relative_position_bucket(
        relative_position,
        bidirectional=bidirectional,
        num_buckets=self.num_buckets,
        max_distance=self.max_distance)
    return rp_bucket

  @nn.compact
  def __call__(self, rp_bucket: Array):
    """Produces relative position embedding attention biases.

    Args:
      rp_bucket: int32 containing values in the range [0, num_buckets). In the
        full attention case, this should have shape (qlen, klen).

    Returns:
      output: Attention bias array with shape `(1, num_heads) + rp_bucket.shape`
    """
    relative_attention_bias = partitioning.param_with_axes(
        'rel_embedding',
        self.embedding_init, (self.num_heads, self.num_buckets),
        jnp.float32,
        axes=('heads', 'relpos_buckets'))
    relative_attention_bias = jnp.asarray(relative_attention_bias, self.dtype)
    # Instead of using a slow gather, we create a leading-dimension one-hot
    # array from rp_bucket and use it to perform the gather-equivalent via a
    # contraction.  For example, if `rp_bucket` has shape (qlen, klen), the
    # contraction looks like:
    # (num_head, num_buckets) x (num_buckets one-hot, qlen, klen).
    # This is equivalent to relative_attention_bias[:, rp_bucket]
    bcast_iota_shape = [self.num_buckets] + [1] * rp_bucket.ndim
    bcast_iota = lax.broadcasted_iota(jnp.int32, bcast_iota_shape, 0)
    rp_bucket_one_hot = jnp.array(
        rp_bucket[jnp.newaxis, ...] == bcast_iota, dtype=self.dtype)
    # --> shape (num_heads, rp_bucket.shape)
    values = lax.dot_general(
        relative_attention_bias,
        rp_bucket_one_hot,
        (
            ((1,), (0,)),  # lhs, rhs contracting dims
            ((), ())))  # no batched dims
    # Add a singleton batch dimension.
    # --> shape (1, num_heads, rp_bucket.shape)
    return values[jnp.newaxis, ...]
