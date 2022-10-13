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

"""Class for relative position biases.

T5 uses a form of relative attention which biases the attention matrix, so each
head effectively attends to things at different scales, irrespective of the
contents of keys and queries.

In the future, this class may be unified with classes which take into account
key and query contents, like the original relative position embeddings of Shaw
et al. and new proposals. However, this will rely on XLA to recover efficiency
for this class (especially when, as in the original T5, the same bias matrix
is shared for all layers).
"""
import abc
from typing import Any, Callable, Sequence

from flax import linen as nn
from flax.linen import partitioning
from jax import lax
import jax.numpy as jnp
import numpy as np

from flaxformer.types import Array


class RelativeAttentionAPI(metaclass=abc.ABCMeta):
  """Interface for relative attention APIs."""

  @abc.abstractmethod
  def __call__(self, qlen: int, klen: int, bidirectional: bool, decode: bool):
    """Produces relative position embedding attention biases.

    This method should return position biases of shape `(1, num_heads, q_len,
    k_len)`.

    Args:
      qlen: Attention query length.
      klen: Attention key length.
      bidirectional: Whether to allow positive memory-query relative position
        embeddings.
      decode: Whether to cache relative position bias during autoregressive
        decoding.
    """
    raise NotImplementedError()


class RelativePositionBiases(nn.Module, RelativeAttentionAPI):
  """Adds T5-style relative positional embeddings to the attention logits.

  Attributes:
    num_buckets: Number of buckets to bucket distances between key and query
      positions into.
    max_distance: Maximum distance before everything is lumped into the last
      distance bucket.
    num_heads: Number of heads in the attention layer. Each head will get a
      different relative position weighting.
    dtype: Type of arrays through this module.
    embedding_init: initializer for relative embedding table.
    head_axis_name: Axis to partition the relpos bias heads on. Setting this
      field trades training performance for unbounded parallelism in mixed
      models.
  """
  num_buckets: int
  max_distance: int
  num_heads: int
  dtype: Any
  embedding_init: Callable[..., Array] = nn.linear.default_embed_init
  head_axis_name: str = 'heads'

  @staticmethod
  def _relative_position_bucket(relative_position,
                                bidirectional=True,
                                num_buckets=32,
                                max_distance=128):
    """Translate relative position to a bucket number for relative attention.

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
      ret += (n < 0).astype(np.int32) * num_buckets
      n = np.abs(n)
    else:
      n = np.maximum(n, 0)
    # now n is in the range [0, inf)
    max_exact = num_buckets // 2
    is_small = (n < max_exact)
    val_if_large = max_exact + (
        np.log(n.astype(np.float32) / max_exact + np.finfo(np.float32).eps) /
        np.log(max_distance / max_exact) *
        (num_buckets - max_exact)).astype(np.int32)
    val_if_large = np.minimum(val_if_large, num_buckets - 1)
    ret += np.where(is_small, n, val_if_large)
    return ret

  @nn.compact
  def __call__(self, qlen, klen, bidirectional=True, decode=False):
    """Produce relative position embedding attention biases.

    Args:
      qlen: attention query length.
      klen: attention key length.
      bidirectional: whether to allow positive memory-query relative position
        embeddings.
      decode: whether to cache relative position bias during autoregressive
        decoding.

    Returns:
      output: `(1, num_heads, q_len, k_len)` attention bias
    """
    # bidirectional embeddings don't make sense when decoding (and break cache).
    if decode and bidirectional:
      raise ValueError(
          'bidirectional RelativePositionBiases are not supported when decode=True.'
      )

    # We only cache the bias if the model was already initialized, i.e. if this
    # module is called with model.apply and decode = True. We raise an error if
    # called with model.init and decode = True, since this can cache incorrect
    # positional embeddings produced by random parameters.
    is_initialized = self.has_variable('params', 'rel_embedding')
    if decode and not is_initialized:
      raise ValueError(
          'decode-mode cannot be enabled during init. use model.apply to '
          'initialize the decoding cache.')

    # Return pre-computed relative position bias in cache during decode steps.
    if decode and self.has_variable('cache', 'cached_bias'):
      cached_bias = self.get_variable('cache', 'cached_bias')
      expected_bias_shape = (1, self.num_heads, qlen, klen)
      if cached_bias.shape != expected_bias_shape:
        raise ValueError(f'The cached relative position attention bias was '
                         f'expected to have shape {expected_bias_shape} but '
                         f'instead has the shape {cached_bias.shape}.')
      return cached_bias

    # TODO: should we be computing this w. numpy as a program
    # constant?
    context_position = np.arange(qlen, dtype=jnp.int32)[:, None]
    memory_position = np.arange(klen, dtype=jnp.int32)[None, :]
    relative_position = memory_position - context_position  # shape (qlen, klen)
    rp_bucket = self._relative_position_bucket(
        relative_position,
        bidirectional=bidirectional,
        num_buckets=self.num_buckets,
        max_distance=self.max_distance)
    relative_attention_bias = partitioning.param_with_axes(
        'rel_embedding',
        self.embedding_init, (self.num_heads, self.num_buckets),
        jnp.float32,
        axes=(self.head_axis_name, 'relpos_buckets'))
    relative_attention_bias = jnp.asarray(relative_attention_bias, self.dtype)
    # Instead of using a slow gather, we create a leading-dimension one-hot
    # array from rp_bucket and use it to perform the gather-equivalent via a
    # contraction, i.e.:
    # (num_head, num_buckets) x (num_buckets one-hot, qlen, klen).
    # This is equivalent to relative_attention_bias[:, rp_bucket]
    bcast_iota = lax.broadcasted_iota(jnp.int32, (self.num_buckets, 1, 1), 0)
    rp_bucket_one_hot = jnp.array(
        rp_bucket[jnp.newaxis, ...] == bcast_iota, dtype=self.dtype)
    # --> shape (qlen, klen, num_heads)
    values = lax.dot_general(
        relative_attention_bias,
        rp_bucket_one_hot,
        (
            ((1,), (0,)),  # rhs, lhs contracting dims
            ((), ())))  # no batched dims
    # Add a singleton batch dimension.
    # --> shape (1, num_heads, qlen, klen)
    out = values[jnp.newaxis, ...]

    # Store computed relative position bias in cache after first calculation.
    if decode:
      _ = self.variable('cache', 'cached_bias', lambda: out)

    return out


