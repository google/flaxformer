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

"""Convolution functions and classes."""

# pylint: disable=attribute-defined-outside-init,g-bare-generic,g-multiple-import

from typing import Any, Optional, Sequence

from flax import linen as nn
from flax.linen import partitioning as flax_partitioning
from flax.training import common_utils
from jax import lax
import jax.numpy as jnp

from flaxformer.types import Array


def constant_init(value):

  def _my_fn(key, shape, dtype=jnp.float32):
    del key
    return jnp.full(shape, value, dtype)

  return _my_fn


def roll_with_zeros(x, shift):
  """Version of jax.numpy.roll that zeros out wraparound values.

  Args:
    x: input tensor
    shift: a list with length equal to the rank of x.

  Returns:
    A tensor with the same shape as x.
  """
  if len(shift) != len(x.shape):
    raise ValueError('shift must have same length as x.shape got %s %s' %
                     (x, shift))
  start_indices = []
  limit_indices = []
  padding = []
  for dimsize, s in zip(x.shape, shift):
    start_indices.append(max(0, -s))
    limit_indices.append(min(dimsize, dimsize - s))
    padding.append((max(0, s), max(0, -s)))
  return jnp.pad(lax.slice(x, start_indices, limit_indices), padding)


def roll_with_zeros_along_axis(x, distance, axis):
  shape = x.shape
  rank = len(shape)
  if axis < 0:
    axis += rank
  shift = [0] * rank
  shift[axis] = distance
  return roll_with_zeros(x, shift)


class Depthwise1dConv(nn.Module):
  """One-dimensional depthwise convolution.

  If autoregressive=True, then position `i` receives information from positions
  in the interval `[i-radius, i]`.

  If autoregressive=False, then position `i` receives information from positions
  in the interval `[i-radius, i+radius]`.

  Attributes:
    radius: Maximum distance to move information.
    autoregressive: Whether to only look left.
    dtype: the dtype of the computation (default: float32).
  """
  axis_names: Sequence[str]
  radius: int = 2
  autoregressive: bool = True
  dtype: Any = jnp.float32
  length_dim: int = 1
  num_feature_dims: int = 1
  use_in_mlp_parallel_fused_layer: bool = False

  @nn.compact
  def __call__(self,
               x: Array,
               decode: bool = False,
               prefill: bool = False,
               prefill_lengths: Optional[Array] = None) -> Array:
    """Apply depthwise convolution to the input.

    Args:
      x: the inputs
      decode: Whether to prepare and use an autoregressive cache.
      prefill: Whether to run a partial sequence to prefill the cache.
      prefill_lengths: The length of each partial sequence we are filling in the
        cache

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    x = jnp.asarray(x, jnp.float32)
    features = tuple(x.shape[-self.num_feature_dims:])
    kernel_size = 1 + self.radius * (1 if self.autoregressive else 2)

    def _make_scale_variable(shift_distance):
      init_value = 0.5 if shift_distance == 0 else 0.5 / kernel_size
      if shift_distance < 0:
        name = 'conv_m%d' % -shift_distance
      else:
        name = 'conv_%d' % shift_distance
      return flax_partitioning.param_with_axes(
          name,
          constant_init(init_value),
          features,
          jnp.float32,
          axes=tuple(self.axis_names),
      )

    if prefill and decode:
      raise ValueError('prefill and decode cannot both be true at the same'
                       'time. If you are using a prefix LM with bidirectional '
                       'attention on the inputs, please make a call with '
                       'prefill=True that includes an attention mask that '
                       'covers your inputs first and then make your decoding '
                       'calls.')
    # During fast autoregressive decoding, we process one position at a time,
    # and cache the few past activations we need.
    if decode or prefill:
      if not self.autoregressive:
        raise ValueError(
            'decode flag should never be set for non-autoregressive conv')
      is_initialized = self.has_variable('cache', 'cached_x_0')
      x_shape = list(x.shape)
      if is_initialized and decode:
        # actual incremental decoding
        if x.shape[self.length_dim] != 1:
          raise ValueError('penultimate dimension (length) must be 1 - got %s' %
                           (x.shape,))
      else:
        # Not actually decoding - just setting up loop vars
        x_shape = (
            x_shape[:self.length_dim] + [1] + x_shape[self.length_dim + 1:])
      cached_x = []
      for shift_distance in range(0, self.radius):
        cached_x.append(
            self.variable('cache', 'cached_x_%d' % shift_distance, jnp.zeros,
                          x_shape, x.dtype))
      if is_initialized and decode:
        values = [x] + [v.value for v in cached_x]
        ret = sum([
            v * _make_scale_variable(shift_distance)
            for shift_distance, v in enumerate(values)
        ])
        for shift_distance in range(0, self.radius):
          cached_x[shift_distance].value = values[shift_distance]
        return ret
      elif prefill:
        if prefill_lengths is None:
          raise NotImplementedError(
              'We need prefill lengths when prefill is set')
        for shift_distance in range(0, self.radius):
          length = x.shape[self.length_dim]
          position = prefill_lengths - (1 + shift_distance)
          onehot = common_utils.onehot(position, num_classes=length)
          if len(x.shape) == 4 or self.use_in_mlp_parallel_fused_layer:
            selected = jnp.einsum('...l, ...lmd->...md', onehot, x)
          else:
            selected = jnp.einsum('...l, ...ld->...d', onehot, x)
          selected = jnp.expand_dims(selected, 1)
          cached_x[shift_distance].value = selected

    ret = x * _make_scale_variable(0)
    x_shifted = x
    for shift_distance in range(1, self.radius + 1):
      # TODO: mask between packed sequences
      x_shifted = roll_with_zeros_along_axis(x_shifted, 1, axis=self.length_dim)
      ret += x_shifted * _make_scale_variable(shift_distance)
    if not self.autoregressive:
      x_shifted = x
      for shift_distance in range(1, self.radius + 1):
        # TODO: mask between packed sequences
        x_shifted = roll_with_zeros_along_axis(
            x_shifted, -1, axis=self.length_dim)
        ret += x_shifted * _make_scale_variable(-shift_distance)
    return ret
