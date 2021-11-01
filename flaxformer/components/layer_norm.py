# Copyright 2021 Google LLC.
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

"""T5 layer norm, which omits subtraction of mean or bias."""

from typing import Optional

from flax import linen as nn
from jax import lax
from jax import numpy as jnp

from flaxformer import sharding
from flaxformer.architectures.common import param_remapping
from flaxformer.types import Array
from flaxformer.types import DType
from flaxformer.types import Initializer


class T5LayerNorm(nn.Module, param_remapping.ParameterRemappable):
  """T5 Layer normalization.

  Operates on the last axis of the input data.

  Attributes:
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the computation (default: float32).
    scale_init: Initializer for scale, by default, one.
    use_scale: boolean - whether to scale by a learned per-channel value
    conv: optional convolution to happen after layer norm
  """
  epsilon: float = 1e-6
  dtype: DType = jnp.float32
  scale_init: Initializer = nn.initializers.ones
  use_scale: bool = True
  conv: Optional[nn.Module] = None

  @nn.compact
  def __call__(self,
               x: Array,
               decode: bool = False,
               prefill: bool = False,
               prefill_lengths: Optional[Array] = None) -> Array:
    """Applies layer normalization on the input.

    Args:
      x: the inputs
      decode: Passed through to optional convolution. Whether to prepare and use
        an autoregressive cache.
      prefill: Whether to run a partial sequence to prefill the cache.
      prefill_lengths: The length of each partial sequence we are filling in the
        cache.

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    x = jnp.asarray(x, jnp.float32)
    features = x.shape[-1]
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
    if self.conv is not None:
      y = self.conv(  # pylint: disable=not-callable
          y,
          decode=decode,
          prefill=prefill,
          prefill_lengths=prefill_lengths)
    if not self.use_scale:
      return y
    scale = self.param('scale', self.scale_init, (features,), jnp.float32)
    self.sow(
        'param_axes',
        'scale_axes',
        sharding.axis_names('embed'),
        reduce_fn=sharding.reduce_fn)
    scale = jnp.asarray(scale, self.dtype)
    return y * scale
