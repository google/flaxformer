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

"""L2 norm, which omits subtraction of mean or bias."""

from flax import linen as nn
from jax import lax
from jax import numpy as jnp

from flaxformer.types import Array
from flaxformer.types import DType


class L2Norm(nn.Module):
  """L2 normalization.

  Operates on the last axis of the input data.

  Attributes:
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the computation (default: float32).
  """
  epsilon: float = 1e-6
  dtype: DType = jnp.float32

  @nn.compact
  def __call__(self, x: Array) -> Array:
    """Applies l2 normalization on the input.

    Args:
      x: the inputs

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    x = jnp.asarray(x, jnp.float32)
    sum2 = jnp.sum(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(sum2 + self.epsilon), self.dtype)
    return y
