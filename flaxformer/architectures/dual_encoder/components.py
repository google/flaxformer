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

"""Reusable component modules."""

from flax import linen as nn
from flax.linen import partitioning
import jax.numpy as jnp
from flaxformer import types


class LearnableScaling(nn.Module):
  """A module with just one single learnable scalar."""

  dtype: types.DType = jnp.float32
  init_scaling_value: float = 100.0

  def setup(self):
    self.scalar = partitioning.param_with_axes(
        "learnable_scalar",
        nn.initializers.constant(self.init_scaling_value), (1,),
        jnp.float32,
        axes=("embed",))

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               enable_dropout: bool = True) -> jnp.ndarray:
    if enable_dropout:
      # Only apply logit scaling during training since during eval the scaling
      # will not affect the eval metrics.
      broadscast_scalar = jnp.expand_dims(self.scalar, axis=1)
      return jnp.asarray(x, self.dtype) * broadscast_scalar
    return jnp.asarray(x, self.dtype)
