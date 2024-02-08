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

"""Extensions to Jax/Flax core functions for Mixture of Experts training."""

from typing import Sequence

import jax.numpy as jnp

from flaxformer.types import Array


def scatter_nd(indices: Array, updates: Array, shape: Sequence[int]) -> Array:
  """JAX implementation of tf.scatter_nd.

  See https://www.tensorflow.org/api_docs/python/tf/scatter_nd, and
  https://github.com/google/jax/discussions/3658.

  Notes:
  - If multiple indices point to the same position, the output value at this
    position is accumulated.
  - Indices falling outside of the created array are quietly ignored.

  Args:
    indices: [num_items, n_dims] array of indices to update.
    updates: [num_items, ...] array of new data points.
    shape: Dimensions of the output array.

  Returns:
    An array of shape `shape` and the same type as `updates`, with updated
    values at given indices.
  """
  zeros = jnp.zeros(shape, updates.dtype)
  # Following `tf.scatter_nd`'s API, the inner vectors of `indices` have `n_dim`
  # values which index into `zeros`. We unpack it into arrays for each
  # dimension. This code is equivalent to `tf.unstack(indices, axis=-1)`.
  key = tuple(jnp.moveaxis(indices, -1, 0))
  return zeros.at[key].add(updates)
