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

"""Initializers for Flaxformer models."""
from typing import Union
import jax
from jax import numpy as jnp
import numpy as np

from flaxformer.types import Array
from flaxformer.types import DType
from flaxformer.types import Initializer
from flaxformer.types import PRNGKey
from flaxformer.types import Shape


def sinusoidal(min_scale: float = 1.0,
               max_scale: float = 10000.0,
               dtype: DType = jnp.float32) -> Initializer:
  """Creates 1D Sinusoidal Position Embedding Initializer.

  Args:
    min_scale: Minimum frequency-scale in sine grating.
    max_scale: Maximum frequency-scale in sine grating.
    dtype: The DType of the returned values.

  Returns:
    The sinusoidal initialization function.
  """

  def init(key: PRNGKey, shape: Shape, dtype: DType = dtype) -> Array:
    """Sinusoidal init."""
    del key
    if dtype != np.float32:
      raise ValueError('The sinusoidal initializer only supports float32.')
    if len(list(shape)) != 2:
      raise ValueError(
          f'Expected a 2D shape (max_len, features), but got {shape}.')
    max_len, features = shape
    pe = np.zeros((max_len, features), dtype=dtype)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (features // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, features // 2) * scale_factor)
    pe[:, :features // 2] = np.sin(position * div_term)
    pe[:, features // 2:2 * (features // 2)] = np.cos(position * div_term)
    return jnp.array(pe)

  return init


def truncated_normal(mean: Union[float, Array] = 0.0,
                     stddev: Union[float, Array] = 0.05,
                     dtype: DType = jnp.float32) -> Initializer:
  """Returns an initialization function "truncated normal".

  This is the initialization that is used in the original BERT implementation.

  Args:
    mean: The mean of the random values to generate.
    stddev: The standard deviation of the random values to generate.
    dtype: dtype of the initialized values.

  Returns:
    The truncated normal initializer.
  """

  def init(key: PRNGKey, shape: Shape, dtype: DType = dtype) -> Array:
    return jax.random.truncated_normal(
        key=key, lower=-2., upper=2., shape=shape, dtype=dtype) * stddev + mean

  return init
