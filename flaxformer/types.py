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

"""JAX generic types used as pytype annotations throughout Flaxformer."""

from typing import Callable, Sequence

import jax.numpy as jnp

Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
# TODO: Fix types in flax.linen such that we can use `Tuple[int, ...]`.
Shape = Sequence[int]

Activation = Callable[..., Array]

# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]
