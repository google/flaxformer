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

"""Defines an API for "rich attention" mechanisms.

These require the entire input vector.

"""
import abc
from typing import Any, Callable

from flax import linen as nn
from flax.linen import initializers
from flax.linen import partitioning
from jax import lax
import jax.numpy as jnp
import numpy as np

from flaxformer.components import dense
from flaxformer.types import Array


class RichAttentionApi(metaclass=abc.ABCMeta):
  """Interface for relative attention APIs that need the entire input vector."""

  @abc.abstractmethod
  def __call__(self,
               q_inputs: Array,
               k_inputs: Array,
               bidirectional: bool = True,
               is_cross_attention: bool = False) -> Array:
    raise NotImplementedError()


