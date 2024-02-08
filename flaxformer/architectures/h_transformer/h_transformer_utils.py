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

"""Utility classes and functions for h_transformer architectures."""

import enum
from typing import Callable, Tuple

from flax import linen as nn
import jax

from flaxformer.components import transforms


@enum.unique
class LayerRematOptions(enum.Enum):
  """Options for layer remat configuration.

  Attributes:
    NONE: For no use of jax.remat.
    MINIMAL: For recomputing only non-matmul ops in backprop.
    FULL: For recomputing the whole layer in backprop.
    LEGACY: For compatibility with existing configs. Previously
      scan_layers=False implied NONE, scan_layers=True implied FULL.
  """
  NONE = enum.auto()
  MINIMAL = enum.auto()
  FULL = enum.auto()
  LEGACY = enum.auto()


def maybe_remat(lyrf: Callable[[], nn.Module], layer_remat: LayerRematOptions,
                scan_layers: bool,
                static_argnums: Tuple[int, ...]) -> Callable[[], nn.Module]:
  """Maybe applies jax.remat with the indicated policy to a layer factory.

  Args:
    lyrf: Encoder or decoder layer factory.
    layer_remat: Config for per-layer remat. See commenst for LayerRematOptions.
    scan_layers: Whether to use jax.lax.scan for the stack of layers.
    static_argnums: The static_argnums to use for the jax.remat call.

  Returns:
    Potentially remat-wrapped layer factory.

  Raises:
    ValueError: This is triggered by an unsupported layer_mat option.
  """
  if layer_remat == LayerRematOptions.LEGACY:
    layer_remat = (
        LayerRematOptions.FULL if scan_layers else LayerRematOptions.NONE)

  if layer_remat == LayerRematOptions.NONE:
    return lyrf

  if layer_remat == LayerRematOptions.FULL:
    remat_policy = None
  elif layer_remat == LayerRematOptions.MINIMAL:
    remat_policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
  else:
    raise ValueError('Unsupported layer_remat option.')

  lyrf = transforms.factory_remat(
      lyrf,
      concrete=False,
      prevent_cse=False,
      static_argnums=static_argnums,
      policy=remat_policy)
  return lyrf
