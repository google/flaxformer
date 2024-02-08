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

"""Enums for configuring MoE layer layouts and dispatch/router algorithms."""

import enum
import gin


@gin.constants_from_enum
class LayerLayout(str, enum.Enum):
  """Specifies how sparse and dense layers are interleaved in a model.

  Attributes:
    BOTTOM: Sparse layers come first in the network, followed by dense layers.
    MIDDLE: Sparse layers are in the middle of the network, sandwiched between
      dense layers.
    MIXED: Sparse layers are interleaved throughout the model; e.g. every third
      layer.
    TOP: Dense layers come first in the network, with sparse layers placed in
      the final few layers.
  """
  BOTTOM = 'bottom'
  MIDDLE = 'middle'
  MIXED = 'mixed'
  TOP = 'top'

  def __format__(self, format_spec: str) -> str:
    return self.value.__format__(format_spec)
