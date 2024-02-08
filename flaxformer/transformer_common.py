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

"""Common Transformer classes."""
import dataclasses
from typing import Any, Callable, Dict, List, Optional

from flax import linen as nn

from flaxformer.architectures.common import param_remapping
from flaxformer.types import Array


class LayerSequence(nn.Module, param_remapping.ParameterRemappable):
  """Common object responsible for holding and applying Transformer layers.

  Can be used in encoders/decoders and supports applying a range of sublayers.

  Attributes:
    num_layers: The number of Transformer layers to create for this encoder.
    make_layer: A function that returns a single encoder layer.
    extra_modules: Extra modules provided to each make_layer call as kwargs.
  """
  num_layers: int
  make_layer: Callable[..., Any]
  extra_modules: Optional[Dict[str, Optional[nn.Module]]] = None

  def setup(self):
    kwargs = self.extra_modules if self.extra_modules is not None else {}
    self.layers = [self.make_layer(**kwargs) for _ in range(self.num_layers)]

  def __call__(self, inputs: Array, *args, **kwargs) -> Array:
    """Applies all Transformer layers to the inputs sequentially.

    Args:
      inputs: The inputs to the first layer <float>[..., seq_len, hidden_size].
        Typically these are the embedded token IDs, combined with embedded
        position IDs (or sinusoidal position encodings) and segment IDs.
      *args: Positional arguments to be passed to each layer.
      **kwargs: Keyword arguments to be passed to each layer.

    Returns:
      The encoded inputs <float>[..., seq_len, hidden_size].
    """
    # Apply all layers and return the output of the last layer.
    return self.apply_range_of_layers(0, None, inputs, *args, **kwargs)

  def apply_range_of_layers(self, start_idx: int, end_idx: Optional[int],
                            inputs: Array, *args, **kwargs) -> Array:
    """Passes the inputs to layers [start_idx, end_idx) and returns the output.

    Args:
      start_idx: The first layer to be applied to the inputs. Numeration starts
        from layer zero.
      end_idx: The last layer to be applied to the inputs. This layer is
        excluded from the interval, i.e. outputs will be returned from layers in
        the [start_idx, end_idx) interval. You can set this to None to apply all
        layers starting from start_idx.
      inputs: The inputs to the first layer. [batch_size..., length, features]
      *args: Positional arguments to be passed to each layer.
      **kwargs: Keyword arguments to be passed to each layer.

    Returns:
      The output of the last layer that was applied.
    """
    current_activations = inputs
    for layer in self.layers[start_idx:end_idx]:
      current_activations = layer(current_activations, *args, **kwargs)
    return current_activations


@dataclasses.dataclass(frozen=True)
class TransparentLayerSequence:
  """Version of LayerSequence that doesn't add pytree keys.

  Normally one should instantiate the layers in a parent module.

  Attributes:
    layers: List of nn.Modules, which should be owned by a parent Flax module.
  """
  layers: List[nn.Module]

  def __call__(self, inputs: Array, *args, **kwargs) -> Array:
    """Applies all Transformer layers to the inputs sequentially.

    Args:
      inputs: The inputs to the first layer <float>[..., seq_len, hidden_size].
        Typically these are the embedded token IDs, combined with embedded
        position IDs (or sinusoidal position encodings) and segment IDs.
      *args: Positional arguments to be passed to each layer.
      **kwargs: Keyword arguments to be passed to each layer.

    Returns:
      The encoded inputs <float>[..., seq_len, hidden_size].
    """
    # Apply all layers and return the output of the last layer.
    return self.apply_range_of_layers(0, None, inputs, *args, **kwargs)

  def apply_range_of_layers(self, start_idx: int, end_idx: Optional[int],
                            inputs: Array, *args, **kwargs) -> Array:
    """Passes the inputs to layers [start_idx, end_idx) and returns the output.

    Args:
      start_idx: The first layer to be applied to the inputs. Numeration starts
        from layer zero.
      end_idx: The last layer to be applied to the inputs. This layer is
        excluded from the interval, i.e. outputs will be returned from layers in
        the [start_idx, end_idx) interval. You can set this to None to apply all
        layers starting from start_idx.
      inputs: The inputs to the first layer. [batch_size..., length, features]
      *args: Positional arguments to be passed to each layer.
      **kwargs: Keyword arguments to be passed to each layer.

    Returns:
      The output of the last layer that was applied.
    """
    current_activations = inputs
    for layer in self.layers[start_idx:end_idx]:
      current_activations = layer(current_activations, *args, **kwargs)  # pytype: disable=not-callable
    return current_activations
