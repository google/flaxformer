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

"""Provides encoders/decoders with Mixture of Experts support."""

from typing import Callable, Optional, Union

from flaxformer import transformer_common as common
from flaxformer.architectures.moe import moe_enums
from flaxformer.architectures.moe import moe_layers
from flaxformer.architectures.t5 import t5_architecture
from flaxformer.components import embedding

LayerLayout = moe_enums.LayerLayout
MakeDecoderLayerFn = t5_architecture.MakeDecoderLayerFn
MakeEncoderLayerFn = t5_architecture.MakeEncoderLayerFn
MoeLayer = moe_layers.MoeLayer


class SparseEncoder(t5_architecture.Encoder):
  """A stack of encoder layers with configurable dense and sparse MLP modules.

  This module does NOT support scanned layers.

  Although some attributes below default to None, they MUST be specified by the
  user. We are forced to use defaults here as the parent Encoder class contains
  attributes with default values.

  Attributes:
    sparse_layer_factory: A callable that returns a EncoderLayer containing a
      sparse MLP sublayer and an attention sublayer. The "dense" variant of this
      factory is named `layer_factory` and is inherited from the super class.
    num_sparse_layers: Total number of sparse sublayers in encoder.
    sparse_layout: Placement of sparse modules within encoder. All other MLP
      sublayers are filled with dense MLP sublayers.
  """
  sparse_layer_factory: Optional[MakeEncoderLayerFn] = None
  num_sparse_layers: Optional[int] = None
  sparse_layout: LayerLayout = LayerLayout.MIXED

  def setup(self):
    _validate_module_construction(self.sparse_layer_factory,
                                  self.num_sparse_layers, self.scan_layers)

    if (self.token_embedder_factory,
        self.shared_token_embedder).count(None) != 1:
      raise ValueError(
          'Please set exactly one of token_embedder_factory or '
          'shared_token_embedder. token_embedder_factory was %s, and '
          'shared_token_embedder was %s.' %
          (self.token_embedder_factory, self.shared_token_embedder))
    if self.shared_token_embedder is not None:
      embedders = {'token_ids': self.shared_token_embedder}
    else:
      self.token_embedder_factory: Callable[[], embedding.Embed]
      self.token_embedder = self.token_embedder_factory()
      embedders = {'token_ids': self.token_embedder}
    if self.position_embedder_factory is not None:
      self.position_embedder_factory: Callable[[], embedding.Embed]
      self.position_embedder = self.position_embedder_factory()
      embedders['position_ids'] = self.position_embedder
    self.embedder = embedding.MultiEmbed(
        embedders,
        sow_intermediates=self.sow_intermediates,
        capture_gradients=self.capture_gradients)

    self.input_dropout = self.input_dropout_factory()

    self.relpos_bias = (
        self.shared_relative_position_bias_factory()
        if self.shared_relative_position_bias_factory is not None else None)

    def lyrf(layer: int) -> t5_architecture.EncoderLayer:
      """Encoder layer factory return sparse or dense layer."""
      if _is_sparse_layer(layer, self.num_layers, self.num_sparse_layers,
                          self.sparse_layout):
        return self.sparse_layer_factory(  # pylint: disable=not-callable
            shared_relative_position_bias=self.relpos_bias)
      else:
        # layer_factory is the "dense_layer_factory".
        return self.layer_factory(
            shared_relative_position_bias=self.relpos_bias)

    self.layers = [lyrf(layer) for layer in range(self.num_layers)]
    self.encoder = common.TransparentLayerSequence(self.layers)

    self.encoder_norm = self.layer_norm_factory()
    self.output_dropout = self.output_dropout_factory()


class SparseDecoder(t5_architecture.Decoder):
  """A stack of decoder layers with configurable dense and sparse MLP modules.

  This module does NOT support scanned layers.

  Although some attributes below default to None, they MUST be specified by the
  user. We are forced to use defaults here as the parent Decoder class contains
  attributes with default values.

  Attributes:
    sparse_layer_factory: A callable that returns a DecoderLayer containing a
      sparse MLP sublayer and attention sublayers. The "dense" variant of this
      factory is named `layer_factory` and is inherited from the super class.
    num_sparse_layers: Total number of sparse sublayers in decoder.
    sparse_layout: Placement of sparse modules within decoder. All other MLP
      sublayers are filled with dense MLP sublayers.
  """
  sparse_layer_factory: Optional[MakeDecoderLayerFn] = None
  num_sparse_layers: Optional[int] = None
  sparse_layout: LayerLayout = LayerLayout.MIXED

  def setup(self):
    _validate_module_construction(self.sparse_layer_factory,
                                  self.num_sparse_layers, self.scan_layers)

    if (self.token_embedder_factory,
        self.shared_token_embedder).count(None) != 1:
      raise ValueError(
          'Please set exactly one of token_embedder_factory or '
          'shared_token_embedder. token_embedder_factory was %s, and '
          'shared_token_embedder was %s.' %
          (self.token_embedder_factory, self.shared_token_embedder))
    if self.shared_token_embedder is not None:
      embedders = {'token_ids': self.shared_token_embedder}
    else:
      self.token_embedder_factory: Callable[[], embedding.Embed]
      self.token_embedder = self.token_embedder_factory()
      embedders = {'token_ids': self.token_embedder}
    if self.position_embedder_factory is not None:
      self.position_embedder_factory: Callable[[], embedding.Embed]
      self.position_embedder = self.position_embedder_factory()
      embedders['position_ids'] = self.position_embedder
    self.embedder = embedding.MultiEmbed(
        embedders,
        sow_intermediates=self.sow_intermediates,
        capture_gradients=self.capture_gradients)

    self.input_dropout = self.dropout_factory()

    self.relpos_bias = (
        self.shared_relative_position_bias_factory()
        if self.shared_relative_position_bias_factory is not None else None)

    def lyrf(layer: int) -> t5_architecture.DecoderLayer:
      """Decoder layer factory return sparse or dense layer."""
      if _is_sparse_layer(layer, self.num_layers, self.num_sparse_layers,
                          self.sparse_layout):
        return self.sparse_layer_factory(  # pylint: disable=not-callable
            shared_relative_position_bias=self.relpos_bias)
      else:
        # layer_factory is the "dense_layer_factory".
        return self.layer_factory(
            shared_relative_position_bias=self.relpos_bias)

    self.layers = [lyrf(layer) for layer in range(self.num_layers)]
    self.decoder = common.TransparentLayerSequence(self.layers)

    self.decoder_norm = self.layer_norm_factory()
    self.output_dropout = self.dropout_factory()
    self.setup_output_logits()


def _validate_module_construction(
    sparse_layer_factory: Union[Optional[MakeEncoderLayerFn],
                                Optional[MakeDecoderLayerFn]],
    num_sparse_layers: Optional[int], scan_layers: bool):
  """Validates that sparse layer attributes are correctly specified."""
  if sparse_layer_factory is None:
    raise ValueError(
        'sparse_layer_factory must be specified but was left as None.')
  if num_sparse_layers is None:
    raise ValueError(
        'num_sparse_layers must be specified but was left as None.')
  if scan_layers:
    raise ValueError(
        'Scanned layers are not yet supported in Mixture-of-Experts models.')


def _is_sparse_layer(layer: int, num_layers: int, num_sparse_layers: int,
                     sparse_layout: LayerLayout) -> bool:
  """Returns true if the current layer should be a sparse layer."""
  if sparse_layout == LayerLayout.BOTTOM:
    return layer < num_sparse_layers
  elif sparse_layout == LayerLayout.MIDDLE:
    return (num_layers - num_sparse_layers <= 2 * layer <
            num_layers + num_sparse_layers)
  elif sparse_layout == LayerLayout.MIXED and num_sparse_layers > 0:
    if num_layers % num_sparse_layers != 0:
      raise ValueError(
          'For MIXED sparse (MoE) layer layouts, the number of '
          'sparse layers must divide evenly into the total number of '
          f'encoder/decoder layers, but num_layers={num_layers} while '
          f'num_sparse_layers={num_sparse_layers}')
    # Every sparse_index'th layer is sparse.
    sparse_index = num_layers // num_sparse_layers
    return layer % sparse_index == sparse_index - 1
  elif sparse_layout == LayerLayout.TOP:
    return layer >= num_layers - num_sparse_layers
  else:
    return False
