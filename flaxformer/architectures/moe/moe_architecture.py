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

"""Provides encoders/decoders with Mixture of Experts support."""

import functools
from typing import Callable, Optional, Sequence, Tuple, TypeVar, Union

from flax import linen as nn

from flaxformer import activation_partitioning
from flaxformer import transformer_common as common
from flaxformer.architectures.common import param_remapping
from flaxformer.architectures.moe import moe_enums
from flaxformer.architectures.moe import moe_layers
from flaxformer.architectures.t5 import t5_architecture
from flaxformer.components import embedding
from flaxformer.types import Array

T = TypeVar('T')  # Generic type

DecoderLayer = t5_architecture.DecoderLayer
EncoderLayer = t5_architecture.EncoderLayer
LayerLayout = moe_enums.LayerLayout
MakeDecoderLayerFn = t5_architecture.MakeDecoderLayerFn
MakeEncoderLayerFn = t5_architecture.MakeEncoderLayerFn
MoeLayer = moe_layers.MoeLayer


class SparseEncoderLayer(EncoderLayer):
  """Sparse Transformer encoder layer, with optional ST-MoE support.

  Dense/sparse encoder layers can be constructed by specifying `mlp` (parent
  class attribute) with a dense/sparse MoeLayer/MlpBlock. The `extra_mlp` allows
  for inserting an extra MLP dense module after the traditional encoder block,
  as in ST-MoE (https://arxiv.org/abs/2202.08906).

  Attributes:
    extra_mlp: Additional MLP module, applied after `mlp` module.
  """
  extra_mlp: Optional[nn.Module] = None

  def setup(self):
    if self.scanned:
      raise ValueError(
          'Individual SparseEncoderLayer(s) are never scanned over. Only '
          'blocks of MoE layers are ever scanned. Please leave '
          'SparseEncoderLayer.scanned = False.')
    super().setup()
    self.pre_extra_mlp_layer_norm = self.layer_norm_factory()
    self.post_extra_mlp_dropout = self.dropout_factory()

  def __call__(self,
               inputs: Array,
               encoder_mask: Optional[Array] = None,
               *,
               logit_mask: Optional[Array] = None,
               enable_dropout: bool = True) -> Array:
    """Applies a SparseEncoderLayer.

    Args:
      inputs: Input data with shape [batch, length, emb_dim].
      encoder_mask: Encoder self-attention mask.
      logit_mask: Encoder logits mask.
      enable_dropout: Enables dropout if set to True.

    Returns:
      Output after Transformer encoder block.
    """
    y = super().__call__(
        inputs,
        encoder_mask,
        logit_mask=logit_mask,
        enable_dropout=enable_dropout)

    if self.extra_mlp is None:
      return y  # pytype: disable=bad-return-type  # jax-ndarray

    z = self.pre_extra_mlp_layer_norm(y)
    z = activation_partitioning.with_sharding_migration(
        z,
        self.activation_partitioning_dims,
        logical_axis_names=('batch', 'length', 'embed'))

    if logit_mask is not None:
      z = logit_mask * z

    z = self.extra_mlp(z, enable_dropout=enable_dropout)  # pylint: disable=not-callable
    z = y + self.post_extra_mlp_dropout(z, deterministic=not enable_dropout)
    z = activation_partitioning.with_sharding_migration(
        z,
        self.activation_partitioning_dims,
        logical_axis_names=('batch', 'length', 'embed'))

    if self.sow_intermediates:
      self.sow('intermediates', 'extra_mlp_activations', z)

    return z  # pytype: disable=bad-return-type  # jax-ndarray


class SparseDecoderLayer(DecoderLayer):
  """Sparse Transformer encoder-decoder layer, with optional ST-MoE support.

  Dense/sparse encoder layers can be constructed by specifying `mlp` (parent
  class attribute) with a dense/sparse MoeLayer/MlpBlock. The `extra_mlp` allows
  for inserting an extra MLP dense module after the traditional decoder block,
  as in ST-MoE (https://arxiv.org/abs/2202.08906).

  Attributes:
    extra_mlp: Additional MLP module, applied after `mlp` module.
  """
  extra_mlp: Optional[nn.Module] = None

  def setup(self):
    if self.scanned:
      raise ValueError(
          'Individual SparseDecoderLayer(s) are never scanned over. Only '
          'blocks of MoE layers are ever scanned. Please leave '
          'SparseDecoderLayer.scanned = False.')
    super().setup()
    self.pre_extra_mlp_layer_norm = self.layer_norm_factory()
    self.post_extra_mlp_dropout = self.dropout_factory()

  def __call__(self,
               targets: Array,
               encoded: Array,
               decoder_mask: Optional[Array] = None,
               encoder_decoder_mask: Optional[Array] = None,
               *,
               logit_mask: Optional[Array] = None,
               enable_dropout: bool = True,
               decode: bool = False,
               max_decode_length: Optional[int] = None,
               prefill: bool = False,
               prefill_lengths: Optional[Array] = None) -> Array:
    """Applies SparseDecoderLayer module.

    Args:
      targets: Input data for decoder with shape [batch_size,
        decoder_seq_length, decoder_hidden_size].
      encoded: Input data from encoder with shape [batch_size,
        encoder_seq_length, decoder_hidden_size]. If None, block is Decoder
        only.
      decoder_mask: Decoder self-attention mask.
      encoder_decoder_mask: Encoder-decoder attention mask with shape [
        batch_size, 1, decoder_seq_length, encoder_seq_length].
      logit_mask: Mask (e.g., padding logit mask) to be applied to the attention
        logits.
      enable_dropout: Enables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache.
      max_decode_length: An optional integer specifying the maximum decoding
        length. Note that this is only used for defining the relative position
        embedding parameters.
      prefill: Whether to run a partial sequence to prefill the cache.
      prefill_lengths: The length of each partial sequence we are filling in the
        cache, lengths are inferred from the mask if not provided.

    Returns:
      Output after Transformer encoder-decoder block.
    """
    y = super().__call__(
        targets,
        encoded,
        decoder_mask,
        encoder_decoder_mask,
        logit_mask=logit_mask,
        enable_dropout=enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length,
        prefill=prefill,
        prefill_lengths=prefill_lengths)

    if self.extra_mlp is None:
      return y  # pytype: disable=bad-return-type  # always-use-return-annotations

    z = self.pre_extra_mlp_layer_norm(
        y, decode=decode, prefill=prefill, prefill_lengths=prefill_lengths)
    z = activation_partitioning.with_sharding_migration(
        z,
        self.activation_partitioning_dims,
        logical_axis_names=('batch', 'length', 'embed'))

    if logit_mask is not None:
      z = logit_mask * z

    z = self.extra_mlp(  # pylint: disable=not-callable
        z,
        decode=decode,
        prefill=prefill,
        prefill_lengths=prefill_lengths,
        enable_dropout=enable_dropout)
    z = y + self.post_extra_mlp_dropout(z, deterministic=not enable_dropout)
    z = activation_partitioning.with_sharding_migration(
        z,
        self.activation_partitioning_dims,
        logical_axis_names=('batch', 'length', 'embed'))

    if self.sow_intermediates:
      self.sow('intermediates', 'extra_mlp_activations', z)

    return z  # pytype: disable=bad-return-type  # always-use-return-annotations


class MoeEncoderScanBlock(nn.Module, param_remapping.ParameterRemappable):
  """Repeatable block of encoder layers that can be scanned over.

  Attributes:
    dense_layer_factory: A callable that returns a EncoderLayer containing a
      dense MLP sublayer and attention sublayers.
    sparse_layer_factory: A callable that returns a EncoderLayer containing a
      sparse MLP sublayer and attention sublayers.
    num_sparse_layers: Total number of sparse sublayers in encoder.
    num_layers: Total number of layers (dense and sparse) in encoder.
    sparse_layout: Placement of sparse modules within encoder. All other MLP
      sublayers are filled with dense MLP sublayers.
  """
  dense_layer_factory: MakeEncoderLayerFn
  sparse_layer_factory: MakeEncoderLayerFn
  num_layers: int
  num_sparse_layers: int
  sparse_layout: LayerLayout

  def setup(self) -> None:
    self.subblock: Sequence[EncoderLayer] = _scan_block_factory(  # pytype: disable=wrong-arg-types  # re-none
        self.dense_layer_factory, self.sparse_layer_factory, self.num_layers,
        self.num_sparse_layers, self.sparse_layout)

  def __call__(self,
               inputs: Array,
               encoder_mask: Optional[Array] = None,
               *,
               logit_mask: Optional[Array] = None,
               enable_dropout: bool = True) -> Tuple[Array, Optional[Array]]:
    """Applies a MoeEncoderScanBlock.

    Args:
      inputs: Input data with shape [batch, length, emb_dim].
      encoder_mask: Encoder self-attention mask.
      logit_mask: Encoder logits mask.
      enable_dropout: Enables dropout if set to True.

    Returns:
      Output after MoE encoder block.
    """
    hidden_state = inputs
    for layer in self.subblock:
      hidden_state = layer(
          hidden_state,
          encoder_mask,
          logit_mask=logit_mask,
          enable_dropout=enable_dropout)
    # scan expects functions to have a signature: fn(carry, in) --> carry, out
    return hidden_state, None  # pytype: disable=bad-return-type  # jax-ndarray


class MoeDecoderScanBlock(nn.Module, param_remapping.ParameterRemappable):
  """Repeatable block of decoder layers that can be scanned over.

  Attributes:
    dense_layer_factory: A callable that returns a DecoderLayer containing a
      dense MLP sublayer and attention sublayers.
    sparse_layer_factory: A callable that returns a DecoderLayer containing a
      sparse MLP sublayer and attention sublayers.
    num_sparse_layers: Total number of sparse sublayers in decoder.
    num_layers: Total number of layers (dense and sparse) in decoder.
    sparse_layout: Placement of sparse modules within decoder. All other MLP
      sublayers are filled with dense MLP sublayers.
  """
  dense_layer_factory: MakeDecoderLayerFn
  sparse_layer_factory: MakeDecoderLayerFn
  num_layers: int
  num_sparse_layers: int
  sparse_layout: LayerLayout

  def setup(self) -> None:
    self.subblock: Sequence[DecoderLayer] = _scan_block_factory(  # pytype: disable=wrong-arg-types  # re-none
        self.dense_layer_factory, self.sparse_layer_factory, self.num_layers,
        self.num_sparse_layers, self.sparse_layout)

  def __call__(
      self,
      targets: Array,
      encoded: Array,
      decoder_mask: Optional[Array] = None,
      encoder_decoder_mask: Optional[Array] = None,
      *,
      logit_mask: Optional[Array] = None,
      enable_dropout: bool = True,
      decode: bool = False,
      max_decode_length: Optional[int] = None,
      prefill: bool = False,
      prefill_lengths: Optional[Array] = None) -> Tuple[Array, Optional[Array]]:
    """Applies MoeDecoderScanBlock module.

    Args:
      targets: Input data for decoder with shape [batch_size,
        decoder_seq_length, decoder_hidden_size].
      encoded: Input data from encoder with shape [batch_size,
        encoder_seq_length, decoder_hidden_size]. If None, block is Decoder
        only.
      decoder_mask: Decoder self-attention mask.
      encoder_decoder_mask: Encoder-decoder attention mask with shape [
        batch_size, 1, decoder_seq_length, encoder_seq_length].
      logit_mask: Mask (e.g., padding logit mask) to be applied to the attention
        logits.
      enable_dropout: Enables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache.
      max_decode_length: An optional integer specifying the maximum decoding
        length. Note that this is only used for defining the relative position
        embedding parameters.
      prefill: Whether to run a partial sequence to prefill the cache.
      prefill_lengths: The length of each partial sequence we are filling in the
        cache, lengths are inferred from the mask if not provided.

    Returns:
      Output after MoE encoder-decoder block.
    """
    hidden_state = targets
    for layer in self.subblock:
      hidden_state = layer(
          hidden_state,
          encoded,
          decoder_mask,
          encoder_decoder_mask,
          logit_mask=logit_mask,
          enable_dropout=enable_dropout,
          decode=decode,
          max_decode_length=max_decode_length,
          prefill=prefill,
          prefill_lengths=prefill_lengths)
    # scan expects functions to have a signature: fn(carry, in) --> carry, out
    return hidden_state, None  # pytype: disable=bad-return-type  # always-use-return-annotations


class SparseEncoder(t5_architecture.Encoder):
  """A stack of encoder layers with configurable dense and sparse MLP modules.

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
                                  self.num_sparse_layers)
    self.sparse_layer_factory: MakeEncoderLayerFn

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
    # `layer_factory` is the "dense" layer factory.
    dense_layer_factory = lambda: self.layer_factory(  # pylint:disable=g-long-lambda
        shared_relative_position_bias=self.relpos_bias)
    sparse_layer_factory = lambda: self.sparse_layer_factory(  # pylint:disable=g-long-lambda
        shared_relative_position_bias=self.relpos_bias)

    if not self.scan_layers:
      layer_factory = functools.partial(
          _layer_factory,
          dense_layer_factory=dense_layer_factory,
          sparse_layer_factory=sparse_layer_factory,
          num_layers=self.num_layers,
          num_sparse_layers=self.num_sparse_layers,
          sparse_layout=self.sparse_layout)
      self.layers = [layer_factory(layer) for layer in range(self.num_layers)]
      self.encoder = common.TransparentLayerSequence(self.layers)
    else:
      # Convert to factory to conform with Flaxformer API.
      block_factory = lambda: MoeEncoderScanBlock(  # pylint:disable=g-long-lambda
          dense_layer_factory=dense_layer_factory,
          sparse_layer_factory=sparse_layer_factory,
          num_layers=self.num_layers,
          num_sparse_layers=self.num_sparse_layers,
          sparse_layout=self.sparse_layout)
      block_factory = t5_architecture.maybe_remat(
          block_factory,
          self.layer_remat,
          self.scan_layers,
          static_argnums=(3,))
      self.encoder = self._construct_scanned_encoder(
          block_factory,
          num_layers=_num_scan_blocks(self.num_layers, self.num_sparse_layers,
                                      self.sparse_layout))

    self.encoder_norm = self.layer_norm_factory()
    self.output_dropout = self.output_dropout_factory()


class SparseDecoder(t5_architecture.Decoder):
  """A stack of decoder layers with configurable dense and sparse MLP modules.

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
                                  self.num_sparse_layers)
    self.sparse_layer_factory: MakeDecoderLayerFn

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
    # `layer_factory` is the "dense" layer factory.
    dense_layer_factory = lambda: self.layer_factory(  # pylint:disable=g-long-lambda
        shared_relative_position_bias=self.relpos_bias)
    sparse_layer_factory = lambda: self.sparse_layer_factory(  # pylint:disable=g-long-lambda
        shared_relative_position_bias=self.relpos_bias)

    if not self.scan_layers:
      layer_factory = functools.partial(
          _layer_factory,
          dense_layer_factory=dense_layer_factory,
          sparse_layer_factory=sparse_layer_factory,
          num_layers=self.num_layers,
          num_sparse_layers=self.num_sparse_layers,
          sparse_layout=self.sparse_layout)
      self.layers = [layer_factory(layer) for layer in range(self.num_layers)]
      self.decoder = common.TransparentLayerSequence(self.layers)
    else:
      # Convert to factory to conform with Flaxformer API.
      block_factory = lambda: MoeDecoderScanBlock(  # pylint:disable=g-long-lambda
          dense_layer_factory=dense_layer_factory,
          sparse_layer_factory=sparse_layer_factory,
          num_layers=self.num_layers,
          num_sparse_layers=self.num_sparse_layers,
          sparse_layout=self.sparse_layout)
      block_factory = t5_architecture.maybe_remat(
          block_factory,
          self.layer_remat,
          self.scan_layers,
          static_argnums=(5, 6, 7, 8, 9))
      self.decoder = self._construct_scanned_decoder(
          block_factory,
          num_layers=_num_scan_blocks(self.num_layers, self.num_sparse_layers,
                                      self.sparse_layout),
          num_broadcast_args=9)

    self.decoder_norm = self.layer_norm_factory()
    self.output_dropout = self.dropout_factory()
    self.setup_output_logits()


def _validate_module_construction(
    sparse_layer_factory: Union[Optional[MakeEncoderLayerFn],
                                Optional[MakeDecoderLayerFn]],
    num_sparse_layers: Optional[int]):
  """Validates that sparse layer attributes are correctly specified."""
  if sparse_layer_factory is None:
    raise ValueError(
        'sparse_layer_factory must be specified but was left as None.')
  if num_sparse_layers is None:
    raise ValueError(
        'num_sparse_layers must be specified but was left as None.')


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


def _layer_factory(layer: int, dense_layer_factory: Callable[[], T],
                   sparse_layer_factory: Callable[[], T], num_layers: int,
                   num_sparse_layers: int, sparse_layout: Callable[[], T]) -> T:
  """Constructs a sparse or dense layer depending on the model configuration."""
  if _is_sparse_layer(layer, num_layers, num_sparse_layers, sparse_layout):
    return sparse_layer_factory()
  else:
    return dense_layer_factory()


def _scan_block_factory(dense_layer_factory: Callable[[], T],
                        sparse_layer_factory: Callable[[], T], num_layers: int,
                        num_sparse_layers: int,
                        sparse_layout: LayerLayout) -> Sequence[T]:
  """Constructs a repeatable block of layers that can be Scanned."""
  if num_sparse_layers == 0:
    return [dense_layer_factory()]

  if num_sparse_layers == num_layers:
    return [sparse_layer_factory()]

  if sparse_layout in [LayerLayout.BOTTOM, LayerLayout.MIDDLE, LayerLayout.TOP]:
    raise ValueError(
        'Scan is only supported for MIXED sparse (MoE) layer layouts. '
        f'Received: sparse_layout={sparse_layout}.')
  elif sparse_layout == LayerLayout.MIXED:
    if num_layers % num_sparse_layers != 0:
      raise ValueError(
          'For MIXED sparse (MoE) layer layouts, the number of '
          'sparse layers must divide evenly into the total number of '
          f'encoder/decoder layers, but num_layers={num_layers} while '
          f'num_sparse_layers={num_sparse_layers}')
    # Every sparse_index'th layer is sparse.
    sparse_index = num_layers // num_sparse_layers
    return ([dense_layer_factory() for _ in range(sparse_index - 1)] +
            [sparse_layer_factory()])
  else:
    raise ValueError('Unrecognized sparse layer layout: %s' % sparse_layout)


def _num_scan_blocks(num_layers: int, num_sparse_layers: int,
                     sparse_layout: LayerLayout) -> int:
  """Returns number of repeated MoE blocks that can be scanned over."""
  block = _scan_block_factory(
      dense_layer_factory=lambda: None,  # Unused
      sparse_layer_factory=lambda: None,  # Unused
      num_layers=num_layers,
      num_sparse_layers=num_sparse_layers,
      sparse_layout=sparse_layout)
  return num_layers // len(block)
