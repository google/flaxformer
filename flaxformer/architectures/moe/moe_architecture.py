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

import dataclasses
import itertools
from typing import Any, Callable, List, Optional

from flax import linen as nn
from jax import numpy as jnp

from flaxformer.architectures.moe import moe_enums
from flaxformer.architectures.moe import moe_layers
from flaxformer.architectures.t5 import t5_architecture
from flaxformer.components import embedding
from flaxformer.components.attention import dense_attention
from flaxformer.types import Array
from flaxformer.types import DType

LayerLayout = moe_enums.LayerLayout
MoeLayer = moe_layers.MoeLayer


class SparseEncoder(nn.Module):
  """Encoder with configurable dense and sparse MLP modules.

  SparseEncoder does not currently support shared_relative_position_bias or
  scanned layers.

  Attributes:
    num_layers: Total number of encoder blocks in encoder.
    num_sparse_layers: Total number of sparse sublayers in encoder.
    sparse_layout: Placement of sparse modules within encoder. All other MLP
      sublayers are filled with dense MLP sublayers.
    sparse_layer_factory: A callable that returns a EncoderLayer containing a
      sparse MLP sublayer and an attention sublayer.
    dense_layer_factory: A callable that returns a EncoderLayer containing a
      dense MLP sublayer and attention sublayer.
    dropout_factory: A callable that returns a new dropout instance. We use the
      same dropout factor for both input and output.
    layer_norm_factory: A callable that returns a new layer norm. This is
      applied before the attention module and before the MLP module.
    position_embedder_factory: A callable that returns an absolute position
      embedder. Only provide this if you want absolute position embeddings.
    activation_partitioning_dims: When set to 2, partitions intermediate
      variables containing the input and output of the encoder layer.
    dtype: Numerical type of the computation (default: float32).
    shared_token_embedder: Shared token embedder instance.
    spmd_annotations: Optional SPMD annotations for scanned layers.
    use_logit_mask: Whether or not to mask out padding tokens at each encoder
      layer. Empirically, for T5, masking out the padding tokens was found to
      help stabilize the training of large models, although we have noted minor
      accuracy degradations for some sparse encoder configurations.
  """
  num_layers: int
  num_sparse_layers: int
  sparse_layout: LayerLayout
  sparse_layer_factory: t5_architecture.MakeEncoderLayerFn
  dense_layer_factory: t5_architecture.MakeEncoderLayerFn

  dropout_factory: Callable[[], nn.Module]
  layer_norm_factory: Callable[[], nn.Module]
  position_embedder_factory: Optional[Callable[
      [], embedding.Embedder[Array]]] = None
  activation_partitioning_dims: int = 1
  dtype: DType = jnp.float32
  shared_token_embedder: Optional[embedding.Embed] = None
  spmd_annotations: Any = None

  use_logit_mask: bool = True

  def setup(self):
    # is_sparse --> factory.
    layer_factories = {
        True: self.sparse_layer_factory,
        False: self.dense_layer_factory,
    }
    # Group layers by layer factory type (sparse or dense).
    layer_groups = _layer_groups(self.num_layers, self.num_sparse_layers,
                                 self.sparse_layout)

    encoders = []
    for layer_group in layer_groups:
      # Because we want Encoder layer output dropout and layer norms to only be
      # computed once after all EncoderLayer(s) have run, we use fake factories
      # for intermediate layers.
      layer_norm_factory = (
          self.layer_norm_factory
          if layer_group.is_final else _identity_factory)
      output_dropout_factory = (
          self.dropout_factory
          if layer_group.is_final else _fake_dropout_factory)
      encoders.append(
          t5_architecture.Encoder(
              num_layers=layer_group.num_layers,
              layer_factory=layer_factories[layer_group.is_sparse],
              input_dropout_factory=self.dropout_factory,
              output_dropout_factory=output_dropout_factory,
              layer_norm_factory=layer_norm_factory,
              position_embedder_factory=self.position_embedder_factory,
              dtype=self.dtype,
              shared_token_embedder=self.shared_token_embedder,
              spmd_annotations=self.spmd_annotations))

    self.encoders = encoders

  def __call__(self,
               inputs: Array,
               inputs_positions: Optional[Array] = None,
               encoder_mask: Optional[Array] = None,
               *,
               segment_ids: Optional[Array] = None,
               enable_dropout: bool = True):
    """Applies sequence of encoders to the inputs.

    Args:
      inputs: Input data.
      inputs_positions: Input subsequence positions for packed examples.
      encoder_mask: Encoder self-attention mask.
      segment_ids: Input segmentation info for packed examples.
      enable_dropout: Enables dropout if set to True.

    Returns:
      Output of the encoders.
    """
    continuous_outputs = self.encoders[0].embed_and_combine_inputs(
        inputs,
        inputs_positions=inputs_positions,
        segment_ids=segment_ids,
        enable_dropout=enable_dropout)

    # Because we have multiple Encoder(s) in a single SparseEncoder, the
    # logit_mask may be applied more often than in Flaxformer's default T5
    # Encoder architecture.
    if self.use_logit_mask:
      logit_mask = jnp.expand_dims(
          jnp.array((inputs > 0), dtype=continuous_outputs.dtype), axis=-1)
    else:
      logit_mask = None

    for encoder in self.encoders:
      continuous_outputs = encoder.encode_from_continuous_inputs(
          continuous_outputs,
          encoder_mask=encoder_mask,
          logit_mask=logit_mask,
          enable_dropout=enable_dropout)
    return continuous_outputs


class SparseDecoder(nn.Module):
  """Decoder with configurable dense and sparse MLP modules.

  SparseDecoder does not currently support shared_relative_position_bias or
  scanned layers.

  Attributes:
    num_layers: Total number of decoder blocks in decoder.
    sparse_layout: Placement of sparse MLP modules within decoder. All other MLP
      sublayers are filled with dense MLP sublayers.
    sparse_layer_factory: A callable that returns a DecoderLayer containing a
      sparse MLP sublayer and an self-attention sublayer.
    dense_layer_factory: A callable that returns a DecoderLayer containing a
      dense MLP sublayer and attention sublayers.
    dropout_factory: A callable that returns a new dropout instance. This is
      applied after the attention module.
    layer_norm_factory: A callable that returns a new layer norm. This is
      applied before the attention module and before the MLP module.
    position_embedder_factory: A callable that returns an absolute position
      embedder. Only provide this if you want absolute position embeddings.
    activation_partitioning_dims: When set to 2, partitions intermediate
      variables containing the input and output of the decoder layer.
    dtype: Numerical type of the computation (default: float32).
    output_logits_factory: Output projection factory for logits. Only used by
      the final decoder layer. All other decoder layers use the identity
      function to map their outputs to the next layer's inputs.
    shared_token_embedder: Shared token embedder instance.
    spmd_annotations: Optional SPMD annotations for scanned layers.
    use_logit_masK: Whether or not to mask out padding tokens at each decoder
      layer. Empirically, for T5, masking out the padding tokens was found to
      help stabilize the training of large models.
  """
  num_layers: int
  num_sparse_layers: int
  sparse_layout: LayerLayout
  sparse_layer_factory: t5_architecture.MakeDecoderLayerFn
  dense_layer_factory: t5_architecture.MakeDecoderLayerFn

  dropout_factory: Callable[[], nn.Module]
  layer_norm_factory: Callable[[], nn.Module]
  position_embedder_factory: Optional[Callable[
      [], embedding.Embedder[Array]]] = None
  activation_partitioning_dims: int = 1
  dtype: DType = jnp.float32
  output_logits_factory: Optional[Callable[[], nn.Module]] = None
  shared_token_embedder: Optional[embedding.Embed] = None
  spmd_annotations: Any = None

  use_logit_mask: bool = True

  def setup(self):
    # is_sparse --> factory.
    layer_factories = {
        True: self.sparse_layer_factory,
        False: self.dense_layer_factory,
    }
    # Group layers by layer factory type (sparse or dense).
    layer_groups = _layer_groups(self.num_layers, self.num_sparse_layers,
                                 self.sparse_layout)

    decoders = []
    for layer_group in layer_groups:
      # Because we want Decoder layer output logits and layer norms to only be
      # computed once after all DecoderLayer(s) have run, we use fake factories
      # for intermediate layers.
      layer_norm_factory = (
          self.layer_norm_factory
          if layer_group.is_final else _identity_factory)
      output_logits_factory = (
          self.output_logits_factory
          if layer_group.is_final else _identity_factory)

      # Dropout is handled similarly, except that we also need it for the first
      # decoder, which computes input embeddings. (This is not required above
      # for the Encoder because the Encoder separates input and output dropout
      # factories.)
      dropout_factory = (
          self.dropout_factory if (layer_group.is_first or layer_group.is_final)
          else _fake_dropout_factory)

      decoders.append(
          t5_architecture.Decoder(
              num_layers=layer_group.num_layers,
              layer_factory=layer_factories[layer_group.is_sparse],
              dropout_factory=dropout_factory,
              layer_norm_factory=layer_norm_factory,
              position_embedder_factory=self.position_embedder_factory,
              output_logits_factory=output_logits_factory,
              dtype=self.dtype,
              shared_token_embedder=self.shared_token_embedder,
              spmd_annotations=self.spmd_annotations))

    self.decoders = decoders

  def __call__(self,
               encoder_outputs,
               decoder_input_tokens,
               decoder_positions=None,
               decoder_mask=None,
               encoder_decoder_mask=None,
               *,
               segment_ids: Optional[Array] = None,
               enable_dropout: bool = True,
               decode: bool = False,
               max_decode_length: Optional[int] = None,
               prefill: bool = False,
               prefill_lengths: Optional[Array] = None):
    """Applies sequence of decoders to the inputs.

    Args:
      encoder_outputs: The outputs from the encoder. If None, do not attend to
        encoder outputs, resulting in a decoder only model (i.e. language
        model).
      decoder_input_tokens: The decoder input token IDs.
      decoder_positions: Decoder subsequence positions for packed examples.
      decoder_mask: Decoder self-attention mask.
      encoder_decoder_mask: The attention mask for the encoder outputs.
      segment_ids: Input segmentation info for packed examples.
      enable_dropout: Enables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache.
      max_decode_length: An optional integer specifying the maximum decoding
        length. Note that this is only used for defining the relative position
        embedding parameters.
      prefill: Whether to run a partial sequence to prefill the cache.
      prefill_lengths: The length of each partial sequence we are filling in the
        cache. Lengths are inferred from the mask if not provided.

    Returns:
      The decoder output logits for next token prediction.
    """
    continuous_outputs = self.decoders[0].embed_and_combine_inputs(
        decoder_input_tokens,
        decoder_positions=decoder_positions,
        segment_ids=segment_ids,
        enable_dropout=enable_dropout,
        decode=decode,
    )

    # Because we have multiple Decoder(s) in a single SparseDecoder, the
    # logit_mask may be applied more often than in Flaxformer's default T5
    # Decoder architecture.
    if self.use_logit_mask:
      logit_mask = dense_attention.get_decoder_logit_mask(
          decoder_input_tokens, continuous_outputs.dtype)
    else:
      logit_mask = None

    for decoder in self.decoders:
      continuous_outputs = decoder.decode_from_continuous_inputs(
          continuous_outputs,
          encoder_outputs,
          decoder_positions=decoder_positions,
          decoder_mask=decoder_mask,
          encoder_decoder_mask=encoder_decoder_mask,
          logit_mask=logit_mask,
          enable_dropout=enable_dropout,
          decode=decode,
          max_decode_length=max_decode_length,
          prefill=prefill,
          prefill_lengths=prefill_lengths)
    return continuous_outputs


@dataclasses.dataclass
class _LayerGroup:
  """Group of layers which share the same is_sparse setting.

  Attributes:
    is_sparse: Whether or not the group of layers is sparse.
    num_layers: The number of layers in the group.
    is_first: Whether or not this is the first layer group.
    is_final: Whether or not this is the final layer group.
  """
  is_sparse: bool
  num_layers: int
  is_first: bool
  is_final: bool


def _layer_groups(num_layers: int, num_sparse_layers: int,
                  sparse_layout: LayerLayout) -> List[_LayerGroup]:
  """Group layers by whether they are dense or sparse."""
  is_sparse = [
      _is_sparse_layer(i, num_layers, num_sparse_layers, sparse_layout)
      for i in range(num_layers)
  ]
  groups = [
      _LayerGroup(is_sparse, len(list(group)), False, False)
      for is_sparse, group in itertools.groupby(is_sparse)
  ]
  groups[0].is_first = True
  groups[-1].is_final = True
  return groups


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


def _identity_factory():
  """Returns an identity function."""
  return lambda x: x


def _fake_dropout_factory():
  """Returns an dropout-like mapping that leaves the input unchanged."""
  return lambda x, deterministic: x
