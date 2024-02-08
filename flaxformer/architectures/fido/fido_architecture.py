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

"""Modules for FiDO architecture (see https://arxiv.org/abs/2212.08153).

Use Decoder in this file for layer-sparse cross-attention, or standard T5
decoder with DecoderLayerBlock for scanned layer-sparse cross-attention. See
example configs t5_base_lsa.gin and t5_base_lsa_scan.gin.
"""

import dataclasses
from typing import List, Optional

import flax.linen as nn
import jax

from flaxformer.architectures.common import param_remapping
from flaxformer.architectures.t5 import t5_architecture


@dataclasses.dataclass(frozen=True)
class TransparentDecoderLayerSequence:
  """Layer sequence to apply cross attention every k layers."""

  layers: List[nn.Module]
  encoder_decoder_attention_period: int = 1

  def __call__(self, inputs: jax.Array, *args, **kwargs) -> jax.Array:
    """Applies all Transformer layers to the inputs sequentially."""
    return self.apply_range_of_layers(0, None, inputs, *args, **kwargs)

  def apply_range_of_layers(
      self,
      start_idx: int,
      end_idx: Optional[int],
      inputs: jax.Array,
      *args,
      **kwargs,
  ) -> jax.Array:
    """Split off encoded from args and pass only for selected layers."""
    current_activations = inputs
    encoded = args[0]
    for layer_idx, layer in enumerate(self.layers[start_idx:end_idx]):
      apply_encoder_decoder_attention = (
          (layer_idx + 1) % self.encoder_decoder_attention_period
      ) == 0
      current_activations = layer(
          current_activations,
          encoded if apply_encoder_decoder_attention else None,
          *args[1:],
          **kwargs,
      )  # pytype: disable=not-callable
    return current_activations


class Decoder(t5_architecture.Decoder):
  """Decoder with cross-attention every k layers.

  Use this class instead of the T5 decoder for layer-sparse cross-attention
  without scanned layers.

  Attributes:
    encoder_decoder_attention_period: apply cross-attention every this many
      layers. For example, if there are 24 decoder layers and
      encoder_decoder_attention_period=6, then layers 5, 11, 17 and 23 have
      cross-attention.
  """

  encoder_decoder_attention_period: int = 1

  def _setup_layer_sequence(self):
    lyrf = lambda: self.layer_factory(  # pylint: disable=g-long-lambda
        shared_relative_position_bias=self.relpos_bias
    )
    # If scanning layers, use standard T5 decoder with DecoderLayerBlock below.
    if self.scan_layers:
      raise ValueError('Scan not supported for this decoder class.')
    lyrf = t5_architecture.maybe_remat(
        lyrf, self.layer_remat, self.scan_layers, static_argnums=(5, 6, 7, 8, 9)
    )
    self.layers = [lyrf() for _ in range(self.num_layers)]
    return TransparentDecoderLayerSequence(
        self.layers,
        encoder_decoder_attention_period=self.encoder_decoder_attention_period,
    )


class DecoderLayerBlock(nn.Module, param_remapping.ParameterRemappable):
  """Block of decoder layers with single cross-attention layer.

  Use this class as a layer factory for the standard T5 decoder in order to
  employ layer-sparse cross-attention with scanned layers, with scanned equal to
  True in the decoder and this block, but not in the T5 Decoderlayer. Each block
  has a single cross-attention layer in the last layer of the block, so to build
  a decoder with 12 layers and sparsity 4, set num_layers=4 in this class and
  use 3 blocks in the T5 decoder.

  Attributes:
    num_layers: Number of decoder layers in block.
    layer_factory: T5 decoder layer factory.
    shared_relative_position_bias: Supply in case of shared relative position
      bias. Scanning normally prevents sharing relative position bias, but here
      we can optionally share within blocks.
    scanned: Whether the block is part of a scanned decoder. Normally true,
      otherwise no reason to use this module.
    layer_remat: Rematerialization strategy.
  """

  num_layers: int
  layer_factory: t5_architecture.MakeDecoderLayerFn
  shared_relative_position_bias: Optional[nn.Module] = None
  scanned: bool = True
  layer_remat: str = 'legacy'

  def setup(self):
    lyrf = lambda: self.layer_factory(  # pylint: disable=g-long-lambda
        shared_relative_position_bias=self.shared_relative_position_bias
    )
    lyrf = t5_architecture.maybe_remat(
        lyrf, self.layer_remat, False, static_argnums=(5, 6, 7, 8, 9)
    )
    self.layers = [lyrf() for _ in range(self.num_layers)]
    self.block_layers = TransparentDecoderLayerSequence(
        self.layers,
        encoder_decoder_attention_period=self.num_layers,
    )

  def __call__(
      self,
      targets,
      encoded,
      decoder_mask=None,
      encoder_decoder_mask=None,
      *,
      logit_mask=None,
      enable_dropout: bool = True,
      decode: bool = False,
      max_decode_length: Optional[int] = None,
      prefill: bool = False,
      prefill_lengths: Optional[jax.Array] = None,
      **kwargs,
  ):
    output = self.block_layers(
        targets,
        encoded,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        logit_mask=logit_mask,
        enable_dropout=enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length,
        prefill=prefill,
        prefill_lengths=prefill_lengths,
    )
    if self.scanned:
      return output, None
    else:
      return output
