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

"""This file contains "architecture" classes for T5 models.

These are combinators which assemble components (LayerNorm, MLP, etc.) into
networks.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

from flax import linen as nn
import jax.numpy as jnp

from flaxformer import activation_partitioning
from flaxformer.architectures.common import param_remapping
from flaxformer.architectures.perceiver_ar import perceiver_ar_architecture
from flaxformer.components import relative_position_biases
from flaxformer.components import rich_attention_position_scores
from flaxformer.types import Array


# pylint: disable=not-callable
# pytype: disable=not-callable


class DecoderLayer(nn.Module, param_remapping.ParameterRemappable):
  """Transformer decoder layer.

  Forked from the original to support Perceiver AR slicing.

  Attributes:
    self_attention: An instance of a self-attention module.
    encoder_decoder_attention: Encoder-decoder attention module. This must be
      non-None if attending to encoded representations.
    mlp: The MLP module, applied after both attention modules.
    dropout_factory:  A callable that returns a new dropout instance. This is
      applied after the attention module.
    layer_norm_factory:  A callable that returns a new layer norm. This is
      applied before the attention module and before the MLP.
    relative_position_bias_factory: A callable that returns relative position
      bias instances. This should only be used for per-layer relative position
      biases; please use `shared_relative_position_bias` if they are shared
      among layers.
    shared_relative_position_bias: An instance of a shared relative position
      bias module, usually owned by the Decoder.
    activation_partitioning_dims: When set to 2, partitions intermediate
      variables containing the input and output of the decoder layer.
    parallel: whether to call attention and mlp in parallel
    sow_intermediates: whether to track intermediates using Module.sow.
    num_latents: Number of latents and outputs.
  """
  self_attention: nn.Module
  encoder_decoder_attention: Optional[nn.Module]
  mlp: nn.Module
  dropout_factory: Callable[[], nn.Module]
  layer_norm_factory: Callable[[], nn.Module]
  relative_position_bias_factory: Optional[Callable[[], nn.Module]] = None
  shared_relative_position_bias: Optional[nn.Module] = None
  activation_partitioning_dims: int = 1
  parallel: bool = False
  sow_intermediates: bool = False
  scanned: bool = False
  # num_latents is actually required, but has to be marked as optional because
  # we don't yet require Python 3.10, which provides keyword-only dataclasses.
  num_latents: Optional[int] = None

  def setup(self):
    if self.num_latents is None:
      raise ValueError('num_latents must be specified.')

    if (self.relative_position_bias_factory is not None and
        self.shared_relative_position_bias is not None):
      raise ValueError(
          'Please set at most one of relative_position_bias_factory and shared_relative_position_bias. '
          '(They can both be None however, e.g. for absolute position embeds.)')
    self.relpos_bias = (
        self.relative_position_bias_factory()
        if self.relative_position_bias_factory is not None else
        self.shared_relative_position_bias)

    # TODO: Support relative position bias.
    if self.relpos_bias is not None:
      raise NotImplementedError(
          'Relative position bias support not yet implemented for Perceiver AR.'
      )

    if self.parallel:
      self.layer_norm = self.layer_norm_factory()
      self.dropout = self.dropout_factory()
    else:
      self.pre_self_attention_layer_norm = self.layer_norm_factory()
      self.post_self_attention_dropout = self.dropout_factory()
      self.pre_cross_attention_layer_norm = self.layer_norm_factory()
      self.post_cross_attention_dropout = self.dropout_factory()
      self.pre_mlp_layer_norm = self.layer_norm_factory()
      self.post_mlp_dropout = self.dropout_factory()

  def get_bias(self, max_decode_length: Optional[int], decode: bool,
               layer_input: Array,
               encoded: Array) -> Tuple[Optional[Array], Optional[Array]]:
    decoder_bias = None
    encoder_decoder_bias = None
    if self.relpos_bias:
      if isinstance(self.relpos_bias,
                    relative_position_biases.RelativeAttentionAPI):
        if max_decode_length:
          relpos_length = max_decode_length
        else:
          relpos_length = layer_input.shape[-2]

        # during decoding, the layer will be called with decode=True first to
        # initialize the decoder cache, including a cached relpos bias cache.
        # the prefill codepath will call this once again with decode=False,
        # which is slightly wasteful but generally harmless. During subsequent
        # decode steps, this will be called with decode=True and will reuse the
        # cached bias. this significantly improves performance during decoding
        # with many decode steps.
        decoder_bias = self.relpos_bias(
            relpos_length, relpos_length, False, decode=decode)

      elif isinstance(self.relpos_bias,
                      rich_attention_position_scores.RichAttentionApi):
        decoder_bias = self.relpos_bias(
            layer_input,
            layer_input,
            bidirectional=False,
            is_cross_attention=False)
        encoder_decoder_bias = self.relpos_bias(
            layer_input, encoded, bidirectional=False, is_cross_attention=True)
      else:
        raise TypeError(
            f'{type(self.relpos_bias)} is not a supported relative position '
            f'bias factory.\nInstance value: {self.relpos_bias}')
    return decoder_bias, encoder_decoder_bias

  def __call__(self,
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
               prefill_lengths: Optional[Array] = None,
               num_latents: Optional[int] = None,
               sequence_lengths: Optional[Array] = None) -> Array:
    """Applies EncoderDecoder1DBlock module.

    Args:
      targets: Input data for decoder with shape [batch_size,
        decoder_seq_length, decoder_hidden_size].
      encoded: Input data from encoder with shape [batch_size,
        encoder_seq_length, decoder_hidden_size]. If None, block is Decoder
        only.
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask with shape [
        batch_size, 1, decoder_seq_length, encoder_seq_length].
      logit_mask: a mask (e.g., padding logit mask) to be applied to the
        attention logits.
      enable_dropout: Enables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache.
      max_decode_length: An optional integer specifying the maximum decoding
        length. Note that this is only used for defining the relative position
        embedding parameters.
      prefill: Whether to run a partial sequence to prefill the cache.
      prefill_lengths: The length of each partial sequence we are filling in the
        cache, lengths are inferred from the mask if not provided.
      num_latents: Used to override the number of output Perceiver AR latents
        during decoding.
      sequence_lengths: Lengths of all target sequences. Required for Perceiver
        AR operation.

    Returns:
      output after transformer encoder-decoder block.
    """
    if num_latents and num_latents > self.num_latents:
      raise ValueError(
          f'Overridden num_latents ({num_latents}) must be <= self.num_latents '
          f'({self.num_latents}).')
    num_latents = num_latents or self.num_latents

    layer_input = targets
    del targets

    # Decoder block.
    assert layer_input.ndim == 3
    layer_input = activation_partitioning.with_sharding_migration(
        layer_input,
        self.activation_partitioning_dims,
        logical_axis_names=('batch', 'length', 'embed'))

    if prefill and prefill_lengths is None:
      # Figure out how far each element in the batch fills the cache based
      # on the mask. We index each element in the batch, the first head
      # dim (because this is always set to one), and the first query
      # vector. If there is any prefix at all, the first element in the
      # prefix would be part of it.
      prefill_lengths = jnp.sum(
          decoder_mask[:, 0, 0, :], axis=-1).astype(jnp.int32)

    if self.parallel:
      x = self.layer_norm(
          layer_input,
          decode=decode,
          prefill=prefill,
          prefill_lengths=prefill_lengths)
      x = activation_partitioning.with_sharding_migration(
          x,
          self.activation_partitioning_dims,
          logical_axis_names=('batch', 'length', 'embed'))

      layer_input_residual, x_queries, query_position_offset, logit_mask_queries = (
          perceiver_ar_architecture.create_residuals_and_queries(
              layer_input,
              x,
              logit_mask,
              num_latents=num_latents,
              sequence_lengths=sequence_lengths))

      # Shared relative position embedding attention biases.
      decoder_bias, encoder_decoder_bias = self.get_bias(
          max_decode_length, decode, layer_input=x, encoded=encoded)

      y = (
          self.self_attention(
              x_queries,
              x,
              decoder_mask,
              decoder_bias,
              enable_dropout=enable_dropout,
              decode=decode,
              prefill=prefill,
              prefill_lengths=prefill_lengths,
              query_position_offset=query_position_offset) + self.mlp(
                  x_queries,
                  decode=decode,
                  prefill=prefill,
                  prefill_lengths=prefill_lengths,
                  enable_dropout=enable_dropout))
      if encoded is not None:
        y += self.encoder_decoder_attention(
            x,
            encoded,
            encoder_decoder_mask,
            encoder_decoder_bias,
            enable_dropout=enable_dropout)
      y *= (3 if encoded is not None else 2)**-0.5
      z = layer_input_residual + self.dropout(
          y, deterministic=not enable_dropout)
    else:
      # layer_input is derived from decoder_input_tokens.
      x = self.pre_self_attention_layer_norm(
          layer_input,
          decode=decode,
          prefill=prefill,
          prefill_lengths=prefill_lengths)
      x = activation_partitioning.with_sharding_migration(
          x,
          self.activation_partitioning_dims,
          logical_axis_names=('batch', 'length', 'embed'))

      layer_input_residual, x_queries, query_position_offset, logit_mask_queries = (
          perceiver_ar_architecture.create_residuals_and_queries(
              layer_input,
              x,
              logit_mask,
              num_latents=num_latents,
              sequence_lengths=sequence_lengths))

      if logit_mask is not None:
        # When using QKV fusion, x and x_queries must be the exact same
        # Python object, so reuse the object if possible.
        if x is x_queries and logit_mask is logit_mask_queries:
          x = logit_mask * x
          x_queries = x
        else:
          x = logit_mask * x
          x_queries = logit_mask_queries * x_queries

      # Shared relative position embedding attention biases.
      decoder_bias, encoder_decoder_bias = self.get_bias(
          max_decode_length, decode, layer_input=x, encoded=encoded)

      # The first and second arguments to the attention are the same,
      # i.e., this is a self-attention layer.
      x = self.self_attention(
          x_queries,
          x,
          decoder_mask,
          decoder_bias,
          enable_dropout=enable_dropout,
          decode=decode,
          prefill=prefill,
          prefill_lengths=prefill_lengths,
          query_position_offset=query_position_offset)
      x = layer_input_residual + self.post_self_attention_dropout(
          x, deterministic=not enable_dropout)
      x = activation_partitioning.with_sharding_migration(
          x,
          self.activation_partitioning_dims,
          logical_axis_names=('batch', 'length', 'embed'))

      # Encoder-Decoder block.
      if encoded is None:
        # If encoder outputs not provided, skip attending from decoder to
        # encoder.  This results in a decoder only block.
        y = x
      else:
        if self.encoder_decoder_attention is None:
          raise ValueError('Expected encoder_decoder_attention to be populated '
                           'when called with `encoded` inputs.')
        y = self.pre_cross_attention_layer_norm(
            x, decode=decode, prefill=prefill, prefill_lengths=prefill_lengths)
        y = activation_partitioning.with_sharding_migration(
            y,
            self.activation_partitioning_dims,
            logical_axis_names=('batch', 'length', 'embed'))

        if logit_mask is not None:
          y = logit_mask_queries * y

        y = self.encoder_decoder_attention(
            y,
            encoded,
            encoder_decoder_mask,
            encoder_decoder_bias,
            enable_dropout=enable_dropout)
        y = x + self.post_cross_attention_dropout(
            y, deterministic=not enable_dropout)
        y = activation_partitioning.with_sharding_migration(
            y,
            self.activation_partitioning_dims,
            logical_axis_names=('batch', 'length', 'embed'))

      # MLP block.
      z = self.pre_mlp_layer_norm(
          y, decode=decode, prefill=prefill, prefill_lengths=prefill_lengths)
      z = activation_partitioning.with_sharding_migration(
          z,
          self.activation_partitioning_dims,
          logical_axis_names=('batch', 'length', 'embed'))

      if logit_mask is not None:
        z = logit_mask_queries * z

      z = self.mlp(
          z,
          decode=decode,
          prefill=prefill,
          prefill_lengths=prefill_lengths,
          enable_dropout=enable_dropout)
      z = y + self.post_mlp_dropout(z, deterministic=not enable_dropout)
    z = activation_partitioning.with_sharding_migration(
        z,
        self.activation_partitioning_dims,
        logical_axis_names=('batch', 'length', 'embed'))
    if self.sow_intermediates:
      self.sow('intermediates', 'activations', z)

    # scan expects functions to have a signature: fn(carry, in) --> carry, out
    # TODO: automate this detail.
    if self.scanned:
      return z, None  # pytype: disable=bad-return-type  # jax-ndarray
    else:
      return z
