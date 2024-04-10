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

"""Perceiver AR Architecture implementation.

As described in:
"General-purpose, long-context autoregressive modeling with Perceiver AR"
https://arxiv.org/abs/2202.07765
"""

import dataclasses
from typing import List, Optional, Tuple

from flax import linen as nn
import jax.numpy as jnp

from flaxformer.architectures.perceiver_ar import attention
from flaxformer.architectures.perceiver_ar import slicing
from flaxformer.architectures.t5 import t5_architecture
from flaxformer.types import Array


@dataclasses.dataclass(frozen=True)
class PerceiverARTransparentLayerSequence:
  """Perceiver AR version of TransparentLayerSequence that manages slicing.

  The decoder_mask is different for the first layer vs. the remaining layers.
  Similar for the logit mask and prefill lengths. It's better to do the change
  outside of the scan-over-layers so that it is done only once.

  Attributes:
    layers: List of nn.Modules, which should be owned by a parent Flax module.
    num_latents: Number of latents and outputs.
  """
  layers: List[nn.Module]
  num_latents: int

  def __call__(self,
               inputs: Array,
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
    """Applies all Transformer layers to the inputs sequentially.

    Args:
      inputs: Input data for decoder with shape [batch_size, decoder_seq_length,
        decoder_hidden_size].
      encoded: required to be None, block is Decoder only, only kept for
        __call__ signature uniformity.
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: required to be None, block is Decoder only, only
        kept for __call__ signature uniformity.
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
      The encoded inputs <float>[..., seq_len, hidden_size].
    """
    if num_latents and num_latents > self.num_latents:
      raise ValueError(
          f'Overridden num_latents ({num_latents}) must be <= self.num_latents '
          f'({self.num_latents}).')
    num_latents = num_latents or self.num_latents

    current_activations = inputs
    for i, layer in enumerate(self.layers):
      layer_decoder_mask = decoder_mask

      if (layer_decoder_mask is not None and
          layer_decoder_mask.shape[-1] != current_activations.shape[-2]):
        assert i > 0
        # If we're in the self-attention stack, then kv should also be sliced.
        # From: [batch, 1, num_latents, input_length]
        # To: [batch, 1, num_latents, num_latents]
        assert layer_decoder_mask.shape[-1] >= current_activations.shape[-2]
        layer_decoder_mask = slicing.slice_sequences_vmap(
            layer_decoder_mask,
            sequence_lengths,
            num_latents,
            axis_within_vmap=-1)

      layer_prefill_lengths = prefill_lengths
      if prefill:
        if layer_prefill_lengths is None:
          layer_prefill_lengths = sequence_lengths

        # Ensure prefill_lengths isn't longer than the input length.
        # For Perceiver AR, this can happen in the self-attention stack, which
        # is narrower than the actual sequence length.
        layer_prefill_lengths = jnp.minimum(current_activations.shape[-2],
                                            layer_prefill_lengths)

      layer_logit_mask = logit_mask
      if (layer_logit_mask is not None and
          layer_logit_mask.shape[-2] != current_activations.shape[-2]):
        assert layer_logit_mask.shape[-2] >= current_activations.shape[-2]
        layer_logit_mask = slicing.slice_sequences_vmap(
            layer_logit_mask,
            sequence_lengths,
            current_activations.shape[-2],
            axis_within_vmap=0)

      current_activations = layer(
          current_activations,
          encoded,
          layer_decoder_mask,
          encoder_decoder_mask,
          logit_mask=layer_logit_mask,
          enable_dropout=enable_dropout,
          decode=decode,
          max_decode_length=max_decode_length,
          prefill=prefill,
          prefill_lengths=layer_prefill_lengths,
          num_latents=num_latents,
          sequence_lengths=sequence_lengths)
    return current_activations


class Decoder(t5_architecture.Decoder):
  """Perceiver AR Decoder.

  Attributes:
    num_latents: Number of latents for queries and number of output latents.
  """
  # num_latents is actually required, but has to be marked as optional because
  # we don't yet require Python 3.10, which provides keyword-only dataclasses.
  num_latents: Optional[int] = None

  def setup(self):
    if self.num_latents is None:
      raise ValueError('num_latents must be specified.')

    super().setup()

  def _setup_layer_sequence(self):
    lyrf = lambda: self.layer_factory(  # pylint: disable=g-long-lambda
        shared_relative_position_bias=self.relpos_bias)
    lyrf = t5_architecture.maybe_remat(
        lyrf,
        self.layer_remat,
        self.scan_layers,
        static_argnums=(5, 6, 7, 8, 9, 10))

    if not self.scan_layers:
      self.layers = [lyrf() for _ in range(self.num_layers)]
      return PerceiverARTransparentLayerSequence(self.layers, self.num_latents)
    else:
      # Create a non-scanned version of lyrf to use for the first layer.
      lyrf_notscanned = lambda: self.layer_factory(  # pylint: disable=g-long-lambda  # pytype: disable=wrong-keyword-args
          shared_relative_position_bias=self.relpos_bias,
          scanned=False)
      lyrf_notscanned = t5_architecture.maybe_remat(
          lyrf_notscanned,
          self.layer_remat,
          self.scan_layers,
          static_argnums=(5, 6, 7, 8, 9, 10))

      self.layers = [
          lyrf_notscanned(),
          self._construct_scanned_decoder(
              lyrf, self.num_layers - 1, num_broadcast_args=11)
      ]
      return PerceiverARTransparentLayerSequence(self.layers, self.num_latents)

  def decode_from_continuous_inputs(self,
                                    embedded_inputs,
                                    encoder_outputs,
                                    decoder_positions=None,
                                    decoder_mask=None,
                                    encoder_decoder_mask=None,
                                    logit_mask=None,
                                    *,
                                    enable_dropout: bool = True,
                                    decode: bool = False,
                                    max_decode_length: Optional[int] = None,
                                    prefill: bool = False,
                                    prefill_lengths: Optional[Array] = None,
                                    num_latents: Optional[int] = None,
                                    sequence_lengths: Optional[Array] = None):
    """Applies the decoder on the continuous (embedded) inputs."""
    if decoder_positions is not None:
      raise NotImplementedError('Perceiver AR does not yet support packing.')

    # sequence_lengths is required, but has to be defined as optional to
    # maintain API compatibility.
    if sequence_lengths is None:
      raise ValueError('sequence_lengths must be supplied fo Perceiver AR.')

    if num_latents and num_latents > self.num_latents:
      raise ValueError(
          f'Overridden num_latents ({num_latents}) must be <= self.num_latents '
          f'({self.num_latents}).')
    num_latents = num_latents or self.num_latents

    # If encoded is not given, this block is decoder only and does not contain
    # attention from decoder to encoder.
    if encoder_outputs is not None:
      assert encoder_outputs.ndim == 3  # (batch, len, depth)

    # Apply the decoder layers, attending to the encoder outputs (if provided),
    # and attending to previous decoder inputs (by masking future inputs).
    decoder_outputs = self.decoder(
        embedded_inputs,
        encoder_outputs,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        logit_mask=logit_mask,
        enable_dropout=enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length,
        prefill=prefill,
        prefill_lengths=prefill_lengths,
        num_latents=num_latents,
        sequence_lengths=sequence_lengths)

    if self.scan_layers:
      decoder_outputs = decoder_outputs[0]

    # Output length should always be <= the number of latents regardless of
    # input length or configured number of latents. During training it will be
    # the same. During fast decoding, it may just be 1.
    assert decoder_outputs.shape[-2] <= num_latents

    # Post-process final decoder layer outputs.
    decoder_outputs = self.decoder_norm(decoder_outputs)
    decoder_outputs = self.output_dropout(
        decoder_outputs, deterministic=not enable_dropout)

    # Slice logit_mask to match output positions.
    if logit_mask is not None:
      if logit_mask.shape[-2] != decoder_outputs.shape[-2]:
        assert logit_mask.shape[-2] >= decoder_outputs.shape[-2]
        logit_mask = slicing.slice_sequences_vmap(
            logit_mask,
            sequence_lengths,
            decoder_outputs.shape[-2],
            axis_within_vmap=-2)
      decoder_outputs = logit_mask * decoder_outputs

    if self.sow_intermediates:
      self.sow('intermediates', 'pre_logits_layer', decoder_outputs)

    # Decoded Logits
    if self.logits_dense is not None:
      logits = self.logits_dense(decoder_outputs)
    else:
      # Use the transpose of embedding matrix for logit transform.
      logits = self.embedder.embedders['token_ids'].attend(decoder_outputs)  # pytype: disable=attribute-error
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(decoder_outputs.shape[-1])

    if self.sow_intermediates:
      self.sow('intermediates', 'logits', logits)
    return logits


class DecoderOnly(t5_architecture.DecoderOnly):
  """Perceiver AR Decoder-only model."""
  # num_latents is actually required, but has to be marked as optional because
  # we don't yet require Python 3.10, which provides keyword-only dataclasses.
  num_latents: Optional[int] = None

  def setup(self):
    if self.num_latents is None:
      raise ValueError('num_latents must be specified.')

    super().setup()

  def __call__(self,
               decoder_input_tokens,
               decoder_target_tokens,
               decoder_segment_ids=None,
               decoder_positions=None,
               decoder_causal_attention=None,
               *,
               enable_dropout: bool = True,
               decode: bool = False,
               max_decode_length: Optional[int] = None,
               prefill: bool = False,
               prefill_lengths: Optional[Array] = None,
               num_latents: Optional[int] = None,
               **kwargs):
    """Applies Perceiver AR Decoder-only model on the inputs.

    This method requires both decoder_target_tokens and decoder_input_tokens,
    which is typically a shifted version of the former. For a packed dataset, it

    Packing is not currently supported for Perceiver AR.

    Args:
      decoder_input_tokens: input token to the decoder.
      decoder_target_tokens: target token to the decoder.
      decoder_segment_ids: decoder segmentation info for packed examples.
      decoder_positions: decoder subsequence positions for packed examples.
      decoder_causal_attention: a binary mask indicating the "inputs" portion of
        the concatenated sequence for a prefix LM.
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
      **kwargs: Additional keyword arguments to pass on to the decoder.

    Returns:
      logits array from LanguageModel.
    """
    if decode and prefill:
      raise ValueError('Only one of `decode` and `prefill` can be set. Use '
                       '`prefill` to pre-populate the cache for Prefix LMs '
                       'before using `decode`')

    # Perceiver AR operation does not support packing.
    if decoder_positions is not None:
      raise NotImplementedError(
          'decoder_positions is provided, but Perceiver AR does not yet '
          'support packing.')
    if decoder_segment_ids is not None:
      raise NotImplementedError(
          'decoder_segment_ids is provided, but Perceiver AR does not yet '
          'support packing.')

    if num_latents and num_latents > self.num_latents:
      raise ValueError(
          f'Overridden num_latents ({num_latents}) must be <= self.num_latents '
          f'({self.num_latents}).')
    num_latents = num_latents or self.num_latents

    # Calculate sequence lengths based on target tokens.
    sequence_lengths = slicing.get_sequence_lengths(
        decoder_target_tokens=decoder_target_tokens)

    if decode:
      decoder_mask = None
    else:
      decoder_mask = attention.make_decoder_mask(
          decoder_target_tokens=decoder_target_tokens,
          sequence_lengths=sequence_lengths,
          num_latents=num_latents,
          dtype=self.dtype,
          decoder_causal_attention=decoder_causal_attention)

    # We reuse Decoder class, which can optionally takes in encoded and
    # encoder_decoder_mask. These are used when Decoder is used in the context
    # of encoder-decoder model. For LM, we don't have an encoder. So set these
    # to None.
    return self.decoder(  # pytype: disable=attribute-error
        encoder_outputs=None,
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=None,
        segment_ids=decoder_segment_ids,
        enable_dropout=enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length,
        prefill=prefill,
        prefill_lengths=prefill_lengths,
        num_latents=num_latents,
        sequence_lengths=sequence_lengths,
        **kwargs)


def create_residuals_and_queries(
    layer_input: Array, x: Array, logit_mask, *, num_latents: Optional[Array],
    sequence_lengths: Array) -> Tuple[Array, Array, Optional[Array], Array]:
  """Slice layer inputs to get versions to use as queries."""
  if x.shape[-2] > num_latents:
    layer_input_residuals = slicing.slice_sequences_shard_map(  # pytype: disable=wrong-arg-types  # jax-ndarray
        layer_input, sequence_lengths, num_latents, axis_within_map=0
    )
    x_queries = slicing.slice_sequences_shard_map(  # pytype: disable=wrong-arg-types  # jax-ndarray
        x, sequence_lengths, num_latents, axis_within_map=0
    )
    query_offset = slicing.sequence_slice_start(sequence_lengths, num_latents)  # pytype: disable=wrong-arg-types  # jax-ndarray
  else:
    layer_input_residuals = layer_input
    x_queries = x
    query_offset = None

  if logit_mask.shape[-2] > num_latents:
    logit_mask_queries = slicing.slice_sequences_vmap(  # pytype: disable=wrong-arg-types  # jax-ndarray
        logit_mask, sequence_lengths, num_latents, axis_within_vmap=0)
  else:
    logit_mask_queries = logit_mask

  return layer_input_residuals, x_queries, query_offset, logit_mask_queries
