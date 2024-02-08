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

"""Defines architecture classes for h_transformer_1d models."""

import inspect
from typing import Callable, Optional, Any

from absl import logging
from flax import linen as nn
from jax import numpy as jnp
from typing_extensions import Protocol

from flaxformer import transformer_common as common
from flaxformer.architectures.common import param_remapping
from flaxformer.architectures.h_transformer import h_transformer_utils as utils
from flaxformer.architectures.h_transformer import partitioning
from flaxformer.components import embedding
from flaxformer.components import transforms
from flaxformer.components.attention import dense_attention
from flaxformer.types import Array

_SCAN_AXIS = 1


class MakeEncoderFn(Protocol):
  """Signature for functions that will make a low-level Encoder."""

  def __call__(
      self,
      *,
      shared_token_embedder: Optional[embedding.Embed] = None,
      spmd_annotations: Any = None,
  ) -> 'Encoder':
    """Makes a low-level Encoder instance.

    Args:
      shared_token_embedder: Shared token embedder instance, which should be
        passed to the returned module. If this is non-None, you should use it
        instead of providing your own token embedder.
      spmd_annotations: Optional SPMD annotations for scanned layers.

    Returns:
      Encoder instance.
    """


class EncoderLayer(nn.Module, param_remapping.ParameterRemappable):
  """H-Transformer encoder layer.

  Attributes:
    attention: The h_attention module.
    mlp: The MLP module, applied after attention.
    dropout_factory:  A callable that returns a new dropout instance. This is
      applied after the attention module.
    layer_norm_factory:  A callable that returns a new layer norm. This is
      applied before the attention module and before the MLP.
    parallel: Whether to call attention and mlp in parallel
    sow_intermediates: Whether to track intermediates using Module.sow.
    scanned: Whether this layer is being scanned over.
  """
  attention: nn.Module
  mlp: nn.Module
  dropout_factory: Callable[[], nn.Module]
  layer_norm_factory: Callable[[], nn.Module]
  partitioner_factory: Callable[[], Any] = partitioning.Partitioner1D
  parallel: bool = False
  sow_intermediates: bool = False
  scanned: bool = False

  def setup(self):
    self.pre_attention_layer_norm = self.layer_norm_factory()
    self.post_attention_dropout = self.dropout_factory()
    self.partitioner = self.partitioner_factory()
    if not self.parallel:
      self.pre_mlp_layer_norm = self.layer_norm_factory()
      self.post_mlp_dropout = self.dropout_factory()

  def _validate_inputs(self, inputs):
    if inputs.ndim != 3:
      raise ValueError(f'Expect inputs.ndim=3, but inputs.ndim={inputs.ndim}')

  def __call__(self,
               inputs: Array,
               inputs_mask: Array,
               *,
               enable_dropout: bool = True) -> Array:
    """Applies a single h_transformer encoder layer.

    Args:
      inputs: Input data with shape <float>[batch, length, emb_dim].
      inputs_mask: Input padding mask with shape <bool>[batch, length, emb_dim].
        Entries are True for non-padding tokens and False for padding tokens.
      enable_dropout: Enables dropout if set to True.

    Returns:
      Outputs from an h-transformer encoder layer.

    Raises:
      ValueError: This is triggered if inputs array has the wrong rank.
    """
    self._validate_inputs(inputs)

    layer_input = self.partitioner.annotate_layer_activation(inputs)
    layer_input = self.pre_attention_layer_norm(layer_input)
    layer_input = self.partitioner.annotate_layer_activation(layer_input)
    if self.parallel:
      y = (
          self.attention(
              layer_input, inputs_mask, enable_dropout=enable_dropout) +
          self.mlp(layer_input, enable_dropout=enable_dropout))
      # This scaling follows t5_architecture.py for compatibility. I suspect
      # that it is to make the scale comparable to that of layer_input. It is
      # possible that leaving it out makes no difference to the final results.
      # TODO: Remove this scaling once the integration tests confirm
      # that it is unnecessary.
      y *= 2**-0.5
      y = layer_input + self.post_attention_dropout(
          y, deterministic=not enable_dropout)
    else:
      # Attention block.
      x = self.attention(
          layer_input, inputs_mask, enable_dropout=enable_dropout)
      x = layer_input + self.post_attention_dropout(
          x, deterministic=not enable_dropout)
      x = self.partitioner.annotate_layer_activation(x)
      # MLP block.
      y = self.pre_mlp_layer_norm(x)
      y = self.partitioner.annotate_layer_activation(y)
      y = self.mlp(y, enable_dropout=enable_dropout)
      y = x + self.post_mlp_dropout(y, deterministic=not enable_dropout)
    y = self.partitioner.annotate_layer_activation(y)

    if self.sow_intermediates:
      self.sow('intermediates', 'activations', y)

    # Scan expects functions to have a signature: fn(carry, in) --> carry, out
    if self.scanned:
      return y, None  # pytype: disable=bad-return-type  # jax-ndarray
    else:
      return y


class EncoderAndDecoderLayers(nn.Module, param_remapping.ParameterRemappable):
  """Base class for Encoder and Decoder layers.

  Attributes:
    layer_factory: A callable that returns an EncoderLayer or DecoderOnlyLayer.
    input_dropout_factory: A callable that returns the dropout to apply to the
      input.
    output_dropout_factory: A callable that returns the dropout to apply to the
      output. Perhaps for legacy rather than essential reasons, the broadcasting
      pattern is sometimes different from input_dropout_factory().
    layer_norm_factory: A callable that returns a layer norm.
    num_layers: Number of layers to generate.
    layer_remat: Whether and how to apply jax.remat to each layer to perform
      recomputation in the backward pass.
    scan_layers: Whether to scan over layers.
    spmd_annotations: The spmd annotations needed for scanned layers.
  """
  layer_factory: Callable[[], nn.Module]
  input_dropout_factory: Callable[[], nn.Module]
  output_dropout_factory: Callable[[], nn.Module]
  layer_norm_factory: Callable[[], nn.Module]
  num_layers: int
  layer_remat: utils.LayerRematOptions = utils.LayerRematOptions.LEGACY
  scan_layers: bool = False
  spmd_annotations: Any = None

  def setup(self):
    self.input_dropout = self.input_dropout_factory()
    self.output_layer_norm = self.layer_norm_factory()
    self.output_dropout = self.output_dropout_factory()

  def _setup_layers(self, module_name: str,
                    num_arguments: int) -> Callable[..., Array]:
    lyrf = utils.maybe_remat(
        self.layer_factory,
        self.layer_remat,
        self.scan_layers,
        static_argnums=(2,))
    logging.info(
        'Finished setting up a set of %d h-transformer encoder/decoder layers,',
        self.num_layers)

    if self.scan_layers:
      initializing = self.is_mutable_collection('params')
      # We scan the parameters along axis 1 as an XLA layout optimization.
      params_spec = _SCAN_AXIS if initializing else transforms.ScanIn(
          _SCAN_AXIS)
      cache_spec = 0
      scan_annotation = (
          self.spmd_annotations[module_name]
          if self.spmd_annotations is not None else None)
      in_axes = (nn.broadcast,) * num_arguments
      lyrf = transforms.factory_scan(
          lyrf,
          in_axes=in_axes,
          variable_axes={
              'params': params_spec,
              'cache': cache_spec
          },
          split_rngs={
              'params': True,
              'dropout': True
          },
          length=self.num_layers,
          data_transform=transforms.inner_scan_spmd(scan_annotation,
                                                    _SCAN_AXIS),
          axes_collections=('params', 'cache'),
      )
      return lyrf()
    else:
      self.layers = [lyrf() for _ in range(self.num_layers)]
      return common.TransparentLayerSequence(self.layers)


class EncoderAndDecoderBase(EncoderAndDecoderLayers):
  """Base class for Encoder and Decoder classes.

  Attributes:
    token_embedder_factory: A callable that returns a token embedder. Please
      provide either this or `shared_token_embedder`.
    shared_token_embedder: A callable that returns a token embedder shared
      between both encoder and decoder.
  """
  # Embedders: Either a token_embedder_factory factory or shared_token_embedder
  # must be provided.
  token_embedder_factory: Optional[Callable[[], embedding.Embed]] = None
  shared_token_embedder: Optional[embedding.Embed] = None

  def setup(self):
    super().setup()
    self._setup_embedders()

  def _setup_embedders(self):
    if (self.token_embedder_factory,
        self.shared_token_embedder).count(None) != 1:
      raise ValueError(
          'Please set exactly one of token_embedder_factory or '
          'shared_token_embedder. The token_embedder_factory was '
          f'{self.token_embedder_factory}, and shared_token_embedder was '
          f'{self.shared_token_embedder}.')
    if self.shared_token_embedder is not None:
      self.embedder = self.shared_token_embedder
    else:
      self.token_embedder_factory: Callable[[], embedding.Embed]
      self.embedder = self.token_embedder_factory()
    logging.info('Finished setting up an embedder for h-transformer.')


class Encoder(EncoderAndDecoderBase):
  """A stack of input encoder layers."""

  def setup(self):
    super().setup()
    self.encoder = self._setup_layers('encoder', num_arguments=2)
    logging.info('Finished setting up h-transformer encoder.')

  def __call__(self,
               inputs: Array,
               inputs_mask: Optional[Array] = None,
               *,
               enable_dropout: bool = True) -> Array:
    """Applies H-Transformer encoder on the inputs.

    Args:
      inputs: Input data with shape <float>[batch, length]
      inputs_mask: Input padding mask with shape <bool>[batch, length], where
        True for non-padding tokens and False for padding.
      enable_dropout: Enables dropout if set to True.

    Returns:
      Outputs of an h-transformer encoder.

    Raises:
      ValueError: This is triggered if inputs array has the wrong rank.
    """
    if inputs.ndim != 2:  # (batch, len)
      raise ValueError(f'Expect inputs.ndim=2, but inputs.ndim={inputs.ndim}')

    embedded_inputs = self.embedder(inputs)
    embedded_inputs = self.input_dropout(
        embedded_inputs, deterministic=not enable_dropout)

    # Apply all encoder layers.
    encoder_outputs = self.encoder(
        embedded_inputs, inputs_mask, enable_dropout=enable_dropout)
    if self.scan_layers:
      encoder_outputs = encoder_outputs[0]

    # Post-process the outputs of the final encoder layer.
    encoder_outputs = self.output_layer_norm(encoder_outputs)
    encoder_outputs = self.output_dropout(
        encoder_outputs, deterministic=not enable_dropout)
    return encoder_outputs


class MakeDecoderFn(Protocol):
  """Signature for functions that make a low-level Decoder instance."""

  def __call__(
      self,
      *,
      shared_token_embedder: Optional[embedding.Embed] = None,
      spmd_annotations: Any = None,
  ) -> 'Decoder':
    """Makes a low-level Decoder instance.

    Args:
      shared_token_embedder: Shared token embedder instance, which should be
        passed to the returned module. If this is non-None, you should use it
        instead of providing your own token embedder.
      spmd_annotations: Optional SPMD annotations for scanned layers.

    Returns:
      Decoder instance.
    """


class DecoderOnlyLayer(EncoderLayer):
  """H-Transformer decoder-only layer.

  This decoder-only layer does not have cross attention. It is designed to be
  used by DecoderOnly below.

  A decoder performs exactly the same set of operations on the input as
  those by an encoder. This is because all input tokens are available due to
  standard teacher-forcing method during training phase. The implementation
  for both encoder and decoder is the same if we do not use the cache for
  decoder. This is fine for h-transformer Decoder because it has a linear
  complexity. So the runtime gain from using the cache is smaller.

  The only difference is that the attention component is an instance of
  OneDimDecoderSelfAttention which does not attend to future tokens.

  Attributes:
    attention: An instance of a OneDimDecoderSelfAttention module.
  """
  attention: nn.Module


class DecoderOnly(EncoderAndDecoderBase):
  """A stack of DecoderOnly layers.

  Attributes:
    output_logits_factory: A callable that returns the output logits. If not
      provided, then the token embedders are used.
    sow_intermediates: Whether to track intermediates using Module.sow.
  """
  output_logits_factory: Optional[Callable[[], nn.Module]] = None
  sow_intermediates: bool = False

  def setup(self):
    super().setup()
    self.decoder = self._setup_layers('decoder', num_arguments=2)
    self.output_logits_factory: Callable[[], nn.Module]
    self.output_logits: Optional[nn.Module]
    self.output_logits = (
        self.output_logits_factory() if self.output_logits_factory else None)
    logging.info('Finished setting up h-transformer decoder-only.')

  def __call__(self,
               inputs: Array,
               inputs_mask: Optional[Array] = None,
               decoder_target_tokens: Optional[Array] = None,
               decoder_segment_ids: Optional[Array] = None,
               decoder_positions: Optional[Array] = None,
               decoder_causal_attention: Optional[Array] = None,
               decode: Optional[bool] = False,
               *,
               enable_dropout: bool = True) -> Array:
    """Applies H-Transformer model on the inputs.

    Args:
      inputs: Input data with shape <float>[batch, length]
      inputs_mask: Input padding mask with shape <bool>[batch, length], where
        True for non-padding tokens and False for padding.
      decoder_target_tokens: target token to the decoder.
      decoder_segment_ids: decoder segmentation info for packed examples.
      decoder_positions: decoder subsequence positions for packed examples.
      decoder_causal_attention: a binary mask indicating the "inputs" portion of
        the concatenated sequence for a prefix LM.
      decode: Whether to prepare and use an autoregressive cache. This is unused
        in h-transformer.
      enable_dropout: Enables dropout if set to True.

    Returns:
      Outputs of an h-transformer encoder.

    Raises:
      ValueError: This is triggered if inputs array has the wrong rank or
        an unsupported argument is passed.
    """
    if decoder_segment_ids is not None or decoder_positions is not None:
      raise ValueError('Packed examples (segment IDs, positions) are not '
                       'supported by H-Transformer.')

    if inputs.ndim != 2:  # (batch, len)
      raise ValueError(f'Expect inputs.ndim=2, but inputs.ndim={inputs.ndim}')

    # These are in the argument list to conform to t5x.models.DecoderOnlyModel.
    # They are not used.
    del decoder_target_tokens
    del decoder_segment_ids
    del decoder_positions
    del decoder_causal_attention
    del decode

    embedded_inputs = self.embedder(inputs)
    embedded_inputs = self.input_dropout(
        embedded_inputs, deterministic=not enable_dropout)

    decoder_outputs = self.decoder(
        embedded_inputs, inputs_mask, enable_dropout=enable_dropout)
    if self.scan_layers:
      decoder_outputs = decoder_outputs[0]

    # Post-process the outputs of the final decoder layer.
    decoder_outputs = self.output_layer_norm(decoder_outputs)
    decoder_outputs = self.output_dropout(
        decoder_outputs, deterministic=not enable_dropout)
    logit_mask = dense_attention.get_decoder_logit_mask(inputs,
                                                        decoder_outputs.dtype)
    decoder_outputs = logit_mask * decoder_outputs

    if self.sow_intermediates:
      self.sow('intermediates', 'pre_logits_layer', decoder_outputs)

    # Decoded Logits
    if self.output_logits is not None:
      self.output_logits: nn.Module
      logits = self.output_logits(decoder_outputs)
    else:
      logits = self.embedder.attend(decoder_outputs)  # pytype: disable=attribute-error
      # Correctly normalizes pre-softmax logits for this shared embedder case.
      logits = logits / jnp.sqrt(decoder_outputs.shape[-1])
    return logits


class DecoderLayer(nn.Module, param_remapping.ParameterRemappable):
  """H-Transformer decoder layer with cross attention.

  Attributes:
    self_attention: An instance of DecoderSelfAttention module.
    encoder_decoder_attention: An instance of encoder-decoder cross-attention.
      If this is None, then this is a decoder-only layer.
    mlp: The MLP module, applied after both attention modules.
    dropout_factory:  A callable that returns a new dropout instance. This is
      applied after the attention module.
    layer_norm_factory:  A callable that returns a new layer norm. This is
      applied before the attention module and before the MLP.
    parallel: whether to call attention and mlp in parallel
    sow_intermediates: whether to track intermediates using Module.sow.
  """
  self_attention: nn.Module
  encoder_decoder_attention: Optional[nn.Module]
  mlp: nn.Module
  dropout_factory: Callable[[], nn.Module]
  layer_norm_factory: Callable[[], nn.Module]
  partitioner_factory: Callable[[], Any] = partitioning.Partitioner1D
  parallel: bool = False
  sow_intermediates: bool = False
  scanned: bool = False

  def setup(self):
    self.pre_self_attention_layer_norm = self.layer_norm_factory()
    self.partitioner = self.partitioner_factory()
    if self.parallel:
      self.dropout = self.dropout_factory()
    else:
      self.post_self_attention_dropout = self.dropout_factory()
      self.pre_cross_attention_layer_norm = self.layer_norm_factory()
      self.post_cross_attention_dropout = self.dropout_factory()
      self.pre_mlp_layer_norm = self.layer_norm_factory()
      self.post_mlp_dropout = self.dropout_factory()

  def __call__(self,
               decoder_inputs: Array,
               decoder_inputs_mask: Array,
               *,
               enable_dropout: bool = True,
               encoder_outputs: Optional[Array] = None,
               encoder_decoder_mask: Optional[Array] = None,
               logit_mask: Optional[Array] = None) -> Array:
    """Applies a single h_transformer decoder layer.

    Args:
      decoder_inputs: Input data for decoder with shape <float>[batch_size,
        decoder_seq_length, decoder_hidden_size].
      decoder_inputs_mask: Inputs mask for decoder with shape <bool>[batch_size,
        decoder_seq_length].
      enable_dropout: Enables dropout if set to True.
      encoder_outputs: Encoder outputs with shape [batch_size,
        encoder_seq_length, decoder_hidden_size]. If None, this is a DecoderOnly
        layer.
      encoder_decoder_mask: encoder-decoder attention mask with shape
        <bool>[batch_size, 1, decoder_seq_length, encoder_seq_length].
      logit_mask: a mask (e.g., padding logit mask) to be applied to the
        attention logits, with shape <bool>[batch_size, decoder_seq_length, 1].

    Returns:
      Outputs from an h-transformer decoder layer with shape <float>[batch_size,
        decoder_seq_length, decoder_hidden_size].

    Raises:
      ValueError: This is triggered if decoder_inputs array has the wrong rank
        or self.encoder_decoder_attention is not provided when encoder_outputs
        is not None.
    """
    if decoder_inputs.ndim != 3:
      raise ValueError('Expect decoder_inputs.ndim=3, but decoder_inputs.ndim='
                       f'{decoder_inputs.ndim}')
    if encoder_outputs is not None and self.encoder_decoder_attention is None:
      raise ValueError('Expected encoder_decoder_attention to be populated.')

    layer_inputs = self.partitioner.annotate_layer_activation(decoder_inputs)
    x = self.pre_self_attention_layer_norm(layer_inputs)
    x = self.partitioner.annotate_layer_activation(x)
    if self.parallel:
      y = (
          self.self_attention(
              x, decoder_inputs_mask, enable_dropout=enable_dropout) +
          self.mlp(x, enable_dropout=enable_dropout))
      if encoder_outputs is not None:
        y += self.encoder_decoder_attention(
            x,
            encoder_outputs,
            mask=encoder_decoder_mask,
            enable_dropout=enable_dropout)
      y *= (3 if encoder_outputs is not None else 2)**-0.5
      z = layer_inputs + self.dropout(y, deterministic=not enable_dropout)
    else:
      if logit_mask is not None:
        x = logit_mask * x

      x = self.self_attention(
          x, decoder_inputs_mask, enable_dropout=enable_dropout)
      x = layer_inputs + self.post_self_attention_dropout(
          x, deterministic=not enable_dropout)
      x = self.partitioner.annotate_layer_activation(x)

      # Encoder-Decoder block.
      if encoder_outputs is None:
        # If encoder outputs not provided, skip attending from decoder to
        # encoder.  This results in a decoder only layer.
        y = x
      else:
        y = self.pre_cross_attention_layer_norm(x)
        y = self.partitioner.annotate_layer_activation(y)

        if logit_mask is not None:
          y = logit_mask * y

        y = self.encoder_decoder_attention(
            y,
            encoder_outputs,
            mask=encoder_decoder_mask,
            enable_dropout=enable_dropout)
        y = x + self.post_cross_attention_dropout(
            y, deterministic=not enable_dropout)
        y = self.partitioner.annotate_layer_activation(y)

      # MLP block.
      z = self.pre_mlp_layer_norm(y)
      z = self.partitioner.annotate_layer_activation(z)

      if logit_mask is not None:
        z = logit_mask * z

      z = self.mlp(z, enable_dropout=enable_dropout)
      z = y + self.post_mlp_dropout(z, deterministic=not enable_dropout)

    z = self.partitioner.annotate_layer_activation(z)

    if self.sow_intermediates:
      self.sow('intermediates', 'activations', z)

    # scan expects functions to have a signature: fn(carry, in) --> carry, out
    # TODO: automate this detail.
    if self.scanned:
      return z, None  # pytype: disable=bad-return-type  # jax-ndarray
    else:
      return z


class Decoder(EncoderAndDecoderBase):
  """A stack of Decoder layers.

  This module can be used with or without the encoder stack. To use without an
  encoder, pass in encoder_outputs=None. This will bypass the encoder-decoder
  attention and hence results in a decoder-only block.

  Attributes:
    output_logits_factory: A callable that returns the output logits. If not
      provided, then the token embedders are used.
    sow_intermediates: Whether to track intermediates using Module.sow.
  """
  output_logits_factory: Optional[Callable[[], nn.Module]] = None
  sow_intermediates: bool = False

  def setup(self):
    super().setup()
    self.decoder = self._setup_layers('decoder', num_arguments=5)
    self.output_logits_factory: Callable[[], nn.Module]
    self.output_logits: Optional[nn.Module]
    self.output_logits = (
        self.output_logits_factory() if self.output_logits_factory else None)
    logging.info('Finished setting up h-transformer decoder.')

  def __call__(self,
               decoder_input_tokens: Array,
               encoder_outputs: Optional[Array] = None,
               decoder_positions: Optional[Array] = None,
               decoder_mask: Optional[Array] = None,
               encoder_decoder_mask: Optional[Array] = None,
               *,
               segment_ids: Optional[Array] = None,
               enable_dropout: bool = True,
               decode: Optional[bool] = False,
               max_decode_length: Optional[int] = None,
               prefill: Optional[bool] = False,
               prefill_lengths: Optional[Array] = None) -> Array:
    """Applies H-Transformer model on the decoder_input_tokens.

    Args:
      decoder_input_tokens: Decoder input tokens with shape <int>[batch,
        decoder_seq_length].
      encoder_outputs: Encoder outputs with shape <float>[batch,
        encoder_seq_length, encoder_hidden_size]. If None, decoder hidden layer
        does not attend to encoder outputs, resulting in a decoder only block.
      decoder_positions: Decoder subsequence positions for packed examples. This
        is unused in h-transformer.
      decoder_mask: Decoder input padding mask with shape <bool>[batch, length],
        where True for non-padding tokens and False for padding tokens.
      encoder_decoder_mask: The attention mask for the encoder outputs.
      segment_ids: decoder segmentation info for packed examples. This is unused
        in h-transformer.
      enable_dropout: Enables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache. This is unused
        in h-transformer.
      max_decode_length: An optional integer specifying the maximum decoding
        length. This is unused in h-transformer.
      prefill: Whether to run a partial sequence to prefill the cache.
      prefill_lengths: The length of each partial sequence we are filling in the
        cache. This is unused in h-transformer.

    Returns:
      Outputs from an h-transformer decoder block with shape <float>[batch_size,
        decoder_seq_length, decoder_hidden_size].

    Raises:
      ValueError: This is triggered if decoder_input_tokens or encoder_outputs
      has the wrong rank or an unsupported argument is passed.
    """
    if decoder_input_tokens.ndim != 2:
      raise ValueError(
          f'Expect decoder_input_tokens.ndim=2, but decoder_input_tokens.ndim={decoder_input_tokens.ndim}'
      )

    if encoder_outputs is not None and encoder_outputs.ndim != 3:
      raise ValueError(
          f'Expect encoder_outputs.ndim=3, but encoder_outputs.ndim={encoder_outputs.ndim}'
      )

    if segment_ids is not None or decoder_positions is not None:
      raise ValueError('Packed examples (segment IDs, positions) are not '
                       'supported by H-Transformer.')

    if prefill or decode or prefill_lengths is not None:
      raise ValueError(
          'Autoregressive cache is not supported by H-Transformer.')

    # These are in the argument list to conform to t5_architecture.Decoder.
    # They are not used.
    del decoder_positions
    del segment_ids
    del decode
    del max_decode_length
    del prefill
    del prefill_lengths

    embedded_decoder_inputs = self.embedder(decoder_input_tokens)
    embedded_decoder_inputs = self.input_dropout(
        embedded_decoder_inputs, deterministic=not enable_dropout)

    logit_mask = None
    if encoder_outputs is not None:
      # Only needs logit_mask for the dense cross_attention attending to
      # the encoder_outputs.
      logit_mask = dense_attention.get_decoder_logit_mask(
          decoder_input_tokens, embedded_decoder_inputs.dtype)
    decoder_outputs = self.decoder(
        embedded_decoder_inputs,
        decoder_mask,
        enable_dropout=enable_dropout,
        encoder_outputs=encoder_outputs,
        encoder_decoder_mask=encoder_decoder_mask,
        logit_mask=logit_mask)
    if self.scan_layers:
      decoder_outputs = decoder_outputs[0]

    # Post-process the outputs of the final decoder layer.
    decoder_outputs = self.output_layer_norm(decoder_outputs)
    decoder_outputs = self.output_dropout(
        decoder_outputs, deterministic=not enable_dropout)
    if logit_mask is not None:
      decoder_outputs = logit_mask * decoder_outputs

    if self.sow_intermediates:
      self.sow('intermediates', 'pre_logits_layer', decoder_outputs)

    # Decoded Logits
    if self.output_logits is not None:
      self.output_logits: nn.Module
      logits = self.output_logits(decoder_outputs)
    else:
      logits = self.embedder.attend(decoder_outputs)  # pytype: disable=attribute-error
      # Correctly normalizes pre-softmax logits for this shared embedder case.
      logits = logits / jnp.sqrt(decoder_outputs.shape[-1])

    if self.sow_intermediates:
      self.sow('intermediates', 'logits', logits)
    return logits


class EncoderDecoder(nn.Module, param_remapping.ParameterRemappable):
  """H-Transformer EncoderDecoder Model for sequence to sequence translation.

  Attributes:
    encoder_factory: A callable that returns the lower-level Encoder object. If
      shared_token_embedder_factory is non-None, then the result of it will be
      passed as the `shared_token_embedder` argument to `encoder_factory`.
    decoder_factory: A callable that returns the lower-level Decoder object. If
      shared_token_embedder_factory is non-None, then the result of it will be
      passed as the `shared_token_embedder` argument to `decoder_factory`.
    scan_layers: whether to scan over layers.
    shared_token_embedder_factory: A callable that returns an embedder that can
      be shared between the encoder and decoder.
  """
  encoder_factory: MakeEncoderFn
  decoder_factory: MakeDecoderFn
  scan_layers: bool = False
  spmd_annotations: Any = None
  shared_token_embedder_factory: Optional[Callable[[], embedding.Embed]] = None

  def setup(self):
    self.shared_token_embedder_factory: Callable[[], embedding.Embed]
    self.token_embedder = (
        self.shared_token_embedder_factory()
        if self.shared_token_embedder_factory else None)

    # TODO: Clean up SPMD annotation code.
    if self.spmd_annotations is None:
      encoder_annotations = None
      decoder_annotations = None
    else:
      encoder_annotations = self.spmd_annotations['encoder']
      decoder_annotations = self.spmd_annotations['decoder']

    encoder_factory_params = tuple(
        inspect.signature(self.encoder_factory).parameters.keys())
    if 'spmd_annotations' in encoder_factory_params:
      self.encoder = self.encoder_factory(
          shared_token_embedder=self.token_embedder,
          spmd_annotations=encoder_annotations)
    else:
      self.encoder = self.encoder_factory(
          shared_token_embedder=self.token_embedder)

    decoder_factory_params = tuple(
        inspect.signature(self.decoder_factory).parameters.keys())
    if 'spmd_annotations' in decoder_factory_params:
      self.decoder = self.decoder_factory(
          shared_token_embedder=self.token_embedder,
          spmd_annotations=decoder_annotations)
    else:
      self.decoder = self.decoder_factory(
          shared_token_embedder=self.token_embedder)

    logging.info('Finished setting up h-transformer encoder-decoder.')

  @property
  def encoder_embedder(self) -> embedding.Embed:
    return self.encoder.embedder

  @property
  def decoder_embedder(self) -> embedding.Embed:
    return self.decoder.embedder

  def __call__(self,
               encoder_input_tokens: Array,
               decoder_input_tokens: Array,
               decoder_target_tokens: Array,
               encoder_segment_ids: Optional[Array] = None,
               decoder_segment_ids: Optional[Array] = None,
               encoder_positions: Optional[Array] = None,
               decoder_positions: Optional[Array] = None,
               *,
               enable_dropout: bool = True,
               decode: Optional[bool] = False,
               max_decode_length: Optional[int] = None) -> Array:
    """Applies H-Transformer encoder-decoder model on the inputs.

    This method requires both decoder_target_tokens and decoder_input_tokens,
    which is a shifted version of the former.

    Args:
      encoder_input_tokens: Inputs to encoder with shape <int>[batch, length].
      decoder_input_tokens: Inputs to decoder with shape <int>[batch, length].
      decoder_target_tokens: Target tokens to the decoder with shape
        <int>[batch, length].
      encoder_segment_ids: encoder segmentation info for packed examples. This
        is not used.
      decoder_segment_ids: decoder segmentation info for packed examples. This
        is not used.
      encoder_positions: encoder subsequence positions for packed examples. This
        is not used.
      decoder_positions: decoder subsequence positions for packed examples. This
        is not used.
      enable_dropout: Enables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache. This is not
        used.
      max_decode_length: An optional integer specifying the maximum decoding
        length. This is not used.

    Returns:
      logits array from h-transformer with shape <float>[batch_size,
        decoder_seq_length, decoder_hidden_size].
    """
    if encoder_segment_ids is not None or decoder_segment_ids is not None or (
        encoder_positions is not None or decoder_positions is not None):
      raise ValueError('Packed examples (segment IDs, positions) are not '
                       'supported by H-Transformer.')

    # These are here to conform to t5_architecture.EncoderDecoder.
    # They are not used.
    del encoder_segment_ids
    del decoder_segment_ids
    del encoder_positions
    del decoder_positions
    del decode
    del max_decode_length

    encoder_mask = encoder_input_tokens > 0
    encoder_outputs = self.encoder(
        encoder_input_tokens, encoder_mask, enable_dropout=enable_dropout)

    decoder_mask = decoder_target_tokens > 0
    encoder_decoder_mask = dense_attention.make_attention_mask(
        decoder_mask, encoder_mask, dtype=encoder_outputs.dtype)
    return self.decoder(
        decoder_input_tokens,
        encoder_outputs=encoder_outputs,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        enable_dropout=enable_dropout)
