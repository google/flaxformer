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

"""Provides T5 architecture with CALM decoding-time early-exiting."""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Callable, List, Optional, Tuple, Union

from flax import linen as nn
import jax.numpy as jnp

from flaxformer import activation_partitioning
from flaxformer import transformer_common as common
from flaxformer.architectures.calm_t5 import components as calm_components
from flaxformer.architectures.common import param_remapping
from flaxformer.architectures.t5 import t5_architecture
from flaxformer.components import embedding
from flaxformer.components import transforms
from flaxformer.components.attention import dense_attention
from flaxformer.types import Array
from flaxformer.types import DType


# pylint: disable=not-callable
# pytype: disable=not-callable


@dataclasses.dataclass(frozen=True)
class TransparentLayerSequencePartial(common.TransparentLayerSequence):
  """Extending TransparentLayerSequence for CALM.

  Supports passing only through a subsequence of all layers, and also allows
  returning all intermediate activations (used for anytime prediction training).

  Attributes:
    layers: List of nn.Modules, which should be owned by a parent Flax module.
    return_all_representations: Whether to return intermediate encodings.
  """
  layers: List[nn.Module]
  return_all_representations: bool = False

  def __call__(self, inputs: Array, *args, **kwargs) -> Array:
    """Applies Transformer layers to the inputs sequentially.

    Args:
      inputs: The inputs to the first layer <float>[..., seq_len, hidden_size].
        Typically these are the embedded token IDs, combined with embedded
        position IDs (or sinusoidal position encodings) and segment IDs.
      *args: Positional arguments to be passed to each layer.
      **kwargs: Keyword arguments to be passed to each layer.

    Returns:
      The encoded inputs <float>[..., seq_len, hidden_size].
      If return_all_representations is True, returns an Array
      <float>[..., end_idx - start_idx, seq_len, hidden_size]
    """
    start_idx = kwargs.pop('start_idx', 0)
    end_idx = kwargs.pop('end_idx', None)

    return self.apply_range_of_layers(start_idx, end_idx, inputs, *args,
                                      **kwargs)

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
      The output of the last layer that was applied, if
      self.return_all_representations is False. If True, returns the outputs
      of all layers (including both intermediate layers and the last one).
    """
    decode = kwargs.get('decode', False)
    current_activations = inputs
    all_activations = []
    for layer in self.layers[start_idx:end_idx]:
      current_activations = layer(current_activations, *args, **kwargs)  # pytype: disable=not-callable
      all_activations.append(current_activations)

    if self.return_all_representations and not decode:
      all_activations = jnp.array(all_activations)
      return all_activations
    else:
      return current_activations


class DecoderLayer(t5_architecture.DecoderLayer):
  """Extends DecoderLayer to allow cache propagation."""

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
               only_propagate_state: bool = False,
               **kwargs):
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
      only_propagate_state: Will run the decoder layer only until the key-value
        self-attention values are computed and stored in cache, and will exit
        after.
      **kwargs: Remaining keyword arguments. Passed to
        _create_residuals_and_queries.

    Returns:
      Output after transformer encoder-decoder block.
    """
    layer_input = targets
    del targets

    # Decoder block.
    assert layer_input.ndim == 3
    layer_input = activation_partitioning.with_sharding_migration(
        layer_input,
        self.activation_partitioning_dims,
        logical_axis_names=('batch', 'length', 'embed'))

    if only_propagate_state:
      # Calls the self attention in order to compute the key-values and
      # (automatically) store them in cache. They are computed with a dense
      # Transformation on over the inputs inside the self_attention module.
      if not isinstance(self.self_attention,
                        calm_components.MultiHeadDotProductAttention):
        raise TypeError(
            'Self-attention should be the one implemented in '
            'architectures/calm_t5/components.py to allow cache propagation. '
            f'Got {type(self.self_attention)}.'
        )

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

      # Normally a no-op unless overridden by a subclass.
      layer_input_residual, x_queries, logit_mask_queries = (
          self._create_residuals_and_queries(layer_input, x, logit_mask,
                                             **kwargs))

      # Shared relative position embedding attention biases.
      decoder_bias, encoder_decoder_bias = self.get_bias(
          max_decode_length, decode, layer_input=x, encoded=encoded)

      if only_propagate_state:
        self.self_attention(
            x_queries,
            x,
            decoder_mask,
            decoder_bias,
            enable_dropout=enable_dropout,
            decode=decode,
            prefill=prefill,
            prefill_lengths=prefill_lengths,
            only_propagate_state=True)
        return layer_input

      y = (
          self.self_attention(
              x_queries,
              x,
              decoder_mask,
              decoder_bias,
              enable_dropout=enable_dropout,
              decode=decode,
              prefill=prefill,
              prefill_lengths=prefill_lengths) + self.mlp(
                  x,
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

      # Normally a no-op unless overridden by a subclass.
      layer_input_residual, x_queries, logit_mask_queries = (
          self._create_residuals_and_queries(layer_input, x, logit_mask,
                                             **kwargs))

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

      if only_propagate_state:
        self.self_attention(
            x_queries,
            x,
            decoder_mask,
            decoder_bias,
            enable_dropout=enable_dropout,
            decode=decode,
            prefill=prefill,
            prefill_lengths=prefill_lengths,
            only_propagate_state=True)
        return layer_input

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
          prefill_lengths=prefill_lengths)
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
    if self.scanned:
      return z, None
    else:
      return z


class Decoder(nn.Module, param_remapping.ParameterRemappable):
  """A stack of decoder layers.

  This module can be used with or without the encoder stack. To use without an
  encoder, pass in encoded=None. This will bypass the encoder-decoder attention.

  Attributes:
    layer_factory: A callable that returns a DecoderLayer.
    dropout_factory: A callable that returns the dropout to apply to the input
      and before the final logits.
    layer_norm_factory: A callable that returns a layer norm.
    output_logits_factory: A callable that returns the output logits. If not
      provided, then the token embedders are used.
    num_layers: Number of layers to generate.
    dtype: DType to cast the embedded inputs.
    layer_remat: whether and how to apply jax.remat to each layer to perform
      recomputation in the backward pass. Supported values are 'none', for no
      use of jax.remat; 'minimal', for a policy that recomputes only non-matmul
      operations (typically optimal); and 'full', for full recomputation of each
      layer. The (legacy) default is to use 'none' when `scan_layers=False` and
      and 'full' when `scan_layers=True`.
    scan_layers: whether to scan over layers.
    spmd_annotations: spmd annotations needed for scanned layers.
    shared_relative_position_bias_factory: A callable that returns a relative
      position bias instance which will be shared for all encoder layers. Only
      set this if using shared relative position biases.
    token_embedder_factory: A callable that returns a token embedder. Please
      provide either this or `shared_token_embedder`.
    shared_token_embedder: A callable that returns a token embedder shared
      between both encoder and decoder.
    position_embedder_factory: A callable that returns an absolute position
      embedder. Only provide this if you want absolute position embeddings.
    return_all_logits: If true, instead of returning only the logits of the last
      layer. All logits of potential "exit" layers (determined by `first_exit`
      and `exit_interval`) are returned. Adds a dimension to the returned Array.
    first_exit: First layer to compute the logits from for `return_all_logits`.
    exit_interval: Interval between layers to for `return_all_logits`.
      from layer `first_exit`.
    sow_intermediates: whether to track intermediates using Module.sow.
    scan_axis: axis over which to do scan over layers.
    capture_gradients: whether to track input gradients using a variable in the
      `grads` collection. This captures the gradient of the (combined) embedded
      inputs, i.e. the input to the first encoder layer.
  """
  layer_factory: t5_architecture.MakeDecoderLayerFn
  dropout_factory: Callable[[], nn.Module]
  layer_norm_factory: Callable[[], nn.Module]
  num_layers: int
  dtype: DType = jnp.float32
  layer_remat: str = 'legacy'
  scan_layers: bool = False
  spmd_annotations: Any = None
  shared_relative_position_bias_factory: Optional[Callable[[],
                                                           nn.Module]] = None
  output_logits_factory: Optional[Callable[[], nn.Module]] = None

  # Embedders: Either a token_embedder_factory factory or shared token embedder
  # must be provided. The position embedder is optional and provided when
  # absolute position embeddings are desired.
  token_embedder_factory: Optional[Callable[[],
                                            embedding.Embedder[Array]]] = None
  shared_token_embedder: Optional[embedding.Embed] = None
  position_embedder_factory: Optional[Callable[
      [], embedding.Embedder[Array]]] = None
  return_all_logits: bool = False
  first_exit: int = 0  # Zero means the first contextual representations.
  exit_interval: int = 1

  meta_cls_factory: Optional[Callable[[], nn.Module]] = None

  sow_intermediates: bool = False
  scan_axis: int = 1
  capture_gradients: bool = False

  def setup(self):
    # Set up the embedders.
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

    if self.scan_layers and self.shared_relative_position_bias_factory:
      raise ValueError("Scanned layer mode doesn't support shared relative"
                       'position biases.')
    self.relpos_bias = (
        self.shared_relative_position_bias_factory()
        if self.shared_relative_position_bias_factory is not None else None)

    self.decoder = self._setup_layer_sequence()

    self.decoder_norm = self.layer_norm_factory()
    self.output_dropout = self.dropout_factory()
    self.setup_output_logits()
    self.setup_meta_cls()

  def _setup_layer_sequence(self):
    lyrf = lambda: self.layer_factory(  # pylint: disable=g-long-lambda
        shared_relative_position_bias=self.relpos_bias)
    lyrf = t5_architecture.maybe_remat(
        lyrf,
        self.layer_remat,
        self.scan_layers,
        static_argnums=(5, 6, 7, 8, 9))
    if not self.scan_layers:
      self.layers = [lyrf() for _ in range(self.num_layers)]
      return TransparentLayerSequencePartial(
          layers=self.layers, return_all_representations=self.return_all_logits)
    else:
      # TODO: add adaptive computation support with scan_layers.
      raise ValueError('Early exiting is not supported with scan_layers.')
      # return self._construct_scanned_decoder(lyrf, self.num_layers)

  def _construct_scanned_decoder(
      self,
      lyrf: Callable[[], nn.Module],
      num_layers: int,
      num_broadcast_args: int = 10) -> Callable[..., Array]:
    """Constructs decoder from layer factory using scan."""

    initializing = self.is_mutable_collection('params')
    # We scan the parameters along scan_axis (default =1) as
    # an XLA layout optimization.
    params_spec = self.scan_axis if initializing else transforms.ScanIn(
        self.scan_axis)
    cache_spec = 0
    intermediates_spec = 2  # Stacks intermediate layer outputs in dimension 2.
    scan_annotation = (
        self.spmd_annotations['decoder']
        if self.spmd_annotations is not None else None)
    lyrf = transforms.factory_scan(
        lyrf,
        in_axes=(nn.broadcast,) * num_broadcast_args,
        variable_axes={
            'params': params_spec,
            'cache': cache_spec,
            'intermediates': intermediates_spec,
        },
        split_rngs={
            'params': True,
            'dropout': True
        },
        length=num_layers,
        data_transform=transforms.inner_scan_spmd(scan_annotation,
                                                  self.scan_axis),
        axis_name='layers',
        axes_collections=('params', 'cache'),
    )
    return lyrf()

  @nn.nowrap
  def setup_output_logits(self):
    """Sets up output logits; this method provides flexiblity for subclasses."""
    # TODO: Re-merge with setup() once it's easier to Gin-configure
    # shared modules, and directly pass submodules (instead of using factories).
    if self.output_logits_factory:
      self.output_logits_factory: Callable[[], nn.Module]
      self.logits_dense = self.output_logits_factory()
    else:
      self.logits_dense = None

  @nn.nowrap
  def setup_meta_cls(self):
    if self.meta_cls_factory:
      self.meta_cls_factory: Callable[[], nn.Module]
      self.meta_cls = self.meta_cls_factory()
    else:
      self.meta_cls = None

  def embed_and_combine_inputs(
      self,
      decoder_input_tokens,
      decoder_positions=None,
      *,
      segment_ids: Optional[Array] = None,
      enable_dropout: bool = True,
      decode: bool = False,
  ):
    """Returns the combined embedded decoder inputs for further processing."""
    assert decoder_input_tokens.ndim == 2  # (batch, len)

    embedder_inputs = {'token_ids': decoder_input_tokens}
    if 'position_ids' in self.embedder.embedders:
      if decoder_positions is None:
        seq_length = decoder_input_tokens.shape[-1]
        decoder_positions = jnp.arange(seq_length)[None, :]
      embedder_inputs['position_ids'] = decoder_positions
    embedded_inputs = self.embedder(
        segment_ids=segment_ids, decode=decode, **embedder_inputs)

    embedded_inputs = self.input_dropout(
        embedded_inputs, deterministic=not enable_dropout)

    embedded_inputs = embedded_inputs.astype(self.dtype)
    return embedded_inputs

  def compute_logits(
      self,  # pytype: disable=annotation-type-mismatch  # jax-ndarray
      decoder_outputs: Array,
      logit_mask: Array = None,
      enable_dropout: bool = True,
  ) -> Array:
    # Post-process final decoder layer outputs.
    decoder_outputs = self.decoder_norm(decoder_outputs)
    decoder_outputs = self.output_dropout(
        decoder_outputs, deterministic=not enable_dropout)

    if logit_mask is not None:
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

    return logits

  def decode_from_continuous_inputs(
      self,
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
      return_prelogits: bool = False,
      **kwargs) -> Union[Array, Tuple[Array, Array]]:
    """Applies the decoder on the continuous (embedded) inputs."""
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
        **kwargs)
    if self.scan_layers:
      decoder_outputs = decoder_outputs[0]

    if self.sow_intermediates:
      self.sow('intermediates', 'pre_logits_layer', decoder_outputs)

    if self.return_all_logits and not return_prelogits and not decode:
      # Keep only part of the layers (first_exit, first_exit+exit_interval, ...)
      keep_inds = jnp.arange(
          self.first_exit, self.num_layers - 1, step=self.exit_interval)
      # And always keep the last layer.
      decoder_outputs = jnp.concatenate([
          decoder_outputs.take(keep_inds, 0),
          jnp.expand_dims(decoder_outputs[-1, ...], 0)
      ], 0)

      all_logits = self.compute_logits(
          decoder_outputs, jnp.resize(logit_mask, decoder_outputs.shape),
          enable_dropout)
      outputs = all_logits
    elif return_prelogits:
      outputs = decoder_outputs
    else:
      logits = self.compute_logits(decoder_outputs, logit_mask, enable_dropout)
      if self.sow_intermediates:
        self.sow('intermediates', 'logits', logits)
      outputs = logits

    if self.meta_cls is not None:
      meta_preds = self.meta_cls(decoder_outputs)
      outputs = (outputs, meta_preds)

    return outputs

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
               prefill_lengths: Optional[Array] = None,
               decoder_embedded_input: Optional[Array] = None,
               return_prelogits: bool = False,
               **kwargs):
    """Applies Transformer model on the inputs.

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
        cache, lengths are inferred from the mask if not provided.
      decoder_embedded_input: If given, it is passed directly to the decoder as
        the embedded_inputs instead of embedding the `decoder_input_tokens`. Can
        be useful for calling the decoder multiple times for the same token with
        different intervals of layers, passing the hidden-state between calls.
      return_prelogits: Returns the decoder output directly, before the logits
        computation.
      **kwargs: Optional keyword arguments to pass to
        decode_from_continuous_inputs.

    Returns:
      The decoder output logits for next token prediction.
    """
    if decoder_embedded_input is None:
      embedded_inputs = self.embed_and_combine_inputs(
          decoder_input_tokens,
          decoder_positions=decoder_positions,
          segment_ids=segment_ids,
          enable_dropout=enable_dropout,
          decode=decode,
      )
    else:
      embedded_inputs = decoder_embedded_input

    logit_mask = dense_attention.get_decoder_logit_mask(decoder_input_tokens,
                                                        embedded_inputs.dtype)

    logits = self.decode_from_continuous_inputs(
        embedded_inputs,
        encoder_outputs,
        decoder_positions=decoder_positions,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        logit_mask=logit_mask,
        enable_dropout=enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length,
        prefill=prefill,
        prefill_lengths=prefill_lengths,
        return_prelogits=return_prelogits,
        **kwargs)
    return logits


class EncoderDecoder(nn.Module, param_remapping.ParameterRemappable):
  """Transformer Model for sequence to sequence translation.

  Attributes:
    encoder_factory: A callable that returns the lower-level Encoder object. If
      shared_token_embedder_factory is non-None, then the result of it will be
      passed as the `shared_token_embedder` argument to `encoder_factory`.
    decoder_factory: A callable that returns the lower-level Decoder object. If
      shared_token_embedder_factory is non-None, then the result of it will be
      passed as the `shared_token_embedder` argument to `decoder_factory`.
    dtype: DType for encoder/decoder to cast embedded inputs, and for attention
      mask generation.
    scan_layers: whether to scan over layers.
    shared_token_embedder_factory: A callable that returns an embedder that can
      be shared between the encoder and decoder.
  """
  # Core components: encoder and decoder embedders and layers.
  encoder_factory: t5_architecture.MakeEncoderFn
  decoder_factory: t5_architecture.MakeDecoderFn

  # Configures behavior when the model is called. Many of these might eventually
  # be better as call parameters.
  dtype: DType = jnp.float32
  scan_layers: bool = False  # only used to pass this option to predict_fn.
  spmd_annotations: Any = None  # only used for scanned spmd layers

  shared_token_embedder_factory: Optional[Callable[[], embedding.Embed]] = None

  def setup(self):
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

  def compute_logits(
      self,  # pytype: disable=annotation-type-mismatch  # jax-ndarray
      decoder_outputs: Array,
      logit_mask: Array = None,
      enable_dropout: bool = True,
  ) -> Array:
    return self.decoder.compute_logits(
        decoder_outputs=decoder_outputs,
        logit_mask=logit_mask,
        enable_dropout=enable_dropout)

  def encode(self,
             encoder_input_tokens,
             encoder_segment_ids=None,
             encoder_positions=None,
             *,
             enable_dropout: bool = True) -> Array:
    """Applies Transformer encoder-branch on the inputs.

    Args:
      encoder_input_tokens: input data to the encoder.
      encoder_segment_ids: encoder input segmentation info for packed examples.
      encoder_positions: encoder input subsequence positions for packed
        examples.
      enable_dropout: Enables dropout if set to True.

    Returns:
      Encoded feature array from the transformer encoder.
    """
    # Make padding attention mask.
    encoder_mask = dense_attention.make_attention_mask(
        encoder_input_tokens > 0, encoder_input_tokens > 0, dtype=self.dtype)
    # Add segmentation block-diagonal attention mask if using segmented data.
    if encoder_segment_ids is not None:
      encoder_mask = dense_attention.combine_masks(
          encoder_mask,
          dense_attention.make_attention_mask(
              encoder_segment_ids,
              encoder_segment_ids,
              jnp.equal,
              dtype=self.dtype))

    return self.encoder(  # pytype: disable=attribute-error
        encoder_input_tokens,
        inputs_positions=encoder_positions,
        encoder_mask=encoder_mask,
        segment_ids=encoder_segment_ids,
        enable_dropout=enable_dropout)

  def decode(
      self,
      encoded,
      encoder_input_tokens,  # only needed for masks
      decoder_input_tokens,
      decoder_target_tokens,
      encoder_segment_ids=None,
      decoder_segment_ids=None,
      decoder_positions=None,
      *,
      enable_dropout: bool = True,
      decode: bool = False,
      # Args below were ported from decoder only code.
      max_decode_length: Optional[int] = None,
      prefill: bool = False,
      prefill_lengths: Optional[Array] = None,
      return_prelogits: bool = False,
      **kwargs):
    """Applies Transformer decoder-branch on encoded-input and target.

    Args:
      encoded: encoded input data from encoder.
      encoder_input_tokens: input to the encoder (only needed for masking).
      decoder_input_tokens: input token to the decoder.
      decoder_target_tokens: target token to the decoder.
      encoder_segment_ids: encoder segmentation info for packed examples.
      decoder_segment_ids: decoder segmentation info for packed examples.
      decoder_positions: decoder subsequence positions for packed examples.
      enable_dropout: Enables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache.
      max_decode_length: An optional integer specifying the maximum decoding
        length. Note that this is only used for defining the relative position
        embedding parameters.
      prefill: Whether to run a partial sequence to prefill the cache.
      prefill_lengths: The length of each partial sequence we are filling in the
        cache, lengths are inferred from the mask if not provided.
      return_prelogits: Returns the decoder output directly, without the logit
        computation.
      **kwargs: additional keyword arguments to pass to the decoder layers.

    Returns:
      Logits array from transformer decoder. If return_prelogits is True,
      returns the decoder state without computing the logits.
    """
    # Make padding attention masks.
    if decode:
      # Do not mask decoder attention based on targets padding at
      # decoding/inference time.
      decoder_mask = None
      encoder_decoder_mask = dense_attention.make_attention_mask(
          jnp.ones_like(decoder_target_tokens),
          encoder_input_tokens > 0,
          dtype=self.dtype)
    else:
      decoder_mask = dense_attention.make_decoder_mask(
          decoder_target_tokens=decoder_target_tokens,
          dtype=self.dtype,
          decoder_segment_ids=decoder_segment_ids)
      encoder_decoder_mask = dense_attention.make_attention_mask(
          decoder_target_tokens > 0, encoder_input_tokens > 0, dtype=self.dtype)

    # Add segmentation block-diagonal attention masks if using segmented data.
    if encoder_segment_ids is not None:
      if decode:
        raise ValueError(
            'During decoding, packing should not be used but '
            '`encoder_segment_ids` was passed to `Transformer.decode`.')

      encoder_decoder_mask = dense_attention.combine_masks(
          encoder_decoder_mask,
          dense_attention.make_attention_mask(
              decoder_segment_ids,
              encoder_segment_ids,
              jnp.equal,
              dtype=self.dtype))

    # When computing the logits, we don't need decoder_target_tokens, which is
    # needed for computing the loss.
    return self.decoder(
        encoded,
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        segment_ids=decoder_segment_ids,
        enable_dropout=enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length,
        prefill=prefill,
        prefill_lengths=prefill_lengths,
        return_prelogits=return_prelogits,
        **kwargs)

  @property
  def encoder_embedder(self) -> embedding.MultiEmbed:
    return self.encoder.embedder

  @property
  def decoder_embedder(self) -> embedding.MultiEmbed:
    return self.decoder.embedder

  def __call__(self,
               encoder_input_tokens,
               decoder_input_tokens,
               decoder_target_tokens,
               encoder_segment_ids=None,
               decoder_segment_ids=None,
               encoder_positions=None,
               decoder_positions=None,
               *,
               enable_dropout: bool = True,
               decode: bool = False,
               max_decode_length: Optional[int] = None,
               only_propagate_state: Optional[bool] = False):
    """Applies Transformer model on the inputs.

    This method requires both decoder_target_tokens and decoder_input_tokens,
    which is a shifted version of the former. For a packed dataset, it usually
    has additional processing applied. For example, the first element of each
    sequence has id 0 instead of the shifted EOS id from the previous sequence.

    Args:
      encoder_input_tokens: input data to the encoder.
      decoder_input_tokens: input token to the decoder.
      decoder_target_tokens: target token to the decoder.
      encoder_segment_ids: encoder segmentation info for packed examples.
      decoder_segment_ids: decoder segmentation info for packed examples.
      encoder_positions: encoder subsequence positions for packed examples.
      decoder_positions: decoder subsequence positions for packed examples.
      enable_dropout: Enables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache.
      max_decode_length: An optional integer specifying the maximum decoding
        length. Note that this is only used for defining the relative position
        embedding parameters.
      only_propagate_state: Specifies if only the state should be propagated
        from the last executed layer.

    Returns:
      Logits array from full transformer.
    """
    encoded = self.encode(
        encoder_input_tokens,
        encoder_segment_ids=encoder_segment_ids,
        encoder_positions=encoder_positions,
        enable_dropout=enable_dropout)

    return self.decode(
        encoded,
        encoder_input_tokens,  # Only used for masks.
        decoder_input_tokens,
        decoder_target_tokens,
        encoder_segment_ids=encoder_segment_ids,
        decoder_segment_ids=decoder_segment_ids,
        decoder_positions=decoder_positions,
        enable_dropout=enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length,
        only_propagate_state=only_propagate_state)
