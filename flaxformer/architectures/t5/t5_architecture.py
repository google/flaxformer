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

import enum
import inspect
from typing import Any, Callable, Optional, Protocol, Tuple, Union

from flax import linen as nn
import jax
import jax.numpy as jnp

from flaxformer import activation_partitioning
from flaxformer import transformer_common as common
from flaxformer.architectures.common import param_remapping
from flaxformer.components import embedding
from flaxformer.components import relative_position_biases
from flaxformer.components import rich_attention_position_scores
from flaxformer.components import transforms
from flaxformer.components.attention import dense_attention
from flaxformer.types import Array
from flaxformer.types import DType

# pylint: disable=not-callable
# pytype: disable=not-callable


class MakeEncoderLayerFn(Protocol):
  """Signature for functions that make an encoder layer."""

  def __call__(
      self, *,
      shared_relative_position_bias: Optional[nn.Module]) -> EncoderLayer:
    """Makes an encoder layer.

    Args:
      shared_relative_position_bias: Relative position bias shared for all
        layers within the encoder, which is the result of calling
        `shared_relative_position_bias_factory` at the top-level model. Due to
        Flax limitations, we need to pass this in as an attribute to modules.
        Please use this argument instead of using a Python closure.

    Returns:
      Encoder layer.
    """


class MakeDecoderLayerFn(Protocol):
  """Signature for functions that make a decoder layer."""

  def __call__(
      self, *,
      shared_relative_position_bias: Optional[nn.Module]) -> DecoderLayer:
    """Makes a decoder layer.

    Args:
      shared_relative_position_bias: Relative position bias shared for all
        layers within the decoder, which is the result of calling
        `shared_relative_position_bias_factory` at the top-level model. Due to
        Flax limitations, we need to pass this in as an attribute to modules.
        Please use this argument instead of using a Python closure.

    Returns:
      Decoder layer.
    """


class MakeEncoderFn(Protocol):
  """Signature for functions that will make a low-level Encoder."""

  def __call__(
      self,
      *,
      shared_token_embedder: Optional[embedding.Embedder[Array]] = None,
      spmd_annotations: Any = None,
  ) -> Encoder:
    """Makes a low-level Encoder instance.

    Args:
      shared_token_embedder: Shared token embedder instance, which should be
        passed to the returned module. If this is non-None, you should use it
        instead of providing your own token embedder.
      spmd_annotations: Optional SPMD annotations for scanned layers.

    Returns:
      Encoder instance.
    """


class MakeDecoderFn(Protocol):
  """Signature for functions that will make a low-level Decoder."""

  def __call__(
      self,
      *,
      shared_token_embedder: Optional[embedding.Embedder[Array]] = None,
      spmd_annotations: Any = None,
  ) -> Decoder:
    """Makes a low-level Decoder instance.

    Args:
      shared_token_embedder: Shared token embedder instance, which should be
        passed to the returned module. If this is non-None, you should use it
        instead of providing your own token embedder.
      spmd_annotations: Optional SPMD annotations for scanned layers.

    Returns:
      Decoder instance.
    """


@enum.unique
class LayerRemat(enum.Enum):
  """How to apply per-layer jax.remat for recomputation in the backward pass.

  Attributes:
    NONE: No use of jax.remat.
    LEGACY: Reverts to prior behavior for compatibility with existing configs,
      i.e., use FULL when scanning over layers and NONE otherwise.
    FULL: Recompute the whole layer in backprop.
    MINIMAL: Recompute only non-matmul ops in backprop.
  """

  NONE = 'none'
  LEGACY = 'legacy'
  FULL = 'full'
  MINIMAL = 'minimal'


_LayerRematOrStr = Union[LayerRemat, str]


def maybe_remat(
    lyrf: Callable[[], nn.Module],
    layer_remat: _LayerRematOrStr,
    scan_layers: bool,
    static_argnums: Tuple[int, ...],
) -> Callable[[], nn.Module]:
  """Maybe apply jax.remat with the indicated policy to a layer factory.

  Args:
    lyrf: Encoder or decoder layer factory.
    layer_remat: Config for per-layer remat.
    scan_layers: Whether to use jax.lax.scan for the stack of layers.
    static_argnums: The static_argnums to use for the jax.remat call.

  Returns:
    Potentially remat-wrapped layer factory.
  """
  # TODO: remove this conversion after all callers use the enum
  layer_remat = LayerRemat(layer_remat)

  if layer_remat == LayerRemat.LEGACY:
    layer_remat = LayerRemat.FULL if scan_layers else LayerRemat.NONE
  if layer_remat == LayerRemat.NONE:
    return lyrf

  if layer_remat == LayerRemat.FULL:
    remat_policy = None
  elif layer_remat == LayerRemat.MINIMAL:
    remat_policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
  else:
    raise ValueError(f'Unknown LayerRemat value: {layer_remat}')
  return transforms.factory_remat(
      lyrf,
      concrete=False,
      prevent_cse=False,
      static_argnums=static_argnums,
      policy=remat_policy,
  )


class EncoderLayer(nn.Module, param_remapping.ParameterRemappable):
  """Transformer encoder layer.

  Attributes:
    attention: The attention module.
    mlp: The MLP module, applied after attention.
    dropout_factory:  A callable that returns a new dropout instance. This is
      applied after the attention module.
    layer_norm_factory:  A callable that returns a new layer norm. This is
      applied before the attention module and before the MLP.
    relative_position_bias_factory:  A callable that returns relative position
      bias instances. This should only be used for per-layer relative position
      biases; please use `shared_relative_position_bias` if they are shared
      among layers.
    shared_relative_position_bias: Shared relative position bias module, usually
      owned by the Encoder.
    activation_partitioning_dims: When set to 2, partitions intermediate
      variables containing the input and output of the encoder layer.
    parallel: whether to call attention and mlp in parallel
    sow_intermediates: whether to track intermediates using Module.sow.
    scanned: whether this layer is being scanned over.
  """
  attention: nn.Module
  mlp: nn.Module
  dropout_factory: Callable[[], nn.Module]
  layer_norm_factory: Callable[[], nn.Module]
  relative_position_bias_factory: Optional[Callable[[], nn.Module]] = None
  shared_relative_position_bias: Optional[nn.Module] = None
  activation_partitioning_dims: int = 1
  parallel: bool = False
  sow_intermediates: bool = False
  scanned: bool = False

  def setup(self):
    if (self.relative_position_bias_factory is not None and
        self.shared_relative_position_bias is not None):
      raise ValueError(
          'Please set at most one of relative_position_bias_factory and'
          ' shared_relative_position_bias. (They can both be None however, e.g.'
          ' for absolute position embeds.)'
      )
    self.relpos_bias = (
        self.relative_position_bias_factory()
        if self.relative_position_bias_factory is not None else
        self.shared_relative_position_bias)

    if self.parallel:
      self.layer_norm = self.layer_norm_factory()
      self.dropout = self.dropout_factory()
    else:
      self.pre_attention_layer_norm = self.layer_norm_factory()
      self.pre_mlp_layer_norm = self.layer_norm_factory()
      self.post_attention_dropout = self.dropout_factory()
      self.post_mlp_dropout = self.dropout_factory()

  def get_bias(self, layer_input: Array) -> Optional[Array]:
    if not self.relpos_bias:
      return None

    if isinstance(
        self.relpos_bias, relative_position_biases.RelativePositionBiases
    ):
      encoder_bias = self.relpos_bias(
          layer_input.shape[-2], layer_input.shape[-2], bidirectional=True
      )
    elif isinstance(
        self.relpos_bias, rich_attention_position_scores.RichAttentionApi
    ):
      encoder_bias = self.relpos_bias(
          layer_input, layer_input, bidirectional=True
      )
    else:
      raise TypeError(
          f'{type(self.relpos_bias)} is not a supported relative position '
          f'bias factory.\nInstance value: {self.relpos_bias}'
      )
    return encoder_bias

  def __call__(self,
               inputs: Array,
               encoder_mask: Optional[Array] = None,
               *,
               logit_mask: Optional[Array] = None,
               enable_dropout: bool = True):
    """Applies a single T5 encoder layer.

    Args:
      inputs: input data [batch, length, emb_dim].
      encoder_mask: encoder self-attention mask.
      logit_mask: encoder logits mask.
      enable_dropout: Enables dropout if set to True.

    Returns:
      output after transformer encoder block.
    """
    layer_input = inputs
    del inputs
    # Shared relative position embedding attention biases.

    assert layer_input.ndim == 3
    layer_input = activation_partitioning.with_sharding_migration(
        layer_input,
        self.activation_partitioning_dims,
        logical_axis_names=('batch', 'length', 'embed'))
    if self.parallel:
      x = self.layer_norm(layer_input)
      x = activation_partitioning.with_sharding_migration(
          x,
          self.activation_partitioning_dims,
          logical_axis_names=('batch', 'length', 'embed'))

      encoder_bias = self.get_bias(layer_input=x)

      y = (
          self.attention(
              x, x, encoder_mask, encoder_bias, enable_dropout=enable_dropout) +
          self.mlp(x, enable_dropout=enable_dropout))
      y *= 2**-0.5
      y = layer_input + self.dropout(y, deterministic=not enable_dropout)

    else:
      # Attention block.
      x = self.pre_attention_layer_norm(layer_input)
      x = activation_partitioning.with_sharding_migration(
          x,
          self.activation_partitioning_dims,
          logical_axis_names=('batch', 'length', 'embed'))

      if logit_mask is not None:
        x = logit_mask * x

      encoder_bias = self.get_bias(layer_input=x)

      # The shape should be maintained for the residual connection.
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      x = self.attention(
          x, x, encoder_mask, encoder_bias, enable_dropout=enable_dropout)
      x = layer_input + self.post_attention_dropout(
          x, deterministic=not enable_dropout)
      x = activation_partitioning.with_sharding_migration(
          x,
          self.activation_partitioning_dims,
          logical_axis_names=('batch', 'length', 'embed'))

      # MLP block.
      y = self.pre_mlp_layer_norm(x)
      y = activation_partitioning.with_sharding_migration(
          y,
          self.activation_partitioning_dims,
          logical_axis_names=('batch', 'length', 'embed'))

      if logit_mask is not None:
        y = logit_mask * y

      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      y = self.mlp(y, enable_dropout=enable_dropout)
      y = x + self.post_mlp_dropout(y, deterministic=not enable_dropout)

    y = activation_partitioning.with_sharding_migration(
        y,
        self.activation_partitioning_dims,
        logical_axis_names=('batch', 'length', 'embed'))
    if self.sow_intermediates:
      self.sow('intermediates', 'activations', y)

    # scan expects functions to have a signature: fn(carry, in) --> carry, out
    # TODO: automate this detail.
    if self.scanned:
      return y, None
    else:
      return y


class DecoderLayer(nn.Module, param_remapping.ParameterRemappable):
  """Transformer encoder-decoder layer.

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

  def setup(self):
    if (self.relative_position_bias_factory is not None and
        self.shared_relative_position_bias is not None):
      raise ValueError(
          'Please set at most one of relative_position_bias_factory and'
          ' shared_relative_position_bias. (They can both be None however, e.g.'
          ' for absolute position embeds.)'
      )
    self.relpos_bias = (
        self.relative_position_bias_factory()
        if self.relative_position_bias_factory is not None else
        self.shared_relative_position_bias)

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
    if not self.relpos_bias:
      return None, None

    if isinstance(
        self.relpos_bias, relative_position_biases.RelativeAttentionAPI
    ):
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
          relpos_length, relpos_length, False, decode=decode
      )
      encoder_decoder_bias = None

    elif isinstance(
        self.relpos_bias, rich_attention_position_scores.RichAttentionApi
    ):
      decoder_bias = self.relpos_bias(
          layer_input,
          layer_input,
          bidirectional=False,
          is_cross_attention=False,
      )
      encoder_decoder_bias = self.relpos_bias(
          layer_input, encoded, bidirectional=False, is_cross_attention=True
      )
    else:
      raise TypeError(
          f'{type(self.relpos_bias)} is not a supported relative position '
          f'bias factory.\nInstance value: {self.relpos_bias}'
      )
    return decoder_bias, encoder_decoder_bias

  def _create_residuals_and_queries(self, layer_input: Array, x: Array,
                                    logit_mask: Array,
                                    **kwargs) -> Tuple[Array, Array, Array]:
    """Slice layer inputs to get versions to use as queries."""
    # This is a no-op unless overridden by a subclass.
    return layer_input, x, logit_mask

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
               **kwargs):
    """Applies a single T5 decoder layer.

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
      **kwargs: Remaining keyword arguments. Passed to
        _create_residuals_and_queries.

    Returns:
      output after transformer encoder-decoder block.
    """
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

      # Normally a no-op unless overridden by a subclass.
      layer_input_residual, x_queries, logit_mask_queries = (
          self._create_residuals_and_queries(layer_input, x, logit_mask,
                                             **kwargs))

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
    # TODO: automate this detail.
    if self.scanned:
      return z, None
    else:
      return z


class Encoder(nn.Module, param_remapping.ParameterRemappable):
  """A stack of encoder layers.

  Attributes:
    layer_factory: A callable that returns an EncoderLayer.
    input_dropout_factory: A callable that returns the dropout to apply to the
      input.
    output_dropout_factory: A callable that returns the dropout to apply to the
      output. Perhaps for legacy rather than essential reasons, the broadcasting
      pattern is sometimes different from input_dropout_factory().
    layer_norm_factory: A callable that returns a layer norm.
    num_layers: Number of layers to generate.
    dtype: DType to cast the embedded inputs.
    layer_remat: whether and how to apply jax.remat to each layer to perform
      recomputation in the backward pass. See documentation for LayerRemat enum.
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
    scan_axis: axis over which to do scan over layers.
    sow_intermediates: whether to track intermediates using Module.sow.
    capture_gradients: whether to track input gradients using a variable in the
      `grads` collection. This captures the gradient of the (combined) embedded
      inputs, i.e. the input to the first encoder layer.
  """
  layer_factory: MakeEncoderLayerFn
  input_dropout_factory: Callable[[], nn.Module]
  output_dropout_factory: Callable[[], nn.Module]
  layer_norm_factory: Callable[[], nn.Module]
  num_layers: int
  dtype: DType = jnp.float32
  layer_remat: _LayerRematOrStr = LayerRemat.LEGACY
  scan_layers: bool = False
  spmd_annotations: Any = None
  shared_relative_position_bias_factory: Optional[Callable[[],
                                                           nn.Module]] = None
  scan_axis: int = 1

  # Embedders: Either a token_embedder_factory factory or shared token embedder
  # must be provided. The position embedder is optional and provided when
  # absolute position embeddings are desired.
  token_embedder_factory: Optional[Callable[[],
                                            embedding.Embedder[Array]]] = None
  shared_token_embedder: Optional[embedding.Embedder[Array]] = None
  position_embedder_factory: Optional[Callable[
      [], embedding.Embedder[Array]]] = None
  sow_intermediates: bool = False
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
      self.token_embedder = self.token_embedder_factory()
      embedders = {'token_ids': self.token_embedder}
    if self.position_embedder_factory is not None:
      self.position_embedder = self.position_embedder_factory()
      embedders['position_ids'] = self.position_embedder
    self.embedder = embedding.MultiEmbed(
        embedders,
        sow_intermediates=self.sow_intermediates,
        capture_gradients=self.capture_gradients)

    self.input_dropout = self.input_dropout_factory()

    if self.scan_layers and self.shared_relative_position_bias_factory:
      raise ValueError("Scanned layer mode doesn't support shared relative "
                       'position biases.')
    self.relpos_bias = (
        self.shared_relative_position_bias_factory()
        if self.shared_relative_position_bias_factory is not None else None)

    lyrf = lambda: self.layer_factory(  # pylint: disable=g-long-lambda
        shared_relative_position_bias=self.relpos_bias)
    lyrf = maybe_remat(
        lyrf, self.layer_remat, self.scan_layers, static_argnums=(3,))
    if not self.scan_layers:
      self.layers = [lyrf() for _ in range(self.num_layers)]
      self.encoder = common.TransparentLayerSequence(self.layers)
    else:
      self.encoder = self._construct_scanned_encoder(lyrf, self.num_layers)

    self.encoder_norm = self.layer_norm_factory()
    self.output_dropout = self.output_dropout_factory()

  def _construct_scanned_encoder(self, lyrf: Callable[[], nn.Module],
                                 num_layers: int) -> nn.Module:
    """Constructs encoder from layer factory using scan."""
    initializing = self.is_mutable_collection('params')
    # We scan the parameters along axis scan_axis (default=1)
    # as an XLA layout optimization.
    params_spec = self.scan_axis if initializing else transforms.ScanIn(
        self.scan_axis)
    cache_spec = 0
    intermediates_spec = 2  # Stacks intermediate layer outputs in dimension 2.
    scan_annotation = (
        self.spmd_annotations['encoder']
        if self.spmd_annotations is not None else None)
    lyrf = transforms.factory_scan(
        lyrf,
        in_axes=(nn.broadcast, nn.broadcast, nn.broadcast),
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
        axes_collections=('params', 'cache'),
    )
    return lyrf()

  def embed_and_combine_inputs(self,
                               inputs,
                               inputs_positions=None,
                               *,
                               segment_ids: Optional[Array] = None,
                               enable_dropout: bool = True):
    """Returns the combined embedded inputs for further encoding."""
    assert inputs.ndim == 2  # (batch, len)

    embedder_inputs = {'token_ids': inputs}
    if 'position_ids' in self.embedder.embedders:
      if inputs_positions is None:
        seq_length = inputs.shape[-1]
        inputs_positions = jnp.arange(seq_length)[None, :]
      embedder_inputs['position_ids'] = inputs_positions
    # TODO: Pass `deterministic=not enable_dropout`?
    embedded_inputs = self.embedder(segment_ids=segment_ids, **embedder_inputs)

    embedded_inputs = self.input_dropout(
        embedded_inputs, deterministic=not enable_dropout)

    # TODO: Revert this cast or move to embedder.
    embedded_inputs = embedded_inputs.astype(self.dtype)
    return embedded_inputs

  def encode_from_continuous_inputs(self,
                                    inputs,
                                    encoder_mask=None,
                                    logit_mask=None,
                                    *,
                                    enable_dropout: bool = True):
    """Applies all the layers starting from the continuous (embedded) inputs."""
    # Apply all encoder layers. Because of residual connection, the width of the
    # network is kept at `cfg.emb_dim` throughout.
    encoder_outputs = self.encoder(
        inputs,
        encoder_mask,
        logit_mask=logit_mask,
        enable_dropout=enable_dropout)
    if self.scan_layers:
      encoder_outputs = encoder_outputs[0]

    # Post-process the outputs of the final encoder layer.
    # TODO: We could do this in the common encoder.
    encoder_outputs = self.encoder_norm(encoder_outputs)
    encoder_outputs = self.output_dropout(
        encoder_outputs, deterministic=not enable_dropout)

    if logit_mask is not None:
      encoder_outputs = logit_mask * encoder_outputs
    return encoder_outputs

  def __call__(self,
               inputs,
               inputs_positions=None,
               encoder_mask=None,
               *,
               segment_ids: Optional[Array] = None,
               enable_dropout: bool = True):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      inputs_positions: input subsequence positions for packed examples.
      encoder_mask: encoder self-attention mask.
      segment_ids: Input segmentation info for packed examples.
      enable_dropout: Enables dropout if set to True.

    Returns:
      output of a transformer encoder.
    """
    if self.sow_intermediates:
      self.sow('intermediates', 'input_tokens_ids', inputs)
    embedded_inputs = self.embed_and_combine_inputs(
        inputs,
        inputs_positions=inputs_positions,
        segment_ids=segment_ids,
        enable_dropout=enable_dropout)
    logit_mask = jnp.expand_dims(
        jnp.array((inputs > 0), dtype=embedded_inputs.dtype), axis=-1)
    encoder_outputs = self.encode_from_continuous_inputs(
        embedded_inputs,
        encoder_mask=encoder_mask,
        logit_mask=logit_mask,
        enable_dropout=enable_dropout)
    if self.sow_intermediates:
      self.sow('intermediates', 'final_encoder_outputs', encoder_outputs)
    return encoder_outputs


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
      recomputation in the backward pass. See documentation for LayerRemat enum.
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
    sow_intermediates: whether to track intermediates using Module.sow.
    scan_axis: axis over which to do scan over layers.
    capture_gradients: whether to track input gradients using a variable in the
      `grads` collection. This captures the gradient of the (combined) embedded
      inputs, i.e. the input to the first encoder layer.
  """
  layer_factory: MakeDecoderLayerFn
  dropout_factory: Callable[[], nn.Module]
  layer_norm_factory: Callable[[], nn.Module]
  num_layers: int
  dtype: DType = jnp.float32
  layer_remat: _LayerRematOrStr = LayerRemat.LEGACY
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
  shared_token_embedder: Optional[embedding.Embedder[Array]] = None
  position_embedder_factory: Optional[Callable[
      [], embedding.Embedder[Array]]] = None

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
      self.token_embedder = self.token_embedder_factory()
      embedders = {'token_ids': self.token_embedder}
    if self.position_embedder_factory is not None:
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

  def _setup_layer_sequence(self):
    lyrf = lambda: self.layer_factory(  # pylint: disable=g-long-lambda
        shared_relative_position_bias=self.relpos_bias)
    lyrf = maybe_remat(
        lyrf,
        self.layer_remat,
        self.scan_layers,
        static_argnums=(5, 6, 7, 8, 9))
    if not self.scan_layers:
      self.layers = [lyrf() for _ in range(self.num_layers)]
      return common.TransparentLayerSequence(self.layers)
    else:
      return self._construct_scanned_decoder(lyrf, self.num_layers)

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
      # TODO: Consider renaming to "output_logits".
      self.output_logits_factory: Callable[[], nn.Module]
      self.logits_dense = self.output_logits_factory()
    else:
      self.logits_dense = None

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
    # TODO: Pass `deterministic=not enable_dropout`?
    embedded_inputs = self.embedder(
        segment_ids=segment_ids, decode=decode, **embedder_inputs)

    embedded_inputs = self.input_dropout(
        embedded_inputs, deterministic=not enable_dropout)

    # TODO: Revert this cast or move to embedder.
    embedded_inputs = embedded_inputs.astype(self.dtype)
    return embedded_inputs

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
  ):
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
        prefill_lengths=prefill_lengths)
    if self.scan_layers:
      decoder_outputs = decoder_outputs[0]

    # Post-process final decoder layer outputs.
    # TODO: We could do this in the common decoder.
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
      #
      # TODO: Module subclass API if we want to keep using this.
      logits = self.embedder.embedders['token_ids'].attend(decoder_outputs)  # pytype: disable=attribute-error
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(decoder_outputs.shape[-1])

    if self.sow_intermediates:
      self.sow('intermediates', 'logits', logits)
    return logits

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
               **kwargs):
    """Applies Transformer model on the inputs.

    TODO: For consistency it would be better to flip the order of the
    first two positional arguments here.

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
      **kwargs: Optional keyword arguments to pass to
        decode_from_continuous_inputs.

    Returns:
      The decoder output logits for next token prediction.
    """
    embedded_inputs = self.embed_and_combine_inputs(
        decoder_input_tokens,
        decoder_positions=decoder_positions,
        segment_ids=segment_ids,
        enable_dropout=enable_dropout,
        decode=decode,
    )
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
  encoder_factory: MakeEncoderFn
  decoder_factory: MakeDecoderFn

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

  def _make_padding_attention_mask(self, query_tokens: Array,
                                   key_tokens: Array) -> Array:
    return dense_attention.make_attention_mask(
        query_tokens > 0, key_tokens > 0, dtype=self.dtype)

  def encode(self,
             encoder_input_tokens,
             encoder_segment_ids=None,
             encoder_positions=None,
             *,
             enable_dropout: bool = True):
    """Applies Transformer encoder-branch on the inputs.

    Args:
      encoder_input_tokens: input data to the encoder.
      encoder_segment_ids: encoder input segmentation info for packed examples.
      encoder_positions: encoder input subsequence positions for packed
        examples.
      enable_dropout: Enables dropout if set to True.

    Returns:
      encoded feature array from the transformer encoder.
    """
    # Make padding attention mask.
    encoder_mask = self._make_padding_attention_mask(encoder_input_tokens,
                                                     encoder_input_tokens)
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
      prefill_lengths: Optional[Array] = None):
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

    Returns:
      logits array from transformer decoder.
    """
    # Make padding attention masks.
    if decode:
      # Do not mask decoder attention based on targets padding at
      # decoding/inference time.
      decoder_mask = None
      encoder_decoder_mask = self._make_padding_attention_mask(
          jnp.ones_like(decoder_target_tokens), encoder_input_tokens)
    else:
      decoder_mask = dense_attention.make_decoder_mask(
          decoder_target_tokens=decoder_target_tokens,
          dtype=self.dtype,
          decoder_segment_ids=decoder_segment_ids)
      encoder_decoder_mask = self._make_padding_attention_mask(
          decoder_target_tokens, encoder_input_tokens)

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
        prefill_lengths=prefill_lengths)

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
               max_decode_length: Optional[int] = None):
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

    Returns:
      logits array from full transformer.
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
        max_decode_length=max_decode_length)


class DecoderOnly(nn.Module, param_remapping.ParameterRemappable):
  """Decoder-only model.

  This model sets up the relevant masking and uses Decoder to do the heavy
  lifting.

  Attributes:
    decoder_factory: Factory which will make the lower-level Decoder object. In
      the DecoderOnly usage, it will always be called with
      `shared_token_embedder` as None.
    dtype: DType for encoder/decoder to cast embedded inputs, and for attention
      mask generation.
  """
  # Core sub-component.
  decoder_factory: MakeDecoderFn

  # Only used to pass this option to predict_fn.
  scan_layers: bool = False

  # Configures behavior when the model is called. Many of these might eventually
  # be better as call parameters.
  dtype: DType = jnp.float32

  def setup(self):
    self.decoder = self.decoder_factory(shared_token_embedder=None)

  def __call__(
      self,
      decoder_input_tokens: Array,
      decoder_target_tokens: Optional[Array],
      decoder_segment_ids: Optional[Array] = None,
      decoder_positions: Optional[Array] = None,
      decoder_causal_attention: Optional[Array] = None,
      *,
      enable_dropout: bool = True,
      decode: bool = False,
      max_decode_length: Optional[int] = None,
      prefill: bool = False,
      prefill_lengths: Optional[Array] = None,
      **kwargs,
  ):
    """Applies LanguageModel on the inputs.

    This method requires both decoder_target_tokens and decoder_input_tokens,
    which is typically a shifted version of the former. For a packed dataset, it
    usually has additional processing applied. For example, the first element of
    each sequence has id 0 instead of the shifted EOS id from the previous
    sequence.

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
      **kwargs: Additional keyword arguments to pass on to the decoder. This may
        include `decoder_attention_mask`, which overrides the decoder attention
        mask. If specified, this must be broadcastable to `[batch, head,
        target_length, target_length]`. Meanwhile, `decoder_target_tokens` will
        be ignored and `decoder_causal_attention` should not be set.

    Returns:
      logits array from LanguageModel.
    """
    if decode and prefill:
      raise ValueError('Only one of `decode` and `prefill` can be set. Use '
                       '`prefill` to pre-populate the cache for Prefix LMs '
                       'before using `decode`')
    if decode:
      decoder_mask = None
    else:
      if 'decoder_attention_mask' in kwargs:
        decoder_attention_mask = kwargs.pop('decoder_attention_mask')
        if decoder_causal_attention is not None:
          raise ValueError(
              'Only one of `decoder_causal_attention` and '
              '`decoder_attention_mask` can be set.'
          )
        decoder_mask = jnp.asarray(decoder_attention_mask, dtype=self.dtype)
      else:
        decoder_mask = dense_attention.make_decoder_mask(
            decoder_target_tokens=decoder_target_tokens,
            dtype=self.dtype,
            decoder_causal_attention=decoder_causal_attention,
            decoder_segment_ids=decoder_segment_ids,
        )

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
        **kwargs)
