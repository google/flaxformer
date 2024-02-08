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

"""This file contains "architecture" classes for long input T5 models.

The classes are similar to the original T5 classes but allow custom long
attention classes in the encoder that don't depend on quadratic attention
masks or relative position biases.  Currently the decoder side just uses
the original T5 classes (assuming outputs are not that long).
"""

import inspect
from typing import Callable, Optional, Any, Tuple

from flax import linen as nn
import jax.numpy as jnp
from typing_extensions import Protocol
from flaxformer import activation_partitioning
from flaxformer import transformer_common as common
from flaxformer.architectures.common import param_remapping
from flaxformer.architectures.t5 import t5_architecture
from flaxformer.components import embedding
from flaxformer.components import transforms
from flaxformer.components.attention import dense_attention
from flaxformer.types import Array
from flaxformer.types import DType

# Type Stubs
MakeDecoderFn = t5_architecture.MakeDecoderFn

# pylint: disable=not-callable
# pytype: disable=not-callable


class MakeLongEncoderLayerFn(Protocol):
  """Signature for functions that make a long input encoder layer."""

  def __call__(
      self,
      *,
      shared_relpos_bias: Optional[nn.Module],
      shared_side_relpos_bias: Optional[nn.Module] = None
  ) -> 'LongEncoderLayer':
    """Makes a long input encoder layer.

    Args:
      shared_relpos_bias: Relative position bias shared for all layers within
        the encoder, which is the result of calling `shared_relpos_bias_factory`
        at the top-level model. Due to Flax limitations, we need to pass this in
        as an attribute to modules. Please use this argument instead of using a
        Python closure.
      shared_side_relpos_bias: Side relative position bias shared for all layers
        within the encoder, which is the result of calling
        `shared_side_relpos_bias_factory` at the top-level model. Most
        `LongSelfAttention` implementations do not use this, and instances of
        `MakeLongEncoderLayerFn` do not need to define this parameter if it's
        not used.

    Returns:
      LongEncoderLayer instance.
    """
    pass


class MakeLongEncoderFn(Protocol):
  """Signature for functions that will make a low-level LongEncoder."""

  def __call__(
      self,
      *,
      shared_token_embedder: Optional[embedding.Embed] = None,
      spmd_annotations: Any = None,
  ) -> 'LongEncoder':
    """Makes a low-level LongEncoder instance.

    Args:
      shared_token_embedder: Shared token embedder instance, which should be
        passed to the returned module. If this is non-None, you should use it
        instead of providing your own token embedder.
      spmd_annotations: Optional SPMD annotations for scanned layers.

    Returns:
      LongEncoder instance.
    """
    pass


class MakeLongSelfAttentionFn(Protocol):
  """Signature for functions that will make a LongSelfAttention module.

  See `long_attention.py` for the definition of `LongSelfAttention` and some
  particular implementations.
  """

  def __call__(
      self,
      *,
      relpos_bias: Optional[nn.Module] = None,
      side_relpos_bias: Optional[nn.Module] = None,
  ) -> nn.Module:
    """Makes a low-level LongSelfAttention instance.

    Args:
      relpos_bias: General relative position bias module, which should be passed
        to the returned module. If this is non-None, you should use it instead
        of providing your own module.
      side_relpos_bias: Side general relative position bias module, which should
        be passed to the returned module.  Most `LongSelfAttention`
        implementations do not use this, and instances of
        `MakeLongSelfAttentionFn` do not need to define this parameter if it's
        not used.

    Returns:
      LongSelfAttention instance.
    """
    pass


class LongEncoderLayer(nn.Module, param_remapping.ParameterRemappable):
  """Transformer long input encoder layer.

  Attributes:
    attention_factory: Factory for making the long attention module.
    mlp: The MLP module, applied after attention.
    dropout_factory:  A callable that returns a new dropout instance. This is
      applied after the attention module.
    layer_norm_factory:  A callable that returns a new layer norm. This is
      applied before the attention module and before the MLP.
    relpos_bias_factory:  A callable that returns general relative position bias
      instances. This should only be used for per-layer relative position
      biases; please use `shared_relpos_bias` if they are shared among layers.
    shared_relpos_bias: Shared general relative position bias module, usually
      owned by the Encoder.
    side_relpos_bias_factory:  A callable that returns general relative position
      bias instances like `relpos_bias_factory`.  Most `LongSelfAttention`
      implementations do not use this, so it can be simply left as None when
      unused.
    shared_side_relpos_bias: Optional shared side relative position bias module,
      usually owned by the Encoder.  Most `LongSelfAttention` implementations do
      not use this, so it can be simply left as None when unused.  This should
      not be used if `side_relpos_bias_factory` is used instead.
    activation_partitioning_dims: When set to 2, partitions intermediate
      variables containing the input and output of the encoder layer.
    parallel: whether to call attention and mlp in parallel
    sow_intermediates: whether to track intermediates using Module.sow.
    scanned: whether this layer is being scanned over.
    use_logit_mask: whether the input mask is used to zero out the padding
      representations.
  """
  attention_factory: MakeLongSelfAttentionFn
  mlp: nn.Module
  dropout_factory: Callable[[], nn.Module]
  layer_norm_factory: Callable[[], nn.Module]
  relpos_bias_factory: Optional[Callable[[], nn.Module]] = None
  shared_relpos_bias: Optional[nn.Module] = None
  side_relpos_bias_factory: Optional[Callable[[], nn.Module]] = None
  shared_side_relpos_bias: Optional[nn.Module] = None
  activation_partitioning_dims: int = 1
  parallel: bool = False
  sow_intermediates: bool = False
  scanned: bool = False
  use_logit_mask: bool = True

  def setup(self):
    if (self.relpos_bias_factory is not None and
        self.shared_relpos_bias is not None):
      raise ValueError(
          'Please set at most one of relpos_bias_factory and shared_relpos_bias. '
          '(They can both be None however, e.g. for absolute position embeds.)')
    self.relpos_bias = (
        self.relpos_bias_factory()
        if self.relpos_bias_factory is not None else self.shared_relpos_bias)
    if (self.side_relpos_bias_factory is not None and
        self.shared_side_relpos_bias is not None):
      raise ValueError(
          'Please set at most one of side_relpos_bias_factory and '
          'shared_side_relpos_bias. (They can both be None however.)')
    self.side_relpos_bias = (
        self.side_relpos_bias_factory() if self.side_relpos_bias_factory
        is not None else self.shared_side_relpos_bias)

    attention_factory_kwargs = dict(relpos_bias=self.relpos_bias)
    if self.side_relpos_bias is not None:
      attention_factory_kwargs['side_relpos_bias'] = self.side_relpos_bias
    self.attention = self.attention_factory(**attention_factory_kwargs)  # pytype: disable=wrong-keyword-args  # dict-kwargs

    if self.parallel:
      self.layer_norm = self.layer_norm_factory()
      self.dropout = self.dropout_factory()
    else:
      self.pre_attention_layer_norm = self.layer_norm_factory()
      self.pre_mlp_layer_norm = self.layer_norm_factory()
      self.post_attention_dropout = self.dropout_factory()
      self.post_mlp_dropout = self.dropout_factory()

  def __call__(self,
               inputs: Array,
               inputs_mask: Array,
               *,
               inputs_positions: Optional[Array] = None,
               inputs_segment_ids: Optional[Array] = None,
               enable_dropout: bool = True):
    """Applies a single LongT5 encoder layer.

    Args:
      inputs: input data [batch, length, emb_dim].
      inputs_mask: bool array with same shape as `inputs` indicating True for
        non-padding tokens and False for padding.
      inputs_positions: input subsequence positions for packed examples.
      inputs_segment_ids: input segmentation info for packed examples.
      enable_dropout: Enables dropout if set to True.

    Returns:
      output after transformer encoder block.
    """
    layer_input = inputs
    del inputs

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
      y = (
          self.attention(
              x,
              inputs_mask,
              positions=inputs_positions,
              segment_ids=inputs_segment_ids,
              enable_dropout=enable_dropout) +
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
      # Convert padding representations to zero vectors. Note inputs_mask
      # and x are normally expected to have the same [batch, length] shape.
      # However, if this isn't the case, set use_logit_mask to False to
      # avoid a shape incompatibility error.
      if self.use_logit_mask:
        logit_mask = inputs_mask.astype(x.dtype)[:, :, jnp.newaxis]
        x = x * logit_mask
      # The shape should be maintained for the residual connection.
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      x = self.attention(
          x,
          inputs_mask,
          positions=inputs_positions,
          segment_ids=inputs_segment_ids,
          enable_dropout=enable_dropout)
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
      # Convert padding representations to zero vectors
      if self.use_logit_mask:
        y = y * logit_mask.astype(y.dtype)
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


class LongEncoder(nn.Module, param_remapping.ParameterRemappable):
  """A stack of long input encoder layers.

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
      recomputation in the backward pass. Supported values are 'none', for no
      use of jax.remat; 'minimal', for a policy that recomputes only non-matmul
      operations (typically optimal); and 'full', for full recomputation of each
      layer. The (legacy) default is to use 'none' when `scan_layers=False` and
      and 'full' when `scan_layers=True`.
    scan_layers: whether to scan over layers.
    spmd_annotations: spmd annotations needed for scanned layers.
    shared_relpos_bias_factory: A callable that returns a relative position bias
      instance which will be shared for all encoder layers. Only set this if
      using shared relative position biases.
    shared_side_relpos_bias_factory: A callable that returns a relative position
      bias instance for side inputs which will be shared for all encoder layers.
      Only set this if using shared side relative position biases.  Most
      `LongSelfAttention` implementations do not use this, and it can be safely
      left as `None`.
    token_embedder_factory: A callable that returns a token embedder. Please
      provide either this or `shared_token_embedder`.
    shared_token_embedder: A callable that returns a token embedder shared
      between both encoder and decoder.
    position_embedder_factory: A callable that returns an absolute position
      embedder. Only provide this if you want absolute position embeddings.
  """
  layer_factory: MakeLongEncoderLayerFn
  input_dropout_factory: Callable[[], nn.Module]
  output_dropout_factory: Callable[[], nn.Module]
  layer_norm_factory: Callable[[], nn.Module]
  num_layers: int
  dtype: DType = jnp.float32
  layer_remat: str = 'legacy'
  scan_layers: bool = False
  spmd_annotations: Any = None
  shared_relpos_bias_factory: Optional[Callable[[], nn.Module]] = None
  shared_side_relpos_bias_factory: Optional[Callable[[], nn.Module]] = None

  # Embedders: Either a token_embedder_factory factory or shared token embedder
  # must be provided. The position embedder is optional and provided when
  # absolute position embeddings are desired.
  token_embedder_factory: Optional[Callable[[], embedding.Embed]] = None
  shared_token_embedder: Optional[embedding.Embed] = None
  position_embedder_factory: Optional[Callable[[], embedding.Embed]] = None

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
    self.embedder = embedding.MultiEmbed(embedders)

    self.input_dropout = self.input_dropout_factory()

    if self.scan_layers and (self.shared_relpos_bias_factory or
                             self.shared_side_relpos_bias_factory):
      raise ValueError("Scanned layer mode doesn't support shared relative"
                       'position biases.')
    self.relpos_bias = (
        self.shared_relpos_bias_factory()
        if self.shared_relpos_bias_factory is not None else None)
    self.side_relpos_bias = (
        self.shared_side_relpos_bias_factory()
        if self.shared_side_relpos_bias_factory is not None else None)
    layer_kwargs = dict(shared_relpos_bias=self.relpos_bias)
    if self.side_relpos_bias is not None:
      layer_kwargs['shared_side_relpos_bias'] = self.side_relpos_bias

    lyrf = lambda: self.layer_factory(**layer_kwargs)  # pytype: disable=wrong-keyword-args  # dict-kwargs
    lyrf = t5_architecture.maybe_remat(
        lyrf, self.layer_remat, self.scan_layers, static_argnums=(4,))

    if not self.scan_layers:
      self.layers = [lyrf() for _ in range(self.num_layers)]
      self.encoder = common.TransparentLayerSequence(self.layers)
    else:
      initializing = self.is_mutable_collection('params')
      # We scan the parameters along axis 1 as an XLA layout optimization.
      SCAN_AXIS = 1  # pylint: disable=invalid-name
      params_spec = SCAN_AXIS if initializing else transforms.ScanIn(SCAN_AXIS)
      cache_spec = 0
      scan_annotation = (
          self.spmd_annotations['encoder']
          if self.spmd_annotations is not None else None)
      lyrf = transforms.factory_scan(
          lyrf,
          in_axes=(nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
          variable_axes={
              'params': params_spec,
              'cache': cache_spec
          },
          split_rngs={
              'params': True,
              'dropout': True
          },
          length=self.num_layers,
          data_transform=transforms.inner_scan_spmd(scan_annotation, SCAN_AXIS),
      )
      self.encoder = lyrf()

    self.encoder_norm = self.layer_norm_factory()
    self.output_dropout = self.output_dropout_factory()

  def embed_and_combine_inputs(self,
                               inputs,
                               inputs_positions=None,
                               *,
                               enable_dropout: bool = True):
    """Returns the combined embedded inputs for further encoding."""
    assert inputs.ndim == 2  # (batch, len)

    if 'position_ids' in self.embedder.embedders:
      if inputs_positions is None:
        seq_length = inputs.shape[-1]
        inputs_positions = jnp.arange(seq_length)[None, :]
      embedded_inputs = self.embedder(  # pytype: disable=wrong-arg-types  # jax-ndarray
          token_ids=inputs, position_ids=inputs_positions)
    else:
      embedded_inputs = self.embedder(token_ids=inputs)  # pytype: disable=wrong-arg-types  # jax-ndarray

    embedded_inputs = self.input_dropout(
        embedded_inputs, deterministic=not enable_dropout)

    embedded_inputs = embedded_inputs.astype(self.dtype)
    return embedded_inputs

  def encode_from_continuous_inputs(self,
                                    inputs,
                                    inputs_mask=None,
                                    *,
                                    inputs_positions=None,
                                    inputs_segment_ids=None,
                                    enable_dropout: bool = True):
    """Applies all the layers starting from the continuous (embedded) inputs."""

    # Apply all encoder layers. Because of residual connection, the width of the
    # network is kept at `cfg.emb_dim` throughout.
    encoder_outputs = self.encoder(
        inputs,
        inputs_mask,
        inputs_positions=inputs_positions,
        inputs_segment_ids=inputs_segment_ids,
        enable_dropout=enable_dropout)
    if self.scan_layers:
      encoder_outputs = encoder_outputs[0]


    # Post-process the outputs of the final encoder layer.
    encoder_outputs = self.encoder_norm(encoder_outputs)
    encoder_outputs = self.output_dropout(
        encoder_outputs, deterministic=not enable_dropout)
    return encoder_outputs

  def __call__(self,
               inputs: Array,
               inputs_mask: Optional[Array] = None,
               *,
               inputs_positions: Optional[Array] = None,
               inputs_segment_ids: Optional[Array] = None,
               enable_dropout: bool = True):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      inputs_mask: bool array with same shape as `inputs` indicating True for
        non-padding tokens and False for padding. If `None` (the default), we
        automatically construct the mask based on which `inputs` are nonzero
        (rather than zero for padding).
      inputs_positions: input subsequence positions for packed examples.
      inputs_segment_ids: input segmentation info for packed examples.
      enable_dropout: Enables dropout if set to True.

    Returns:
      output of a transformer encoder.
    """
    if inputs_mask is None:
      inputs_mask = inputs > 0
    embedded_inputs = self.embed_and_combine_inputs(
        inputs,
        inputs_positions=inputs_positions,
        enable_dropout=enable_dropout)
    encoder_outputs = self.encode_from_continuous_inputs(
        embedded_inputs,
        inputs_mask=inputs_mask,
        inputs_positions=inputs_positions,
        inputs_segment_ids=inputs_segment_ids,
        enable_dropout=enable_dropout)
    return encoder_outputs


class LongEncoderDecoder(nn.Module, param_remapping.ParameterRemappable):
  """Transformer Model for sequence to sequence translation with long inputs.

  Attributes:
    encoder_factory: A callable that returns the lower-level LongEncoder object.
      If shared_token_embedder_factory is non-None, then the result of it will
      be passed as the `shared_token_embedder` argument to `encoder_factory`.
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
  encoder_factory: MakeLongEncoderFn
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
    return self.encoder(  # pytype: disable=attribute-error
        encoder_input_tokens,
        inputs_mask=encoder_input_tokens > 0,
        inputs_positions=encoder_positions,
        inputs_segment_ids=encoder_segment_ids,
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
      max_decode_length: Optional[int] = None):
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

    Returns:
      logits array from transformer decoder.
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
        max_decode_length=max_decode_length)

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


