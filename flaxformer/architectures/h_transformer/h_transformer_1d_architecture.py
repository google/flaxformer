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

"""Defines architecture classes for h_transformer_1d models."""

import enum
from typing import Callable, Optional, Any, Tuple

from flax import linen as nn
import jax
from jax import numpy as jnp
from typing_extensions import Protocol

from flaxformer import transformer_common as common
from flaxformer.architectures.common import param_remapping
from flaxformer.components import embedding
from flaxformer.components import transforms
from flaxformer.components.attention import dense_attention
from flaxformer.types import Array

_SCAN_AXIS = 1


@enum.unique
class LayerRematOptions(enum.Enum):
  """Options for layer remat configuration.

  Attributes:
    NONE: For no use of jax.remat.
    MINIMAL: For recomputing only non-matmul ops in backprop.
    FULL: For recomputing the whole layer in backprop.
    LEGACY: For compatibility with existing configs. Previously
      scan_layers=False implied NONE, scan_layers=True implied FULL.
  """
  NONE = enum.auto()
  MINIMAL = enum.auto()
  FULL = enum.auto()
  LEGACY = enum.auto()


class MakeEncoderLayerFn(Protocol):
  """Signature for functions that make an input encoder layer."""

  def __call__(self) -> 'EncoderLayer':
    """Makes an input encoder layer.

    Returns:
      EncoderLayer instance.
    """


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


def maybe_remat(lyrf: Callable[[], nn.Module], layer_remat: LayerRematOptions,
                scan_layers: bool,
                static_argnums: Tuple[int, ...]) -> Callable[[], nn.Module]:
  """Maybe applies jax.remat with the indicated policy to a layer factory.

  Args:
    lyrf: Encoder or decoder layer factory.
    layer_remat: Config for per-layer remat. See commenst for LayerRematOptions.
    scan_layers: Whether to use jax.lax.scan for the stack of layers.
    static_argnums: The static_argnums to use for the jax.remat call.

  Returns:
    Potentially remat-wrapped layer factory.

  Raises:
    ValueError: This is triggered by an unsupported layer_mat option.
  """
  if layer_remat == LayerRematOptions.LEGACY:
    layer_remat = (
        LayerRematOptions.FULL if scan_layers else LayerRematOptions.NONE)

  if layer_remat == LayerRematOptions.NONE:
    return lyrf

  if layer_remat == LayerRematOptions.FULL:
    remat_policy = None
  elif layer_remat == LayerRematOptions.MINIMAL:
    remat_policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
  else:
    raise ValueError('Unsupported layer_remat option.')

  lyrf = transforms.factory_remat(
      lyrf,
      concrete=False,
      prevent_cse=False,
      static_argnums=static_argnums,
      policy=remat_policy)
  return lyrf


def _activation_partitioning_fn(y):
  return nn.partitioning.with_sharding_constraint(y,
                                                  ('batch', 'length', 'embed'))


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
  parallel: bool = False
  sow_intermediates: bool = False
  scanned: bool = False

  def setup(self):
    self.pre_attention_layer_norm = self.layer_norm_factory()
    self.post_attention_dropout = self.dropout_factory()
    if not self.parallel:
      self.pre_mlp_layer_norm = self.layer_norm_factory()
      self.post_mlp_dropout = self.dropout_factory()

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
      Outputs from a transformer encoder layer.

    Raises:
      ValueError: This is triggered if inputs array has the wrong rank.
    """
    if inputs.ndim != 3:
      raise ValueError(f'Expect inputs.ndim=3, but inputs.ndim={inputs.ndim}')

    layer_input = _activation_partitioning_fn(inputs)
    layer_input = self.pre_attention_layer_norm(layer_input)
    layer_input = _activation_partitioning_fn(layer_input)
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
      x = _activation_partitioning_fn(x)
      # MLP block.
      y = self.pre_mlp_layer_norm(x)
      y = _activation_partitioning_fn(y)
      y = self.mlp(y, enable_dropout=enable_dropout)
      y = x + self.post_mlp_dropout(y, deterministic=not enable_dropout)
    y = _activation_partitioning_fn(y)

    if self.sow_intermediates:
      self.sow('intermediates', 'activations', y)

    # Scan expects functions to have a signature: fn(carry, in) --> carry, out
    if self.scanned:
      return y, None
    else:
      return y


class EncoderAndDecoderOnlyBase(nn.Module, param_remapping.ParameterRemappable):
  """Base class for Encoder and DecoderOnly classes.

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
    token_embedder_factory: A callable that returns a token embedder. Please
      provide either this or `shared_token_embedder`.
    shared_token_embedder: A callable that returns a token embedder shared
      between both encoder and decoder.
  """
  layer_factory: Callable[[], nn.Module]
  input_dropout_factory: Callable[[], nn.Module]
  output_dropout_factory: Callable[[], nn.Module]
  layer_norm_factory: Callable[[], nn.Module]
  num_layers: int
  layer_remat: LayerRematOptions = LayerRematOptions.LEGACY
  scan_layers: bool = False
  spmd_annotations: Any = None

  # Embedders: Either a token_embedder_factory factory or shared_token_embedder
  # must be provided.
  token_embedder_factory: Optional[Callable[[], embedding.Embed]] = None
  shared_token_embedder: Optional[embedding.Embed] = None

  def setup(self):
    self._setup_embedders()
    self.input_dropout = self.input_dropout_factory()
    self.output_layer_norm = self.layer_norm_factory()
    self.output_dropout = self.output_dropout_factory()

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

  def _setup_layers(self, module_name: str) -> Callable[..., Array]:
    lyrf = maybe_remat(
        self.layer_factory,
        self.layer_remat,
        self.scan_layers,
        static_argnums=(2,))

    if self.scan_layers:
      initializing = self.is_mutable_collection('params')
      # We scan the parameters along axis 1 as an XLA layout optimization.
      params_spec = _SCAN_AXIS if initializing else transforms.ScanIn(
          _SCAN_AXIS)
      cache_spec = 0
      scan_annotation = (
          self.spmd_annotations[module_name]
          if self.spmd_annotations is not None else None)
      lyrf = transforms.factory_scan(
          lyrf,
          in_axes=(nn.broadcast, nn.broadcast),
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
      )
      return lyrf()
    else:
      self.layers = [lyrf() for _ in range(self.num_layers)]
      return common.TransparentLayerSequence(self.layers)


class Encoder(EncoderAndDecoderOnlyBase):
  """A stack of input encoder layers.

  Attributes:
    layer_factory: A callable that returns an EncoderLayer instance.
  """
  layer_factory: MakeEncoderLayerFn

  def setup(self):
    super().setup()
    self.encoder = self._setup_layers('encoder')

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


class MakeDecoderOnlyLayerFn(Protocol):
  """Signature for functions that make a DecoderOnly layer."""

  def __call__(self) -> 'DecoderOnlyLayer':
    """Makes a DecoderOnly layer.

    Returns:
      DecoderOnlyLayer instance.
    """


class MakeDecoderFn(Protocol):
  """Signature for functions that make a low-level Decoder instance."""

  def __call__(
      self,
      *,
      shared_token_embedder: Optional[embedding.Embed] = None,
      spmd_annotations: Any = None,
  ) -> 'DecoderOnly':
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


class DecoderOnly(EncoderAndDecoderOnlyBase):
  """A stack of DecoderOnly layers.

  Attributes:
    layer_factory: A callable that returns a DecoderLayer instance.
    output_logits_factory: A callable that returns the output logits. If not
      provided, then the token embedders are used.
    sow_intermediates: Whether to track intermediates using Module.sow.
  """
  layer_factory: MakeDecoderOnlyLayerFn
  output_logits_factory: Optional[Callable[[], nn.Module]] = None
  sow_intermediates: bool = False

  def setup(self):
    super().setup()
    self.decoder = self._setup_layers('decoder')
    self.output_logits_factory: Callable[[], nn.Module]
    self.output_logits: Optional[nn.Module]
    self.output_logits = (
        self.output_logits_factory() if self.output_logits_factory else None)

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
      decode: Whether to prepare and use an autoregressive cache. This is
        unused in h-transformer.
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
