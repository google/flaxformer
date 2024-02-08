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

"""Parallel Transformer decoder layer with fused parameters."""

import functools
from typing import Callable, Optional

from absl import logging
from aqt.jax_legacy.jax import flax_layers as aqt_flax_layers
from aqt.jax_legacy.jax import quant_config as aqt_config
from aqt.jax_legacy.jax import quantization as aqt
from flax import linen as nn
from jax import lax
import jax.numpy as jnp

from flaxformer import activation_partitioning
from flaxformer.architectures.common import param_remapping
from flaxformer.architectures.perceiver_ar import dense_attention
from flaxformer.architectures.perceiver_ar import perceiver_ar_architecture
from flaxformer.components import dense
from flaxformer.types import Array

# pylint: disable=not-callable
# pytype: disable=not-callable


class ParallelFusedDecoderLayer(nn.Module, param_remapping.ParameterRemappable):
  """Parallel Transformer decoder layer with fused parameters.

  Forked from the original to support Perceiver AR slicing.

  Attributes:
    self_attention: An instance of a self-attention module.
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
    sow_intermediates: Whether to track intermediates using Module.sow.
    is_quant_finetune_mode: Whether the layer is loaded for quantization
      finetuning. It's only applied in the context of quantization.
    num_latents: Number of latents and outputs.
  """
  self_attention: nn.Module
  mlp: nn.Module
  dropout_factory: Callable[[], nn.Module]
  layer_norm_factory: Callable[[], nn.Module]
  relative_position_bias_factory: Optional[Callable[[], nn.Module]] = None
  shared_relative_position_bias: Optional[nn.Module] = None
  activation_partitioning_dims: int = 1
  sow_intermediates: bool = False
  scanned: bool = False
  use_aqt: bool = False
  weight_params: Optional[aqt.QuantOps.WeightParams] = None
  act_params: Optional[aqt.QuantOps.ActHParams] = None
  possibly_use_quantized_vars: bool = False
  is_quant_finetune_mode: bool = False
  # num_latents is actually required, but has to be marked as optional because
  # we don't yet require Python 3.10, which provides keyword-only dataclasses.
  num_latents: Optional[int] = None

  def setup(self):
    if self.num_latents is None:
      raise ValueError('num_latents must be specified.')

    if self.activation_partitioning_dims != 1:
      logging.warning('ParallelFusedDecoderLayer.activation_partitioning_dims '
                      'is deprecated and will soon be removed.')

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

    self.layer_norm = self.layer_norm_factory()
    self.dropout = self.dropout_factory()

    if not isinstance(self.self_attention,
                      dense_attention.MultiQueryDotProductAttention):
      raise TypeError('ParallelFusedDecoderLayer requires Multiquery '
                      'attention.')
    num_heads = self.self_attention.num_heads
    if self.self_attention.head_dim is not None:
      head_dim = self.self_attention.head_dim
    else:
      head_dim = self.self_attention.qkv_features // num_heads
    if self.self_attention.out_features is None:
      raise ValueError('ParallelFusedDecoderLayer requires self-attention'
                       'with manually specified out_features.')
    embed_dim = self.self_attention.out_features
    n_activations = len(self.mlp.activations)
    mlp_intermediate_dim = self.mlp.intermediate_dim
    if mlp_intermediate_dim % num_heads != 0:
      raise ValueError('num_heads must divide mlp intermediate dimension')
    fused_out_dims = (num_heads,
                      (mlp_intermediate_dim // num_heads) * n_activations +
                      head_dim)

    # TODO: move the  AQT branching code complexity out to the
    # configuration system here and other places in Flaxformer.
    def make_dense(
        axis,
        features,
        use_bias,
        dtype,
        kernel_init,
        bias_init,
        reshape_kernel,
        kernel_axis_names,
        name,
    ):
      if self.use_aqt:
        if self.weight_params is None and self.act_params is None:
          raise ValueError(
              'If use_aqt is True, either of weights or acts quantization need '
              'to be specified using arguments `weight_params` or `act_params`.'
          )
        aqt_context = aqt_config.DynamicContext(
            update_bounds=False, collect_acts_stats=False)
        weight_prec = self.weight_params.prec if self.weight_params else None
        half_shift = self.weight_params.half_shift if self.weight_params else False
        aqt_hparams = aqt_flax_layers.DenseAqt.HParams(
            weight_prec=weight_prec,
            weight_half_shift=half_shift,
            quant_act=self.act_params,  # currently supports fixed bounds only.
            quant_type=aqt.QuantType.AQT,
            weight_quant_granularity=aqt_config.QuantGranularity.PER_CHANNEL,
        )
        if kernel_axis_names == ('heads', 'o_wo_fused', 'embed'):
          assert axis == (-2, -1)
          kernel_axis_names = ('joined_o_wo_fused', 'embed')
        aqt_dense = aqt_flax_layers.DenseAqt(
            features=features,
            hparams=aqt_hparams,
            train=self.is_quant_finetune_mode,
            dynamic_context=aqt_context,
            paxis_name=None,
            # No "cross-replica" reduction expressed in the XLA graph at this
            # stage. Will be imposed later, automatically, by XLA SPMD.
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
            dtype=dtype,
            name=name,
            possibly_use_quantized_vars=self.possibly_use_quantized_vars,
            kernel_axis_names=kernel_axis_names)
        # we do not have reshape kernel option here but we explicitly
        # reshape kernel.
        return functools.partial(aqt_dense, padding_mask=None)
      else:
        return dense.DenseGeneral(
            axis=axis,
            features=features,
            use_bias=use_bias,
            dtype=dtype,
            kernel_init=kernel_init,
            bias_init=bias_init,
            reshape_kernel=reshape_kernel,
            name=name,
            kernel_axis_names=kernel_axis_names)

    self.make_dense = make_dense
    self.q_wi_fused_args = dict(
        axis=-1,
        features=fused_out_dims,
        use_bias=self.self_attention.use_bias,
        dtype=self.self_attention.dtype,
        kernel_init=self.self_attention.kernel_init,
        bias_init=self.self_attention.bias_init,
        reshape_kernel=False,
        name='q_wi_fused',
        kernel_axis_names=('embed', 'heads', 'q_wi_fused'))
    self.kv_fused_args = dict(
        axis=-1,
        features=(1, 2 * head_dim),
        use_bias=self.self_attention.use_bias,
        dtype=self.self_attention.dtype,
        kernel_init=self.self_attention.kernel_init,
        bias_init=self.self_attention.bias_init,
        reshape_kernel=False,
        name='kv_fused',
        kernel_axis_names=('embed', 'multiquery_heads', 'kv_fused'))
    self.o_wo_fused_args = dict(
        axis=(-2, -1),
        features=embed_dim,
        use_bias=self.self_attention.use_bias,
        dtype=self.self_attention.dtype,
        kernel_init=self.self_attention.kernel_init,
        bias_init=self.self_attention.bias_init,
        reshape_kernel=False,
        name='o_wo_fused',
        # o_wo_fused = mlp//heads + head_dim
        kernel_axis_names=('heads', 'o_wo_fused', 'embed'))

  @nn.compact
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
    """Applies ParallelFusedDecoder1DBlock module.

    Args:
      targets: Input data for decoder with shape [batch_size,
        decoder_seq_length, decoder_hidden_size].
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
      output after transformer encoder-decoder block.
    """
    if num_latents and num_latents > self.num_latents:
      raise ValueError(
          f'Overridden num_latents ({num_latents}) must be <= self.num_latents '
          f'({self.num_latents}).')
    num_latents = num_latents or self.num_latents

    assert encoded is None, 'only pure decoder layer supported.'
    assert encoder_decoder_mask is None, 'only pure decoder layer supported.'
    layer_input = targets
    del targets
    # Shared relative position embedding attention biases.
    if self.relpos_bias:
      if decode and max_decode_length:
        decoder_bias = self.relpos_bias(max_decode_length, max_decode_length,
                                        False)
      else:
        decoder_bias = self.relpos_bias(layer_input.shape[-2],
                                        layer_input.shape[-2], False)
    else:
      decoder_bias = None

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

    x = self.layer_norm(
        layer_input,
        decode=decode,
        prefill=prefill,
        prefill_lengths=prefill_lengths)
    x = activation_partitioning.with_sharding_migration(
        x,
        self.activation_partitioning_dims,
        logical_axis_names=('batch', 'length', 'embed'))

    num_heads = self.self_attention.num_heads
    if self.self_attention.head_dim is not None:
      head_dim = self.self_attention.head_dim
    else:
      head_dim = self.self_attention.qkv_features // num_heads
    n_activations = len(self.mlp.activations)
    mlp_intermediate_dim = self.mlp.intermediate_dim

    layer_input_residual, x_queries, query_position_offset, logit_mask_queries = (
        perceiver_ar_architecture.create_residuals_and_queries(
            layer_input,
            x,
            logit_mask,
            num_latents=num_latents,
            sequence_lengths=sequence_lengths))
    del logit_mask_queries

    # Use local fused Q + W_i to calculate fused results.
    # [batch, length, embed], [heads, mlp//heads * n_act + head_dim] ->
    # [batch, length, heads, mlp//heads * n_act + head_dim]
    q_wi = self.make_dense(**self.q_wi_fused_args)(x_queries)
    # Slice out query.
    query = lax.dynamic_slice_in_dim(q_wi, 0, head_dim, -1)
    # Slice out MLP inputs.
    int_size = mlp_intermediate_dim // num_heads
    # wi[i]: [batch, length, heads, mlp//heads]
    wi = [
        lax.dynamic_slice_in_dim(q_wi, head_dim + i * int_size, int_size, -1)
        for i in range(n_activations)
    ]
    # Use local fused K + V to calculate fused results.
    kv = self.make_dense(**self.kv_fused_args)(x)
    kv = activation_partitioning.with_sharding(kv, 1)
    # Slice out key.
    key = jnp.squeeze(lax.dynamic_slice_in_dim(kv, 0, head_dim, -1), -2)
    # Slice out value.
    value = jnp.squeeze(
        lax.dynamic_slice_in_dim(kv, head_dim, head_dim, -1), -2)
    precomputed_qkv = (query, key, value)

    # y_att: [batch, length, heads, head_dim]
    y_att = self.self_attention(
        x_queries,
        x,
        mask=decoder_mask,
        bias=decoder_bias,
        precomputed_qkv=precomputed_qkv,
        enable_dropout=enable_dropout,
        decode=decode,
        prefill=prefill,
        prefill_lengths=prefill_lengths,
        query_position_offset=query_position_offset)
    # y_mlp: [batch, length, heads, mlp//heads]
    y_mlp = self.mlp(
        wi,
        decode=decode,
        prefill=prefill,
        prefill_lengths=prefill_lengths,
        enable_dropout=enable_dropout)
    # y_fused: [batch, length, heads, mlp//heads + head_dim]
    y_fused = jnp.concatenate([y_att, y_mlp], axis=-1)
    if self.use_aqt and self.weight_params is not None:
      weight_prec = self.weight_params.prec if self.weight_params else None
      half_shift = self.weight_params.half_shift if self.weight_params else False
      aqt_hparams = aqt_flax_layers.DenseGeneralAqt.HParams(
          weight_prec=weight_prec,
          weight_half_shift=half_shift,
          quant_act=None,  # currently supports fixed bounds only.
          weight_quant_granularity=aqt_config.QuantGranularity.PER_CHANNEL,
      )
      y_out = aqt_flax_layers.DenseGeneralAqt(
          **self.o_wo_fused_args,
          hparams=aqt_hparams,
          train=self.is_quant_finetune_mode,
          possibly_use_quantized_vars=self.possibly_use_quantized_vars)(
              y_fused)
    else:
      y_out = dense.DenseGeneral(**self.o_wo_fused_args)(y_fused)
    # y *= 2**-0.5
    z = layer_input_residual + self.dropout(
        y_out, deterministic=not enable_dropout)
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
