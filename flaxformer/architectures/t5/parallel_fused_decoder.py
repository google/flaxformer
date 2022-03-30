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

"""Parallel Transformer decoder layer with fused parameters."""

from typing import Callable, Optional

from absl import logging
from flax import linen as nn
from jax import lax
import jax.numpy as jnp
from flaxformer import activation_partitioning
from flaxformer.architectures.common import param_remapping
from flaxformer.components import dense
from flaxformer.components.attention import dense_attention
from flaxformer.types import Array

# pylint: disable=not-callable
# pytype: disable=not-callable


class ParallelFusedDecoderLayer(nn.Module, param_remapping.ParameterRemappable):
  """Parallel Transformer decoder layer with fused parameters.

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
    sow_intermediates: whether to track intermediates using Module.sow.
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

  def setup(self):
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

    self.q_wi_fused = dense.DenseGeneral(
        axis=-1,
        features=fused_out_dims,
        use_bias=self.self_attention.use_bias,
        dtype=self.self_attention.dtype,
        kernel_init=self.self_attention.kernel_init,
        bias_init=self.self_attention.bias_init,
        reshape_kernel=False,
        kernel_axis_names=('embed', 'heads', 'q_wi_fused'))
    self.kv_fused = dense.DenseGeneral(
        axis=-1,
        features=(1, 2 * head_dim),
        use_bias=self.self_attention.use_bias,
        dtype=self.self_attention.dtype,
        kernel_init=self.self_attention.kernel_init,
        bias_init=self.self_attention.bias_init,
        reshape_kernel=False,
        kernel_axis_names=('embed', 'multiquery_heads', 'kv_fused'))
    self.o_wo_fused = dense.DenseGeneral(
        axis=(-2, -1),
        features=embed_dim,
        use_bias=self.self_attention.use_bias,
        dtype=self.self_attention.dtype,
        kernel_init=self.self_attention.kernel_init,
        bias_init=self.self_attention.bias_init,
        reshape_kernel=False,
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
               prefill_lengths: Optional[Array] = None):
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

    Returns:
      output after transformer encoder-decoder block.
    """
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
    # Use local fused Q + W_i to calculate fused results.
    # [batch, length, embed], [heads, mlp//heads * n_act + head_dim] ->
    # [batch, length, heads, mlp//heads * n_act + head_dim]
    q_wi = self.q_wi_fused(x)
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
    kv = self.kv_fused(x)
    kv = activation_partitioning.with_sharding(kv, 1)
    # Slice out key.
    key = jnp.squeeze(lax.dynamic_slice_in_dim(kv, 0, head_dim, -1), -2)
    # Slice out value.
    value = jnp.squeeze(
        lax.dynamic_slice_in_dim(kv, head_dim, head_dim, -1), -2)
    precomputed_qkv = (query, key, value)

    # y_att: [batch, length, heads, head_dim]
    y_att = self.self_attention(
        x,
        x,
        mask=decoder_mask,
        bias=decoder_bias,
        precomputed_qkv=precomputed_qkv,
        enable_dropout=enable_dropout,
        decode=decode,
        prefill=prefill,
        prefill_lengths=prefill_lengths)
    # y_mlp: [batch, length, heads, mlp//heads]
    y_mlp = self.mlp(
        wi,
        decode=decode,
        prefill=prefill,
        prefill_lengths=prefill_lengths,
        enable_dropout=enable_dropout)
    # y_fused: [batch, length, heads, mlp//heads + head_dim]
    y_fused = jnp.concatenate([y_att, y_mlp], axis=-1)
    y_out = self.o_wo_fused(y_fused)
    # y *= 2**-0.5
    z = layer_input + self.dropout(y_out, deterministic=not enable_dropout)
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
