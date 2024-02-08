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

"""Sparse parallel Transformer decoder layer with fused parameters."""

from typing import Callable, Optional, Tuple

from flax import linen as nn
from flax.linen import partitioning as flax_partitioning
from jax import lax
import jax.numpy as jnp

from flaxformer.architectures.common import param_remapping
from flaxformer.components import dense
from flaxformer.components.attention import dense_attention
from flaxformer.types import Array

# pylint: disable=not-callable
# pytype: disable=not-callable


class SparseParallelFusedDecoderLayer(
    nn.Module, param_remapping.ParameterRemappable
):
  """Sparse parallel Transformer decoder layer with fused parameters.

  The projection matrices from the self-attention and MLP models are fused into
  three kernels: Q-Wi, KV, and O-Wo, which are applied using `q_wi_fused`,
  `kv_fused`, and `o_wo_fused`, respectively. Any of these projections can be
  either sparse (MoE) or dense. The attention dot product and MLP activations
  are applied outside these projections.

  Note that, as for the "regular" SparseDecoderLayer, individual
  SparseParallelFusedDecoderLayer(s) cannot be scanned over. Only blocks of MoE
  layers are ever scanned; see also moe_architecture.SparseDecoder.

  Attributes:
    self_attention: An instance of a self-attention module. The projections of
      this module are applied indirectly through the fused projections.
    mlp: The MLP module. The projections of this module are applied indirectly
      through the fused projections.
    q_wi_fused: Projection sublayer applying fused attention-MLP Q-Wi kernel.
    o_wo_fused: Projection sublayer applying fused attention-MLP O-Wo kernel.
    kv_fused: Projection applying fused KV attention kernel.
    dropout_factory:  A callable that returns a new dropout instance. This is
      applied after the attention module.
    layer_norm_factory: A callable that returns a new layer norm. This is
      applied before the attention module and before the MLP.
    relative_position_bias_factory: A callable that returns relative position
      bias instances. This should only be used for per-layer relative position
      biases; please use `shared_relative_position_bias` if they are shared
      among layers.
    shared_relative_position_bias: An instance of a shared relative position
      bias module, usually owned by the Decoder.
    sow_intermediates: Whether to track intermediates using Module.sow.
  """

  self_attention: nn.Module
  mlp: nn.Module
  q_wi_fused: nn.Module
  o_wo_fused: nn.Module
  kv_fused: nn.Module
  dropout_factory: Callable[[], nn.Module]
  layer_norm_factory: Callable[[], nn.Module]
  relative_position_bias_factory: Optional[Callable[[], nn.Module]] = None
  shared_relative_position_bias: Optional[nn.Module] = None
  sow_intermediates: bool = False

  def setup(self):
    if (
        self.relative_position_bias_factory is not None
        and self.shared_relative_position_bias is not None
    ):
      raise ValueError(
          'Please set at most one of `relative_position_bias_factory` and '
          '`shared_relative_position_bias`. (They can both be None however, '
          'e.g. for absolute position embeddings.)'
      )
    self.relpos_bias = (
        self.relative_position_bias_factory()
        if self.relative_position_bias_factory is not None
        else self.shared_relative_position_bias
    )
    self.layer_norm = self.layer_norm_factory()
    self.dropout = self.dropout_factory()

  @nn.compact
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
      prefill_lengths: Optional[Array] = None,
  ) -> Array:
    """Applies SparseParallelFusedDecoder1DBlock module.

    Args:
      targets: Input data for decoder with shape [batch_size,
        decoder_seq_length, decoder_hidden_size].
      encoded: Must be None, as this block is for Decoder-only models. Only kept
        for __call__ signature uniformity.
      decoder_mask: Decoder self-attention mask.
      encoder_decoder_mask: Must be None, as this block is for Decoder-only
        models. Only kept for __call__ signature uniformity.
      logit_mask: A mask (e.g., padding logit mask) to be applied to the
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
      Output after Transformer decoder block.
    """
    assert encoded is None, 'Only pure decoder layer is supported.'
    assert encoder_decoder_mask is None, 'Only pure decoder layer is supported.'

    layer_input = targets
    del targets

    # Shared relative position embedding attention biases.
    if self.relpos_bias:
      if decode and max_decode_length:
        decoder_bias = self.relpos_bias(
            max_decode_length, max_decode_length, False
        )
      else:
        decoder_bias = self.relpos_bias(
            layer_input.shape[-2], layer_input.shape[-2], False
        )
    else:
      decoder_bias = None

    assert layer_input.ndim == 3
    layer_input = flax_partitioning.with_sharding_constraint(
        layer_input, logical_axis_resources=('batch', 'length', 'embed')
    )

    if prefill and prefill_lengths is None:
      # Figure out how far each element in the batch fills the cache based
      # on the mask. We index each element in the batch, the first head
      # dim (because this is always set to one), and the first query
      # vector. If there is any prefix at all, the first element in the
      # prefix would be part of it.
      prefill_lengths = jnp.sum(decoder_mask[:, 0, 0, :], axis=-1).astype(
          jnp.int32
      )

    x = self.layer_norm(
        layer_input,
        decode=decode,
        prefill=prefill,
        prefill_lengths=prefill_lengths,
    )
    x = flax_partitioning.with_sharding_constraint(
        x, logical_axis_resources=('batch', 'length', 'embed')
    )

    num_heads = self.self_attention.num_heads
    if self.self_attention.head_dim is not None:
      head_dim = self.self_attention.head_dim
    else:
      head_dim = self.self_attention.qkv_features // num_heads
    n_activations = len(self.mlp.activations)
    mlp_intermediate_dim = self.mlp.intermediate_dim

    del logit_mask

    # Use local fused Q + W_i to calculate fused results.
    # [batch, length, embed], [heads, mlp//heads * n_act + head_dim] ->
    # Unpack to [batch, length, heads, mlp//heads * n_act + head_dim].
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
    kv = flax_partitioning.with_sharding_constraint(
        kv, ('batch', 'length', 'embed', 'heads')
    )

    # Slice out key.
    key = jnp.squeeze(lax.dynamic_slice_in_dim(kv, 0, head_dim, -1), -2)

    # Slice out value.
    value = jnp.squeeze(
        lax.dynamic_slice_in_dim(kv, head_dim, head_dim, -1), -2
    )
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
        prefill_lengths=prefill_lengths,
    )

    # y_mlp: [batch, length, heads, mlp//heads]
    y_mlp = self.mlp(
        wi,
        decode=decode,
        prefill=prefill,
        prefill_lengths=prefill_lengths,
        enable_dropout=enable_dropout,
    )

    # y_fused: [batch, length, heads, mlp//heads + head_dim]
    y_fused = jnp.concatenate([y_att, y_mlp], axis=-1)

    y_out = self.o_wo_fused(y_fused)
    # y *= 2**-0.5
    z = layer_input + self.dropout(y_out, deterministic=not enable_dropout)
    z = flax_partitioning.with_sharding_constraint(
        z, logical_axis_resources=('batch', 'length', 'embed')
    )

    if self.sow_intermediates:
      self.sow('intermediates', 'activations', z)


    return z


def compute_fused_o_wo_dims(
    attention_module: dense_attention.MultiQueryDotProductAttention,
) -> int:
  """Returns the output dimension of the fused O-Wo projection.

  Args:
    attention_module: Self-attention module used in the fused layer.

  Returns:
    Fused O-Wo projection dimension.

  Raises:
    ValueError if `out_features` is not specified on the attention module.
  """
  if attention_module.out_features is None:
    raise ValueError(
        'SparseParallelFusedDecoderLayer requires self-attention'
        'with manually specified `out_features`.'
    )
  return attention_module.out_features


def compute_fused_kv_dims(
    attention_module: dense_attention.MultiQueryDotProductAttention,
) -> Tuple[int, int]:
  """Returns the output dimensions for the fused KV projection.

  Args:
    attention_module: Self-attention module used in the fused layer.

  Returns:
    Fused KV dimension.
  """
  head_dim = _compute_head_dim(attention_module)
  return 1, 2 * head_dim


def compute_fused_q_wi_dims(
    attention_module: dense_attention.MultiQueryDotProductAttention,
    mlp: dense.MlpBlock,
) -> Tuple[int, int]:
  """Returns the output dimensions for the Q-Wi fused projection.

  Args:
    attention_module: Self-attention module used in the fused layer.
    mlp: MLP module used in the fused layer.

  Returns:
    Q-Wi fused projection dimension.

  Raises:
    ValueError if number of attention heads does not divide MLP intermediate
    dimension.
  """
  num_heads = attention_module.num_heads
  head_dim = _compute_head_dim(attention_module)
  n_activations = len(mlp.activations)
  mlp_intermediate_dim = mlp.intermediate_dim
  if mlp_intermediate_dim % num_heads != 0:
    raise ValueError(
        'Number of attention heads does not divide MLP intermediate dimension'
    )

  return (
      num_heads,
      (mlp_intermediate_dim // num_heads) * n_activations + head_dim,
  )


def _compute_head_dim(
    attention_module: dense_attention.MultiQueryDotProductAttention,
) -> int:
  """Returns the head dimension of the attention module."""
  if attention_module.head_dim is not None:
    return attention_module.head_dim
  else:
    return attention_module.qkv_features // attention_module.num_heads
