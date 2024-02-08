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

"""Utilties for t5_architecture_test, and related tests."""

from typing import Any, Optional

from aqt.jax_legacy.jax import quantization as aqt
from flax import linen as nn
from jax import numpy as jnp
from flaxformer.architectures.t5 import parallel_fused_decoder
from flaxformer.architectures.t5 import t5_architecture
from flaxformer.components import dense
from flaxformer.components import embedding
from flaxformer.components import layer_norm
from flaxformer.components import relative_position_biases
from flaxformer.components.attention import dense_attention



EMBEDDING_INIT = nn.initializers.normal(stddev=1.0)
RELPOS_BIAS_INIT = nn.initializers.variance_scaling(1.0, 'fan_avg', 'uniform')
ATTENTION_KERNEL_INIT = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal'
)
MLP_KERNEL_INIT = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'truncated_normal'
)
FINAL_KERNEL_INIT = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'truncated_normal'
)
BIAS_INIT = nn.initializers.normal(stddev=1e-6)


def make_token_emb1(vocab_size, dtype, features=13):
  """First test configuration for token embeddings."""
  return embedding.Embed(  # pytype: disable=wrong-arg-types  # jax-types
      num_embeddings=vocab_size,
      features=features,
      cast_input_dtype=jnp.int32,
      dtype=dtype,
      attend_dtype=jnp.float32,  # for logit training stability
      embedding_init=EMBEDDING_INIT,
      name='token_embedder',
  )


def make_attention1(num_attn_heads, dtype, use_rotary_embedding=False):
  """First test configuration for attention."""
  return dense_attention.MultiHeadDotProductAttention(  # pytype: disable=wrong-arg-types  # jax-types
      num_heads=num_attn_heads,
      dtype=dtype,
      qkv_features=512,
      head_dim=None,
      kernel_init=ATTENTION_KERNEL_INIT,
      bias_init=BIAS_INIT,
      use_bias=False,
      broadcast_dropout=True,
      dropout_rate=0.1,
      use_rotary_embedding=use_rotary_embedding,
  )


def make_mlp1(dtype):
  """First test configuration for the MLP."""
  return dense.MlpBlock(
      use_bias=False,
      intermediate_dim=2048,
      activations=('relu',),
      kernel_init=MLP_KERNEL_INIT,
      bias_init=BIAS_INIT,
      intermediate_dropout_rate=0.1,
      final_dropout_rate=0.1,
      dtype=dtype,
  )


def _make_relative_position_bias(
    num_attn_heads: int, dtype: Any
) -> relative_position_biases.RelativePositionBiases:
  return relative_position_biases.RelativePositionBiases(
      num_buckets=32,
      max_distance=128,
      num_heads=num_attn_heads,
      dtype=dtype,
      embedding_init=RELPOS_BIAS_INIT,
  )


def make_config1(
    scan_layers=False, layer_remat='legacy', sow_intermediates=False
) -> t5_architecture.EncoderDecoder:
  """Returns an EncoderDecoder."""
  dtype = jnp.float32
  num_attn_heads = 8
  make_dropout = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,))
  make_layer_norm = layer_norm.T5LayerNorm

  def _make_encoder_layer(shared_relative_position_bias):
    assert shared_relative_position_bias is None
    return t5_architecture.EncoderLayer(
        attention=make_attention1(num_attn_heads, dtype),
        mlp=make_mlp1(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        relative_position_bias_factory=(
            lambda: _make_relative_position_bias(num_attn_heads, dtype)
        ),
        scanned=scan_layers,
        sow_intermediates=sow_intermediates,
    )

  def _make_decoder_layer(shared_relative_position_bias):
    assert shared_relative_position_bias is None
    return t5_architecture.DecoderLayer(
        self_attention=make_attention1(num_attn_heads, dtype),
        encoder_decoder_attention=make_attention1(num_attn_heads, dtype),
        mlp=make_mlp1(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        relative_position_bias_factory=(
            lambda: _make_relative_position_bias(num_attn_heads, dtype)
        ),
        scanned=scan_layers,
        sow_intermediates=sow_intermediates,
    )

  def _make_encoder(shared_token_embedder):
    assert shared_token_embedder is None
    return t5_architecture.Encoder(
        num_layers=3,
        token_embedder_factory=lambda: make_token_emb1(2_000, dtype),
        layer_factory=_make_encoder_layer,
        input_dropout_factory=make_dropout,
        output_dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        dtype=dtype,
        scan_layers=scan_layers,
        layer_remat=layer_remat,
        sow_intermediates=sow_intermediates,
    )

  def _make_decoder(shared_token_embedder):
    assert shared_token_embedder is None
    return t5_architecture.Decoder(
        num_layers=2,
        token_embedder_factory=lambda: make_token_emb1(2_000, dtype),
        layer_factory=_make_decoder_layer,
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        output_logits_factory=None,
        dtype=dtype,
        scan_layers=scan_layers,
        layer_remat=layer_remat,
        sow_intermediates=sow_intermediates,
    )

  return t5_architecture.EncoderDecoder(
      shared_token_embedder_factory=lambda: None,
      encoder_factory=_make_encoder,
      decoder_factory=_make_decoder,
      scan_layers=scan_layers,
  )


def make_parallel_transformer_config() -> t5_architecture.EncoderDecoder:
  """Returns an EncoderDecoder with parallel=True."""
  dtype = jnp.bfloat16
  num_attn_heads = 8
  make_dropout = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,))
  make_layer_norm = layer_norm.T5LayerNorm

  def _make_encoder_layer(shared_relative_position_bias):
    assert shared_relative_position_bias is None
    return t5_architecture.EncoderLayer(
        attention=make_attention1(num_attn_heads, dtype),
        mlp=make_mlp1(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        relative_position_bias_factory=(
            lambda: _make_relative_position_bias(num_attn_heads, dtype)
        ),
        parallel=True,
    )

  def _make_decoder_layer(shared_relative_position_bias):
    assert shared_relative_position_bias is None
    return t5_architecture.DecoderLayer(
        self_attention=make_attention1(num_attn_heads, dtype),
        encoder_decoder_attention=make_attention1(num_attn_heads, dtype),
        mlp=make_mlp1(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        relative_position_bias_factory=(
            lambda: _make_relative_position_bias(num_attn_heads, dtype)
        ),
        parallel=True,
    )

  def _make_encoder(shared_token_embedder):
    assert shared_token_embedder is None
    return t5_architecture.Encoder(
        num_layers=3,
        token_embedder_factory=lambda: make_token_emb1(2_000, dtype),
        layer_factory=_make_encoder_layer,
        input_dropout_factory=make_dropout,
        output_dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        dtype=dtype,
    )

  def _make_decoder(shared_token_embedder):
    assert shared_token_embedder is None
    return t5_architecture.Decoder(
        num_layers=2,
        token_embedder_factory=lambda: make_token_emb1(2_000, dtype),
        layer_factory=_make_decoder_layer,
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        output_logits_factory=None,
        dtype=dtype,
    )

  return t5_architecture.EncoderDecoder(
      shared_token_embedder_factory=lambda: None,
      encoder_factory=_make_encoder,
      decoder_factory=_make_decoder,
  )


def make_parallel_fused_transformer_config(
    use_aqt: bool = False,
    weight_params: Optional[aqt.QuantOps.WeightParams] = None,
    possibly_use_quantized_vars: bool = False,
    is_quant_finetune_mode: bool = False,
) -> t5_architecture.DecoderOnly:
  """Returns an EncoderDecoder with parallel=True."""
  dtype = jnp.bfloat16
  num_attn_heads = 8
  num_features = 13
  make_dropout = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,))
  make_layer_norm = layer_norm.T5LayerNorm

  def _make_mq_attention(num_attn_heads, dtype):
    """First test configuration for attention."""
    return dense_attention.MultiQueryDotProductAttention(  # pytype: disable=wrong-arg-types  # jax-types
        num_heads=num_attn_heads,
        dtype=dtype,
        qkv_features=512,
        out_features=num_features,
        head_dim=None,
        kernel_init=ATTENTION_KERNEL_INIT,
        bias_init=BIAS_INIT,
        use_bias=False,
        broadcast_dropout=True,
        dropout_rate=0.1,
        rescale_logits=True,
    )

  def _make_fusion_mlp(dtype):
    """First test configuration for the MLP."""
    return dense.MlpBlock(
        use_bias=False,
        intermediate_dim=2048,
        out_dim=13,
        precomputed_intermediates=True,
        fuse_kernels=False,
        activations=('swish', 'linear'),
        kernel_init=MLP_KERNEL_INIT,
        bias_init=BIAS_INIT,
        intermediate_dropout_rate=0.1,
        final_dropout_rate=0.1,
        dtype=dtype,
    )

  def _make_decoder_layer(shared_relative_position_bias):
    assert shared_relative_position_bias is None
    return parallel_fused_decoder.ParallelFusedDecoderLayer(
        self_attention=_make_mq_attention(num_attn_heads, dtype),
        mlp=_make_fusion_mlp(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        relative_position_bias_factory=(
            lambda: _make_relative_position_bias(num_attn_heads, dtype)
        ),
        use_aqt=use_aqt,
        weight_params=weight_params,
        possibly_use_quantized_vars=possibly_use_quantized_vars,
        is_quant_finetune_mode=is_quant_finetune_mode,
    )

  def _make_output_logits():
    return dense.DenseGeneral(  # pytype: disable=wrong-arg-types  # jax-types
        4,
        dtype=dtype,
        kernel_init=FINAL_KERNEL_INIT,
        bias_init=BIAS_INIT,
        use_bias=False,
    )

  def _embedder():
    return make_token_emb1(2_000, dtype, num_features)

  def _make_decoder(shared_token_embedder):
    assert shared_token_embedder is None
    return t5_architecture.Decoder(
        num_layers=2,
        token_embedder_factory=_embedder,
        layer_factory=_make_decoder_layer,
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        output_logits_factory=_make_output_logits,
        dtype=dtype,
    )

  return t5_architecture.DecoderOnly(
      decoder_factory=_make_decoder,
  )


# TODO: DRY up with above configs.
def make_config2_shared_relative_position_bias() -> (
    t5_architecture.EncoderDecoder
):
  """Returns an EncoderDecoder with shared relative position biases."""
  dtype = jnp.float32
  num_attn_heads = 8
  make_dropout = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,))
  make_layer_norm = layer_norm.T5LayerNorm

  def _make_encoder_layer(shared_relative_position_bias):
    assert shared_relative_position_bias is not None
    return t5_architecture.EncoderLayer(
        attention=make_attention1(num_attn_heads, dtype),
        mlp=make_mlp1(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        shared_relative_position_bias=shared_relative_position_bias,
    )

  def _make_decoder_layer(shared_relative_position_bias):
    assert shared_relative_position_bias is not None
    return t5_architecture.DecoderLayer(
        self_attention=make_attention1(num_attn_heads, dtype),
        encoder_decoder_attention=make_attention1(num_attn_heads, dtype),
        mlp=make_mlp1(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        shared_relative_position_bias=shared_relative_position_bias,
    )

  def _make_encoder(*, shared_token_embedder=None):
    assert shared_token_embedder is None
    return t5_architecture.Encoder(
        num_layers=3,
        token_embedder_factory=lambda: make_token_emb1(2_000, dtype),
        layer_factory=_make_encoder_layer,
        input_dropout_factory=make_dropout,
        output_dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        shared_relative_position_bias_factory=(
            lambda: _make_relative_position_bias(num_attn_heads, dtype)
        ),
        dtype=dtype,
    )

  def _make_decoder(*, shared_token_embedder=None):
    assert shared_token_embedder is None
    return t5_architecture.Decoder(
        num_layers=2,
        token_embedder_factory=lambda: make_token_emb1(2_000, dtype),
        layer_factory=_make_decoder_layer,
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        output_logits_factory=None,
        shared_relative_position_bias_factory=(
            lambda: _make_relative_position_bias(num_attn_heads, dtype)
        ),
        dtype=dtype,
    )

  return t5_architecture.EncoderDecoder(
      shared_token_embedder_factory=lambda: None,
      encoder_factory=_make_encoder,
      decoder_factory=_make_decoder,
  )


# TODO: DRY up with above configs.
def make_config3_shared_token_embedder() -> t5_architecture.EncoderDecoder:
  """Returns an EncoderDecoder with a shared token embedder."""
  dtype = jnp.float32
  num_attn_heads = 8
  make_dropout = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,))
  make_layer_norm = layer_norm.T5LayerNorm
  sow_intermediates = True
  capture_gradients = True

  def _make_encoder_layer(shared_relative_position_bias):
    assert shared_relative_position_bias is None
    return t5_architecture.EncoderLayer(
        attention=make_attention1(num_attn_heads, dtype),
        mlp=make_mlp1(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        relative_position_bias_factory=(
            lambda: _make_relative_position_bias(num_attn_heads, dtype)
        ),
        sow_intermediates=sow_intermediates,
    )

  def _make_decoder_layer(shared_relative_position_bias):
    assert shared_relative_position_bias is None
    return t5_architecture.DecoderLayer(
        self_attention=make_attention1(num_attn_heads, dtype),
        encoder_decoder_attention=make_attention1(num_attn_heads, dtype),
        mlp=make_mlp1(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        relative_position_bias_factory=(
            lambda: _make_relative_position_bias(num_attn_heads, dtype)
        ),
        sow_intermediates=sow_intermediates,
    )

  def _make_encoder(*, shared_token_embedder=None):
    return t5_architecture.Encoder(
        num_layers=3,
        shared_token_embedder=shared_token_embedder,
        layer_factory=_make_encoder_layer,
        input_dropout_factory=make_dropout,
        output_dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        dtype=dtype,
        sow_intermediates=sow_intermediates,
        capture_gradients=capture_gradients,
    )

  def _make_decoder(*, shared_token_embedder=None):
    return t5_architecture.Decoder(
        num_layers=2,
        shared_token_embedder=shared_token_embedder,
        layer_factory=_make_decoder_layer,
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        output_logits_factory=None,
        dtype=dtype,
        sow_intermediates=sow_intermediates,
        capture_gradients=capture_gradients,
    )

  return t5_architecture.EncoderDecoder(
      shared_token_embedder_factory=lambda: make_token_emb1(71, dtype),
      encoder_factory=_make_encoder,
      decoder_factory=_make_decoder,
  )


def test_make_decoder_only1() -> t5_architecture.DecoderOnly:
  """Returns a DecoderOnly."""
  dtype = jnp.float32
  num_attn_heads = 8
  make_dropout = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,))
  make_layer_norm = layer_norm.T5LayerNorm

  def _make_decoder_layer(shared_relative_position_bias):
    assert shared_relative_position_bias is None
    return t5_architecture.DecoderLayer(
        self_attention=make_attention1(num_attn_heads, dtype),
        encoder_decoder_attention=None,
        mlp=make_mlp1(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        relative_position_bias_factory=(
            lambda: _make_relative_position_bias(num_attn_heads, dtype)
        ),
    )

  def make_output_logits():
    return dense.DenseGeneral(  # pytype: disable=wrong-arg-types  # jax-types
        4,
        dtype=dtype,
        kernel_init=FINAL_KERNEL_INIT,
        bias_init=BIAS_INIT,
        use_bias=False,
    )

  def _make_decoder(*, shared_token_embedder=None):
    assert shared_token_embedder is None
    return t5_architecture.Decoder(
        num_layers=2,
        token_embedder_factory=lambda: make_token_emb1(4, dtype),
        layer_factory=_make_decoder_layer,
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        output_logits_factory=make_output_logits,
        dtype=dtype,
    )

  return t5_architecture.DecoderOnly(decoder_factory=_make_decoder)
