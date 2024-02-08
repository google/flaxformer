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

"""Utilities for perceiver_ar_architecture_test."""

from flax import linen as nn
from jax import numpy as jnp

from flaxformer.architectures.perceiver_ar import decoder_layer
from flaxformer.architectures.perceiver_ar import dense_attention
from flaxformer.architectures.perceiver_ar import parallel_fused_decoder
from flaxformer.architectures.perceiver_ar import perceiver_ar_architecture
from flaxformer.architectures.t5 import t5_architecture_test_utils
from flaxformer.components import dense
from flaxformer.components import layer_norm


def make_attention1(num_attn_heads, dtype, use_rotary_embedding=False):
  """First test configuration for attention."""
  return dense_attention.MultiHeadDotProductAttention(  # pytype: disable=wrong-arg-types  # jax-types
      num_heads=num_attn_heads,
      dtype=dtype,
      qkv_features=512,
      head_dim=None,
      kernel_init=t5_architecture_test_utils.ATTENTION_KERNEL_INIT,
      bias_init=t5_architecture_test_utils.BIAS_INIT,
      use_bias=False,
      broadcast_dropout=True,
      dropout_rate=0.1,
      use_rotary_embedding=use_rotary_embedding)


def test_make_decoder_only1(
    num_latents: int, parallel: bool) -> perceiver_ar_architecture.DecoderOnly:
  """Returns a DecoderOnly."""
  dtype = jnp.float32
  num_attn_heads = 8
  make_dropout = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,))
  make_layer_norm = layer_norm.T5LayerNorm

  def _make_decoder_layer(shared_relative_position_bias):
    assert shared_relative_position_bias is None
    return decoder_layer.DecoderLayer(
        self_attention=make_attention1(
            num_attn_heads, dtype, use_rotary_embedding=True),
        encoder_decoder_attention=None,
        mlp=t5_architecture_test_utils.make_mlp1(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        relative_position_bias_factory=None,
        num_latents=num_latents,
        parallel=parallel)

  def make_output_logits():
    return dense.DenseGeneral(  # pytype: disable=wrong-arg-types  # jax-types
        4,
        dtype=dtype,
        kernel_init=t5_architecture_test_utils.FINAL_KERNEL_INIT,
        bias_init=t5_architecture_test_utils.BIAS_INIT,
        use_bias=False)

  def _make_decoder(*, shared_token_embedder=None):
    assert shared_token_embedder is None
    return perceiver_ar_architecture.Decoder(
        num_layers=2,
        token_embedder_factory=(
            lambda: t5_architecture_test_utils.make_token_emb1(4, dtype)),
        layer_factory=_make_decoder_layer,
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        output_logits_factory=make_output_logits,
        dtype=dtype,
        num_latents=num_latents,
    )

  return perceiver_ar_architecture.DecoderOnly(
      decoder_factory=_make_decoder, num_latents=num_latents)


def make_parallel_fused_transformer_config(
    num_latents: int) -> perceiver_ar_architecture.DecoderOnly:
  """Returns a DecoderOnly with parallel=True."""
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
        kernel_init=t5_architecture_test_utils.ATTENTION_KERNEL_INIT,
        bias_init=t5_architecture_test_utils.BIAS_INIT,
        use_bias=False,
        broadcast_dropout=True,
        dropout_rate=0.1,
        rescale_logits=True,
        use_rotary_embedding=True)

  def _make_fusion_mlp(dtype):
    """First test configuration for the MLP."""
    return dense.MlpBlock(
        use_bias=False,
        intermediate_dim=2048,
        out_dim=13,
        precomputed_intermediates=True,
        fuse_kernels=False,
        activations=('swish', 'linear'),
        kernel_init=t5_architecture_test_utils.MLP_KERNEL_INIT,
        bias_init=t5_architecture_test_utils.BIAS_INIT,
        intermediate_dropout_rate=0.1,
        final_dropout_rate=0.1,
        dtype=dtype)

  def _make_decoder_layer(shared_relative_position_bias):
    assert shared_relative_position_bias is None
    return parallel_fused_decoder.ParallelFusedDecoderLayer(
        self_attention=_make_mq_attention(num_attn_heads, dtype),
        mlp=_make_fusion_mlp(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        relative_position_bias_factory=None,
        num_latents=num_latents)

  def _make_output_logits():
    return dense.DenseGeneral(  # pytype: disable=wrong-arg-types  # jax-types
        4,
        dtype=dtype,
        kernel_init=t5_architecture_test_utils.FINAL_KERNEL_INIT,
        bias_init=t5_architecture_test_utils.BIAS_INIT,
        use_bias=False)

  def _embedder():
    return t5_architecture_test_utils.make_token_emb1(2_000, dtype,
                                                      num_features)

  def _make_decoder(shared_token_embedder):
    assert shared_token_embedder is None
    return perceiver_ar_architecture.Decoder(
        num_layers=2,
        token_embedder_factory=_embedder,
        layer_factory=_make_decoder_layer,
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        output_logits_factory=_make_output_logits,
        dtype=dtype,
        num_latents=num_latents)

  return perceiver_ar_architecture.DecoderOnly(
      decoder_factory=_make_decoder, num_latents=num_latents)
