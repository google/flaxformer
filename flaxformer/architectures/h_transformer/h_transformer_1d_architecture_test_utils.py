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

"""Utilties for h_transformer_1d_architecture_test."""

from typing import Callable

from flax import linen as nn
from jax import numpy as jnp

from flaxformer.architectures.h_transformer import h_attention
from flaxformer.architectures.h_transformer import h_transformer_1d_architecture
from flaxformer.components import dense
from flaxformer.components import embedding
from flaxformer.components import layer_norm

_EMBEDDING_INIT = nn.initializers.normal(stddev=1.0)
_ATTENTION_KERNEL_INIT = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal')
_MLP_KERNEL_INIT = nn.initializers.variance_scaling(1.0, 'fan_in',
                                                    'truncated_normal')
_BIAS_INIT = nn.initializers.normal(stddev=1e-6)


def _token_embedder_factory(vocab_size: int, embed_size: int) -> nn.Module:
  return embedding.Embed(
      num_embeddings=vocab_size,
      features=embed_size,
      cast_input_dtype=jnp.int32,
      dtype=jnp.float32,
      attend_dtype=jnp.float32,
      embedding_init=_EMBEDDING_INIT,
      name='token_embedder')


def _mlp_factory(dropout_rate: float = 0.1, embed_size: int = 13) -> nn.Module:
  return dense.MlpBlock(
      use_bias=False,
      intermediate_dim=2 * embed_size,
      activations=('relu',),
      intermediate_dropout_rate=dropout_rate,
      final_dropout_rate=dropout_rate,
      kernel_init=_MLP_KERNEL_INIT,
      bias_init=_BIAS_INIT,
      dtype=jnp.float32)


# The default numbers are consistent with the testdata files.
def config_encoder(
    embed_size: int = 13,
    scan_layers: bool = False,
    layer_remat: h_transformer_1d_architecture
    .LayerRematOptions = h_transformer_1d_architecture.LayerRematOptions.LEGACY,
    layer_norm_factory: Callable[..., nn.Module] = layer_norm.T5LayerNorm,
    dropout_factory: Callable[
        ..., nn.Module] = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,)),
    num_layers: int = 3,
    vocab_size: int = 2000,
    qkv_features: int = 512,
    dropout_rate: float = 0.1,
    num_heads: int = 4,
    num_clusters: int = 2,
    dtype: jnp.dtype = jnp.float32,
) -> h_transformer_1d_architecture.Encoder:
  """Configures an h-transformer encoder."""

  def _encoder_self_attention_factory():
    return h_attention.OneDimEncoderSelfAttention(
        num_heads=num_heads,
        num_clusters=num_clusters,
        qkv_features=qkv_features,
        dtype=dtype,
        kernel_init=_ATTENTION_KERNEL_INIT,
        bias_init=_BIAS_INIT)

  def _encoder_layer_factory():
    return h_transformer_1d_architecture.EncoderLayer(
        attention=_encoder_self_attention_factory(),
        mlp=_mlp_factory(dropout_rate=dropout_rate, embed_size=embed_size),
        dropout_factory=dropout_factory,
        layer_norm_factory=layer_norm_factory,
        scanned=scan_layers)

  return h_transformer_1d_architecture.Encoder(
      layer_factory=_encoder_layer_factory,
      input_dropout_factory=dropout_factory,
      output_dropout_factory=dropout_factory,
      layer_norm_factory=layer_norm_factory,
      num_layers=num_layers,
      layer_remat=layer_remat,
      scan_layers=scan_layers,
      token_embedder_factory=(
          lambda: _token_embedder_factory(vocab_size, embed_size)))


# The default numbers are consistent with the testdata files.
def config_decoder_only(
    embed_size: int = 13,
    scan_layers: bool = False,
    layer_remat: h_transformer_1d_architecture
    .LayerRematOptions = h_transformer_1d_architecture.LayerRematOptions.LEGACY,
    layer_norm_factory: Callable[..., nn.Module] = layer_norm.T5LayerNorm,
    dropout_factory: Callable[
        ..., nn.Module] = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,)),
    num_layers: int = 3,
    vocab_size: int = 2000,
    qkv_features: int = 512,
    dropout_rate: float = 0.1,
    num_heads: int = 4,
    num_clusters: int = 2,
    dtype: jnp.dtype = jnp.float32,
) -> h_transformer_1d_architecture.DecoderOnly:
  """Configures an h-transformer DecoderOnly."""

  def _decoder_self_attention_factory():
    return h_attention.OneDimDecoderSelfAttention(
        num_heads=num_heads,
        num_clusters=num_clusters,
        qkv_features=qkv_features,
        dtype=dtype,
        kernel_init=_ATTENTION_KERNEL_INIT,
        bias_init=_BIAS_INIT)

  def _decoder_only_layer_factory():
    return h_transformer_1d_architecture.DecoderOnlyLayer(
        attention=_decoder_self_attention_factory(),
        mlp=_mlp_factory(dropout_rate=dropout_rate, embed_size=embed_size),
        dropout_factory=dropout_factory,
        layer_norm_factory=layer_norm_factory,
        scanned=scan_layers)

  return h_transformer_1d_architecture.DecoderOnly(
      layer_factory=_decoder_only_layer_factory,
      input_dropout_factory=dropout_factory,
      output_dropout_factory=dropout_factory,
      layer_norm_factory=layer_norm_factory,
      num_layers=num_layers,
      layer_remat=layer_remat,
      scan_layers=scan_layers,
      token_embedder_factory=(
          lambda: _token_embedder_factory(vocab_size, embed_size)))
