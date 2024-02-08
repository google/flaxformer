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

"""Utilties for h_transformer_1d_architecture_test."""

from typing import Callable

from flax import linen as nn
from jax import numpy as jnp

from flaxformer.architectures.h_transformer import h_attention
from flaxformer.architectures.h_transformer import h_transformer_1d_architecture
from flaxformer.architectures.h_transformer import h_transformer_utils as utils
from flaxformer.components import dense
from flaxformer.components import embedding
from flaxformer.components import layer_norm
from flaxformer.components.attention import dense_attention

_EMBEDDING_INIT = nn.initializers.normal(stddev=1.0)
_ATTENTION_KERNEL_INIT = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal')
_MLP_KERNEL_INIT = nn.initializers.variance_scaling(1.0, 'fan_in',
                                                    'truncated_normal')
_BIAS_INIT = nn.initializers.normal(stddev=1e-6)


def _token_embedder_factory(vocab_size: int, embed_size: int) -> nn.Module:
  return embedding.Embed(  # pytype: disable=wrong-arg-types  # jax-types
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


def _encoder_self_attention_factory(num_heads, num_clusters, qkv_features,
                                    use_rpb, use_multihead_rpb) -> nn.Module:
  return h_attention.OneDimEncoderSelfAttention(  # pytype: disable=wrong-arg-types  # jax-types
      num_heads=num_heads,
      num_clusters=num_clusters,
      qkv_features=qkv_features,
      dtype=jnp.float32,
      kernel_init=_ATTENTION_KERNEL_INIT,
      bias_init=_BIAS_INIT,
      use_rpb=use_rpb,
      use_multihead_rpb=use_multihead_rpb,
  )


def _decoder_self_attention_factory(num_heads, num_clusters, qkv_features,
                                    use_rpb, use_multihead_rpb):
  return h_attention.OneDimDecoderSelfAttention(  # pytype: disable=wrong-arg-types  # jax-types
      num_heads=num_heads,
      num_clusters=num_clusters,
      qkv_features=qkv_features,
      dtype=jnp.float32,
      kernel_init=_ATTENTION_KERNEL_INIT,
      bias_init=_BIAS_INIT,
      use_rpb=use_rpb,
      use_multihead_rpb=use_multihead_rpb,
  )


def _cross_attention_factory(num_heads, qkv_features):
  return dense_attention.MultiHeadDotProductAttention(  # pytype: disable=wrong-arg-types  # jax-types
      num_heads=num_heads,
      qkv_features=qkv_features,
      dtype=jnp.float32,
      kernel_init=_ATTENTION_KERNEL_INIT,
      bias_init=_BIAS_INIT,
      head_dim=None,
      use_bias=False,
      broadcast_dropout=True,
      dropout_rate=0.1)


# The default numbers are consistent with the testdata files.
def config_encoder(
    embed_size: int = 13,
    scan_layers: bool = False,
    layer_remat: utils.LayerRematOptions = utils.LayerRematOptions.LEGACY,
    layer_norm_factory: Callable[..., nn.Module] = layer_norm.T5LayerNorm,
    dropout_factory: Callable[
        ..., nn.Module] = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,)),
    num_layers: int = 3,
    vocab_size: int = 2000,
    qkv_features: int = 512,
    dropout_rate: float = 0.1,
    num_heads: int = 4,
    num_clusters: int = 2,
    use_rpb: bool = True,
    use_multihead_rpb: bool = True,
) -> h_transformer_1d_architecture.Encoder:
  """Configures an h-transformer encoder."""

  def _encoder_layer_factory():
    return h_transformer_1d_architecture.EncoderLayer(
        attention=_encoder_self_attention_factory(num_heads, num_clusters,
                                                  qkv_features, use_rpb,
                                                  use_multihead_rpb),
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
    layer_remat: utils.LayerRematOptions = utils.LayerRematOptions.LEGACY,
    layer_norm_factory: Callable[..., nn.Module] = layer_norm.T5LayerNorm,
    dropout_factory: Callable[
        ..., nn.Module] = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,)),
    num_layers: int = 3,
    vocab_size: int = 2000,
    qkv_features: int = 512,
    dropout_rate: float = 0.1,
    num_heads: int = 4,
    num_clusters: int = 4,
    use_rpb: bool = True,
    use_multihead_rpb: bool = True,
) -> h_transformer_1d_architecture.DecoderOnly:
  """Configures an h-transformer DecoderOnly."""

  def _decoder_only_layer_factory():
    return h_transformer_1d_architecture.DecoderOnlyLayer(
        attention=_decoder_self_attention_factory(num_heads, num_clusters,
                                                  qkv_features, use_rpb,
                                                  use_multihead_rpb),
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


# The default numbers are consistent with the testdata files.
def config_decoder(
    embed_size: int = 13,
    scan_layers: bool = False,
    parallel: bool = False,
    layer_remat: utils.LayerRematOptions = utils.LayerRematOptions.LEGACY,
    layer_norm_factory: Callable[..., nn.Module] = layer_norm.T5LayerNorm,
    dropout_factory: Callable[
        ..., nn.Module] = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,)),
    num_layers: int = 3,
    vocab_size: int = 2000,
    qkv_features: int = 512,
    dropout_rate: float = 0.1,
    num_heads: int = 4,
    num_clusters: int = 2,
    use_rpb: bool = True,
    use_multihead_rpb: bool = True,
) -> h_transformer_1d_architecture.Decoder:
  """Configures an h-transformer Decoder."""

  def _decoder_layer_factory():
    return h_transformer_1d_architecture.DecoderLayer(
        self_attention=_decoder_self_attention_factory(num_heads, num_clusters,
                                                       qkv_features, use_rpb,
                                                       use_multihead_rpb),
        encoder_decoder_attention=None,
        mlp=_mlp_factory(dropout_rate=dropout_rate, embed_size=embed_size),
        dropout_factory=dropout_factory,
        layer_norm_factory=layer_norm_factory,
        parallel=parallel,
        scanned=scan_layers)

  return h_transformer_1d_architecture.Decoder(
      layer_factory=_decoder_layer_factory,
      input_dropout_factory=dropout_factory,
      output_dropout_factory=dropout_factory,
      layer_norm_factory=layer_norm_factory,
      num_layers=num_layers,
      layer_remat=layer_remat,
      scan_layers=scan_layers,
      token_embedder_factory=(
          lambda: _token_embedder_factory(vocab_size, embed_size)))


# The default numbers are consistent with the testdata files.
def config_encoder_decoder(
    embed_size: int = 13,
    scan_layers: bool = False,
    layer_remat: utils.LayerRematOptions = utils.LayerRematOptions.LEGACY,
    layer_norm_factory: Callable[..., nn.Module] = layer_norm.T5LayerNorm,
    dropout_factory: Callable[
        ..., nn.Module] = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,)),
    num_layers: int = 3,
    vocab_size: int = 2000,
    qkv_features: int = 512,
    dropout_rate: float = 0.1,
    num_heads: int = 4,
    num_clusters: int = 2,
    use_rpb: bool = True,
    use_multihead_rpb: bool = True,
) -> h_transformer_1d_architecture.EncoderDecoder:
  """Configures an h-transformer EncoderDecoder."""

  def _encoder_layer_factory():
    return h_transformer_1d_architecture.EncoderLayer(
        attention=_decoder_self_attention_factory(num_heads, num_clusters,
                                                  qkv_features, use_rpb,
                                                  use_multihead_rpb),
        mlp=_mlp_factory(dropout_rate=dropout_rate, embed_size=embed_size),
        dropout_factory=dropout_factory,
        layer_norm_factory=layer_norm_factory,
        scanned=scan_layers)

  def _decoder_layer_factory():
    return h_transformer_1d_architecture.DecoderLayer(
        self_attention=_decoder_self_attention_factory(num_heads, num_clusters,
                                                       qkv_features, use_rpb,
                                                       use_multihead_rpb),
        encoder_decoder_attention=_cross_attention_factory(
            num_heads, qkv_features),
        mlp=_mlp_factory(dropout_rate=dropout_rate, embed_size=embed_size),
        dropout_factory=dropout_factory,
        layer_norm_factory=layer_norm_factory,
        scanned=scan_layers)

  def _encoder_factory(shared_token_embedder):
    assert shared_token_embedder is None
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

  def _decoder_factory(shared_token_embedder):
    assert shared_token_embedder is None
    return h_transformer_1d_architecture.Decoder(
        layer_factory=_decoder_layer_factory,
        input_dropout_factory=dropout_factory,
        output_dropout_factory=dropout_factory,
        layer_norm_factory=layer_norm_factory,
        num_layers=num_layers,
        layer_remat=layer_remat,
        scan_layers=scan_layers,
        token_embedder_factory=(
            lambda: _token_embedder_factory(vocab_size, embed_size)))

  return h_transformer_1d_architecture.EncoderDecoder(
      encoder_factory=_encoder_factory,
      decoder_factory=_decoder_factory,
      scan_layers=scan_layers,
      shared_token_embedder_factory=lambda: None,
  )
