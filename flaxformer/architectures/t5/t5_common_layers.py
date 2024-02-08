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

"""A collection of standard layers for the T5-1.0 and T5-1.1 model variants."""
import functools

from flax import linen as nn
from jax import numpy as jnp

from flaxformer.architectures.t5 import t5_architecture
from flaxformer.components import dense
from flaxformer.components import embedding as embedding_layers
from flaxformer.components import layer_norm
from flaxformer.components import relative_position_biases
from flaxformer.components.attention import dense_attention

BIAS_INIT = nn.initializers.normal(stddev=1e-6)
MLP_KERNEL_INIT = nn.initializers.variance_scaling(1.0, 'fan_in',
                                                   'truncated_normal')


def attention_layer(num_heads, head_dim, dropout_rate, dtype=jnp.bfloat16):
  """Create an dense_attention layer for T5-style architectures."""
  return dense_attention.MultiHeadDotProductAttention(  # pytype: disable=wrong-arg-types  # jax-types
      num_heads=num_heads,
      head_dim=head_dim,
      qkv_features=None,
      kernel_init=nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'),
      bias_init=BIAS_INIT,
      use_bias=False,
      broadcast_dropout=True,
      dropout_rate=dropout_rate,
      dtype=dtype)


def mlp_block(mlp_dim, dropout_rate, activations, dtype=jnp.bfloat16):
  """Create an MLP layer for T5-style architectures."""
  return dense.MlpBlock(
      use_bias=False,
      intermediate_dim=mlp_dim,
      activations=activations,
      kernel_init=MLP_KERNEL_INIT,
      bias_init=BIAS_INIT,
      intermediate_dropout_rate=dropout_rate,
      final_dropout_rate=0,
      dtype=dtype)


def relative_position_bias(num_heads, dtype=jnp.bfloat16):
  """Create a standard position bias layer for T5-style architectures."""
  return relative_position_biases.RelativePositionBiases(
      num_heads=num_heads,
      num_buckets=32,
      max_distance=128,
      dtype=dtype,
      embedding_init=nn.initializers.variance_scaling(1.0, 'fan_avg',
                                                      'uniform'))


def embedding(vocabulary_size, embedding_dim, dtype=jnp.bfloat16):
  """Create a standard embedding layer for T5-style architectures."""
  return embedding_layers.Embed(  # pytype: disable=wrong-arg-types  # jax-types
      num_embeddings=vocabulary_size,
      features=embedding_dim,
      cast_input_dtype=jnp.int32,
      dtype=dtype,
      attend_dtype=jnp.float32,  # for logit training stability
      embedding_init=nn.initializers.normal(stddev=1.0),
      one_hot=True,
      name='token_embedder')


def dropout(rate):
  """Create a standard dropout layer for T5-style architectures."""
  return nn.Dropout(rate=rate, broadcast_dims=(-2,))


def encoder_layer(num_heads,
                  head_dim,
                  mlp_dim,
                  dropout_rate,
                  activations,
                  shared_relative_position_bias=None,
                  dtype=jnp.bfloat16):
  """Create a standard encoder layer for T5-style architectures."""
  dropout_factory = functools.partial(dropout, rate=dropout_rate)
  return t5_architecture.EncoderLayer(
      attention=attention_layer(
          num_heads=num_heads,
          head_dim=head_dim,
          dropout_rate=dropout_rate,
          dtype=dtype),
      mlp=mlp_block(
          mlp_dim=mlp_dim,
          activations=activations,
          dropout_rate=dropout_rate,
          dtype=dtype),
      dropout_factory=dropout_factory,
      layer_norm_factory=functools.partial(layer_norm.T5LayerNorm, dtype=dtype),
      shared_relative_position_bias=shared_relative_position_bias)  # pytype: disable=wrong-keyword-args


def decoder_layer(num_heads,
                  head_dim,
                  mlp_dim,
                  dropout_rate,
                  activations,
                  shared_relative_position_bias=None,
                  dtype=jnp.bfloat16):
  """Create a standard decoder layer for T5-style architectures."""
  dropout_factory = functools.partial(dropout, rate=dropout_rate)
  return t5_architecture.DecoderLayer(
      self_attention=attention_layer(
          num_heads=num_heads,
          head_dim=head_dim,
          dropout_rate=dropout_rate,
          dtype=dtype),
      encoder_decoder_attention=attention_layer(
          num_heads=num_heads,
          head_dim=head_dim,
          dropout_rate=dropout_rate,
          dtype=dtype),
      mlp=mlp_block(
          mlp_dim=mlp_dim,
          activations=activations,
          dropout_rate=dropout_rate,
          dtype=dtype),
      dropout_factory=dropout_factory,
      layer_norm_factory=functools.partial(layer_norm.T5LayerNorm, dtype=dtype),
      shared_relative_position_bias=shared_relative_position_bias)  # pytype: disable=wrong-keyword-args


def encoder(num_heads,
            head_dim,
            mlp_dim,
            num_layers,
            shared_token_embedder,
            dropout_rate,
            activations,
            dtype=jnp.bfloat16):
  """Create a standard encoder for T5-style architectures."""
  encoder_layer_factory = functools.partial(
      encoder_layer,
      num_heads=num_heads,
      head_dim=head_dim,
      mlp_dim=mlp_dim,
      activations=activations,
      dropout_rate=dropout_rate,
      dtype=dtype)
  dropout_factory = functools.partial(dropout, rate=dropout_rate)

  relative_position_bias_factory = functools.partial(
      relative_position_bias, num_heads=num_heads, dtype=dtype)
  return t5_architecture.Encoder(
      layer_factory=encoder_layer_factory,
      input_dropout_factory=dropout_factory,
      output_dropout_factory=dropout_factory,
      layer_norm_factory=functools.partial(layer_norm.T5LayerNorm, dtype=dtype),
      num_layers=num_layers,
      shared_token_embedder=shared_token_embedder,
      shared_relative_position_bias_factory=relative_position_bias_factory,
      dtype=dtype)  # pytype: disable=wrong-keyword-args


def decoder(num_heads,
            head_dim,
            mlp_dim,
            num_layers,
            shared_token_embedder,
            dropout_rate,
            activations,
            output_logits_factory=None,
            dtype=jnp.bfloat16):
  """Create a standard decoder for T5-style architectures."""
  decoder_layer_factory = functools.partial(
      decoder_layer,
      num_heads=num_heads,
      head_dim=head_dim,
      mlp_dim=mlp_dim,
      activations=activations,
      dropout_rate=dropout_rate,
      dtype=dtype)
  relative_position_bias_factory = functools.partial(
      relative_position_bias, num_heads=num_heads, dtype=dtype)
  dropout_factory = functools.partial(dropout, rate=dropout_rate)

  return t5_architecture.Decoder(
      layer_factory=decoder_layer_factory,
      dropout_factory=dropout_factory,
      layer_norm_factory=functools.partial(layer_norm.T5LayerNorm, dtype=dtype),
      num_layers=num_layers,
      shared_token_embedder=shared_token_embedder,
      shared_relative_position_bias_factory=relative_position_bias_factory,
      output_logits_factory=output_logits_factory,
      dtype=dtype)  # pytype: disable=wrong-keyword-args
