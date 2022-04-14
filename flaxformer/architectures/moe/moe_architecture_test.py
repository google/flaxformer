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

"""Tests for moe_architecture."""

import functools
from typing import Optional, Sequence

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from jax import numpy as jnp
from jax import random
import numpy as np

from flaxformer.architectures.moe import moe_architecture
from flaxformer.architectures.moe import moe_enums
from flaxformer.architectures.moe import moe_layers
from flaxformer.architectures.moe import routing
from flaxformer.architectures.t5 import t5_architecture
from flaxformer.components import dense
from flaxformer.components import embedding
from flaxformer.components import layer_norm
from flaxformer.components import relative_position_biases
from flaxformer.components.attention import dense_attention

DecoderLayer = t5_architecture.DecoderLayer
EncoderLayer = t5_architecture.EncoderLayer
LayerLayout = moe_enums.LayerLayout
MoeLayer = moe_layers.MoeLayer


EMBEDDING_INIT = nn.initializers.normal(stddev=1.0)
RELPOS_BIAS_INIT = nn.initializers.variance_scaling(1.0, 'fan_avg', 'uniform')
ATTENTION_KERNEL_INIT = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal')
MLP_KERNEL_INIT = nn.initializers.variance_scaling(1.0, 'fan_in',
                                                   'truncated_normal')
BIAS_INIT = nn.initializers.normal(stddev=1e-6)
DTYPE = jnp.float32


_make_dropout = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,))
_make_layer_norm = layer_norm.T5LayerNorm


def _make_token_emb(num_embeddings: int) -> embedding.Embed:
  """Test configuration for token embeddings."""
  return embedding.Embed(
      num_embeddings=num_embeddings,
      features=13,
      cast_input_dtype=jnp.int32,
      dtype=DTYPE,
      attend_dtype=jnp.float32,  # for logit training stability
      embedding_init=EMBEDDING_INIT,
      name='token_embedder')


def _make_multi_query_attention(
) -> dense_attention.MultiQueryDotProductAttention:
  """Test configuration for attention."""
  return dense_attention.MultiQueryDotProductAttention(
      num_heads=8,
      dtype=DTYPE,
      qkv_features=16,
      head_dim=None,
      kernel_init=ATTENTION_KERNEL_INIT,
      bias_init=BIAS_INIT,
      use_bias=False,
      broadcast_dropout=True,
      dropout_rate=0.1)


def _make_dense_mlp() -> dense.MlpBlock:
  """Test configuration for the MLP."""
  return dense.MlpBlock(
      use_bias=False,
      intermediate_dim=8,
      activations=('relu',),
      kernel_init=MLP_KERNEL_INIT,
      bias_init=BIAS_INIT,
      intermediate_dropout_rate=0.1,
      final_dropout_rate=0.1,
      dtype=DTYPE)


def _make_sparse_mlp() -> MoeLayer:
  """Test configuration for the sparse MLP."""
  expert = dense.MlpBlock(
      use_bias=True,
      activations=('gelu',),
      intermediate_dim=16,
      intermediate_dropout_rate=0.1)
  router = routing.TokensChooseMaskedRouter(
      router_weights=routing.RouterWeights(),
      num_selected_experts=1,
      jitter_noise=0.,
      batch_prioritized_routing=True,
      dtype=jnp.float32)
  return MoeLayer(
      num_experts=4,
      router=router,
      max_group_size=2,
      train_capacity_factor=1.,
      eval_capacity_factor=1.,
      expert=expert,
      dtype=jnp.float32)


def _make_relative_position_bias(
) -> relative_position_biases.RelativePositionBiases:
  """Test configuration for the position bias."""
  return relative_position_biases.RelativePositionBiases(
      num_buckets=32,
      max_distance=64,
      num_heads=8,
      dtype=DTYPE,
      embedding_init=RELPOS_BIAS_INIT)


def _make_sparse_encoder_layer(
    shared_relative_position_bias: Optional[nn.Module] = None) -> EncoderLayer:
  """Test configuration for sparse MLP-attention encoder blocks."""
  return EncoderLayer(
      attention=_make_multi_query_attention(),
      mlp=_make_sparse_mlp(),
      dropout_factory=_make_dropout,
      layer_norm_factory=_make_layer_norm,
      relative_position_bias_factory=_make_relative_position_bias,
      shared_relative_position_bias=shared_relative_position_bias)


def _make_dense_encoder_layer(
    shared_relative_position_bias: Optional[nn.Module] = None) -> EncoderLayer:
  """Test configuration for dense MLP-attention encoder blocks."""
  return EncoderLayer(
      attention=_make_multi_query_attention(),
      mlp=_make_dense_mlp(),
      dropout_factory=_make_dropout,
      layer_norm_factory=_make_layer_norm,
      relative_position_bias_factory=_make_relative_position_bias,
      shared_relative_position_bias=shared_relative_position_bias)


def _make_sparse_decoder_layer(
    shared_relative_position_bias: Optional[nn.Module] = None) -> DecoderLayer:
  """Test configuration for sparse MLP-self-attention decoder blocks."""
  return DecoderLayer(
      self_attention=_make_multi_query_attention(),
      encoder_decoder_attention=_make_multi_query_attention(),
      mlp=_make_sparse_mlp(),
      dropout_factory=_make_dropout,
      layer_norm_factory=_make_layer_norm,
      relative_position_bias_factory=_make_relative_position_bias,
      shared_relative_position_bias=shared_relative_position_bias)


def _make_dense_decoder_layer(
    shared_relative_position_bias: Optional[nn.Module] = None) -> DecoderLayer:
  """Test configuration for dense MLP-self-self_attention decoder blocks."""
  return DecoderLayer(
      self_attention=_make_multi_query_attention(),
      encoder_decoder_attention=_make_multi_query_attention(),
      mlp=_make_dense_mlp(),
      dropout_factory=_make_dropout,
      layer_norm_factory=_make_layer_norm,
      relative_position_bias_factory=_make_relative_position_bias,
      shared_relative_position_bias=shared_relative_position_bias)


class MoeArchitectureTest(parameterized.TestCase):

  def test_moe_architecture(self):
    batch_size = 2
    seq_length = 4
    num_embeddings = 64

    encoder_factory = functools.partial(
        moe_architecture.SparseEncoder,
        num_layers=2,
        num_sparse_layers=1,
        sparse_layout=LayerLayout.MIXED,
        sparse_layer_factory=_make_sparse_encoder_layer,
        dense_layer_factory=_make_dense_encoder_layer,
        dropout_factory=_make_dropout,
        layer_norm_factory=_make_layer_norm,
    )
    decoder_factory = functools.partial(
        moe_architecture.SparseDecoder,
        num_layers=2,
        num_sparse_layers=1,
        sparse_layout=LayerLayout.MIDDLE,
        sparse_layer_factory=_make_sparse_decoder_layer,
        dense_layer_factory=_make_dense_decoder_layer,
        dropout_factory=_make_dropout,
        layer_norm_factory=_make_layer_norm,
    )

    transformer = t5_architecture.EncoderDecoder(
        shared_token_embedder_factory=functools.partial(
            _make_token_emb, num_embeddings=num_embeddings),
        encoder_factory=encoder_factory,
        decoder_factory=decoder_factory,
    )

    encoder_input_tokens = np.zeros((batch_size, seq_length), dtype=np.float32)
    decoder_input_tokens = np.zeros((batch_size, seq_length), dtype=np.float32)
    decoder_target_tokens = np.zeros((batch_size, seq_length), dtype=np.float32)
    output, variables = transformer.init_with_output(
        {
            'params': random.PRNGKey(0),
            'dropout': random.PRNGKey(0)
        },
        encoder_input_tokens=encoder_input_tokens,
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        enable_dropout=False,
    )

    self.assertEqual(output.shape, (batch_size, seq_length, num_embeddings))

    # For readability, we only verify the weight names to check that the layouts
    # are correct.

    encoder_0_layer = variables['params']['encoder']['encoders_0']['layers_0']
    # First encoder layer should have regular self-attention.
    self.assertIn('attention', encoder_0_layer)
    # And it should contain a dense MLP (because there is only one sparse MLP
    # layer at the top for MIXED layout).
    self.assertIn('wi', encoder_0_layer['mlp'])  # Dense
    self.assertIn('wo', encoder_0_layer['mlp'])  # Dense
    self.assertNotIn('expert', encoder_0_layer['mlp'])  # Sparse

    encoder_1_layer = variables['params']['encoder']['encoders_1']['layers_0']
    # Second encoder layer should have regular self-attention.
    self.assertIn('attention', encoder_1_layer)
    # And it should contain a sparse MoeLayer (from sparse MIXED layout).
    self.assertIn('expert', encoder_1_layer['mlp'])  # Sparse

    decoder_0_layer = variables['params']['decoder']['decoders_0']['layers_0']
    # First decoder layer should have regular encoder-decoder attention.
    self.assertIn('query', decoder_0_layer['encoder_decoder_attention'])
    # It should contain regular self-attention.
    self.assertIn('self_attention', decoder_0_layer)
    # And it should contain a regular dense MLP (sparse layout is MIDDLE, so
    # first layer will be dense MLP).
    self.assertIn('wi', decoder_0_layer['mlp'])  # Dense
    self.assertIn('wo', decoder_0_layer['mlp'])  # Dense
    self.assertNotIn('expert', decoder_0_layer['mlp'])  # Sparse

    decoder_1_layer = variables['params']['decoder']['decoders_1']['layers_0']
    # Second decoder layer should have regular encoder-decoder attention.
    self.assertIn('query', decoder_1_layer['encoder_decoder_attention'])
    # It should contain regular self-attention.
    self.assertIn('self_attention', decoder_1_layer)
    # And it should contain a sparse MLP (sparse layout is MIDDLE, which for a
    # 2 layer decoder results in top layer being sparse).
    self.assertIn('expert', decoder_1_layer['mlp'])  # Sparse

  def test_degenerate_architecture(self):
    batch_size = 2
    seq_length = 4
    num_embeddings = 64

    encoder_factory = functools.partial(
        moe_architecture.SparseEncoder,
        num_layers=2,
        num_sparse_layers=0,
        sparse_layout=LayerLayout.MIXED,
        sparse_layer_factory=_make_sparse_encoder_layer,
        dense_layer_factory=_make_dense_encoder_layer,
        dropout_factory=_make_dropout,
        layer_norm_factory=_make_layer_norm,
    )
    decoder_factory = functools.partial(
        moe_architecture.SparseDecoder,
        num_layers=2,
        num_sparse_layers=2,
        sparse_layout=LayerLayout.MIDDLE,
        sparse_layer_factory=_make_sparse_decoder_layer,
        dense_layer_factory=_make_dense_decoder_layer,
        dropout_factory=_make_dropout,
        layer_norm_factory=_make_layer_norm,
    )

    transformer = t5_architecture.EncoderDecoder(
        shared_token_embedder_factory=functools.partial(
            _make_token_emb, num_embeddings=num_embeddings),
        encoder_factory=encoder_factory,
        decoder_factory=decoder_factory,
    )

    encoder_input_tokens = np.zeros((batch_size, seq_length), dtype=np.float32)
    decoder_input_tokens = np.zeros((batch_size, seq_length), dtype=np.float32)
    decoder_target_tokens = np.zeros((batch_size, seq_length), dtype=np.float32)
    output, variables = transformer.init_with_output(
        {
            'params': random.PRNGKey(0),
            'dropout': random.PRNGKey(0)
        },
        encoder_input_tokens=encoder_input_tokens,
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        enable_dropout=False,
    )

    self.assertEqual(output.shape, (batch_size, seq_length, num_embeddings))

    # For degenerate cases, we only have one underlying Encoder.
    encoder = variables['params']['encoder']['encoders_0']
    for layer in ['layers_0', 'layers_1']:
      encoder_layer = encoder[layer]
      # All encoder layers should have regular attention.
      self.assertIn('attention', encoder_layer)
      # And dense MLPs.
      self.assertIn('wi', encoder_layer['mlp'])  # Dense
      self.assertIn('wo', encoder_layer['mlp'])  # Dense
      self.assertNotIn('expert', encoder_layer['mlp'])  # Sparse

    # For degenerate cases, we only have one underlying Decoder.
    decoder = variables['params']['decoder']['decoders_0']
    for layer in ['layers_0', 'layers_1']:
      decoder_layer = decoder[layer]
      # All decoder layers should have self-attention.
      self.assertIn('self_attention', decoder_layer)
      # Regular cross encoder-decoder attention.
      self.assertIn('encoder_decoder_attention', decoder_layer)
      # And sparse MLPs.
      self.assertIn('expert', decoder_layer['mlp'])  # Sparse
      self.assertNotIn('wi', decoder_layer['mlp'])  # Dense
      self.assertNotIn('wo', decoder_layer['mlp'])  # Dense

  @parameterized.named_parameters(
      dict(
          testcase_name=LayerLayout.BOTTOM.name,
          sparse_layout=LayerLayout.BOTTOM,
          num_sparse_layers=4,
          expected_sparse_layers=[0, 1, 2, 3]),
      dict(
          testcase_name=LayerLayout.MIDDLE.name,
          sparse_layout=LayerLayout.MIDDLE,
          num_sparse_layers=2,
          expected_sparse_layers=[5, 6]),
      dict(
          testcase_name=LayerLayout.MIXED.name,
          sparse_layout=LayerLayout.MIXED,
          num_sparse_layers=3,
          expected_sparse_layers=[3, 7, 11]),
      dict(
          testcase_name=LayerLayout.TOP.name,
          sparse_layout=LayerLayout.TOP,
          num_sparse_layers=1,
          expected_sparse_layers=[11]))
  def test_sparse_layouts(self, sparse_layout: LayerLayout,
                          num_sparse_layers: int,
                          expected_sparse_layers: Sequence[int]):
    num_layers = 12
    expected_dense_layers = set(range(num_layers)) - set(expected_sparse_layers)

    for layer in expected_sparse_layers:
      self.assertTrue(
          moe_architecture._is_sparse_layer(layer, num_layers,
                                            num_sparse_layers, sparse_layout))
    for layer in expected_dense_layers:
      self.assertFalse(
          moe_architecture._is_sparse_layer(layer, num_layers,
                                            num_sparse_layers, sparse_layout))


if __name__ == '__main__':
  absltest.main()
