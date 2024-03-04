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

"""Tests for moe_architecture."""

import functools
from typing import Optional, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import flax
from flax import linen as nn
import jax
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
SparseDecoderLayer = moe_architecture.SparseDecoderLayer
SparseEncoderLayer = moe_architecture.SparseEncoderLayer
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

MOE_METRICS = ('auxiliary_loss', 'router_z_loss', 'fraction_tokens_left_behind',
               'expert_usage', 'router_confidence')


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


def _make_sparse_mlp(ignore_padding_tokens: bool = False) -> MoeLayer:
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
      dtype=jnp.float32,
      ignore_padding_tokens=ignore_padding_tokens)
  return MoeLayer(
      num_experts=4,
      num_expert_partitions=4,
      router=router,
      max_group_size=2,
      train_capacity_factor=1.,
      eval_capacity_factor=1.,
      expert=expert,
      num_model_partitions=1,
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
    extra_mlp: bool = False,
    shared_relative_position_bias: Optional[nn.Module] = None,
    scanned: bool = False,
    ignore_padding_tokens: bool = False) -> SparseEncoderLayer:
  """Test configuration for sparse MLP-attention encoder blocks."""
  return SparseEncoderLayer(
      attention=_make_multi_query_attention(),
      mlp=_make_sparse_mlp(ignore_padding_tokens=ignore_padding_tokens),
      extra_mlp=_make_dense_mlp() if extra_mlp else None,
      dropout_factory=_make_dropout,
      layer_norm_factory=_make_layer_norm,
      relative_position_bias_factory=_make_relative_position_bias,
      shared_relative_position_bias=shared_relative_position_bias,
      scanned=scanned)


def _make_dense_encoder_layer(shared_relative_position_bias: Optional[
    nn.Module] = None,
                              scanned: bool = False) -> EncoderLayer:
  """Test configuration for dense MLP-attention encoder blocks."""
  return EncoderLayer(
      attention=_make_multi_query_attention(),
      mlp=_make_dense_mlp(),
      dropout_factory=_make_dropout,
      layer_norm_factory=_make_layer_norm,
      relative_position_bias_factory=_make_relative_position_bias,
      shared_relative_position_bias=shared_relative_position_bias,
      scanned=scanned)


def _make_sparse_decoder_layer(
    extra_mlp: bool = False,
    shared_relative_position_bias: Optional[nn.Module] = None,
    scanned: bool = False,
    ignore_padding_tokens: bool = False) -> SparseDecoderLayer:
  """Test configuration for sparse MLP-self-attention decoder blocks."""
  return SparseDecoderLayer(
      self_attention=_make_multi_query_attention(),
      encoder_decoder_attention=_make_multi_query_attention(),
      mlp=_make_sparse_mlp(ignore_padding_tokens=ignore_padding_tokens),
      extra_mlp=_make_dense_mlp() if extra_mlp else None,
      dropout_factory=_make_dropout,
      layer_norm_factory=_make_layer_norm,
      relative_position_bias_factory=_make_relative_position_bias,
      shared_relative_position_bias=shared_relative_position_bias,
      scanned=scanned)


def _make_dense_decoder_layer(shared_relative_position_bias: Optional[
    nn.Module] = None,
                              scanned: bool = False) -> DecoderLayer:
  """Test configuration for dense MLP-self-self_attention decoder blocks."""
  return DecoderLayer(
      self_attention=_make_multi_query_attention(),
      encoder_decoder_attention=_make_multi_query_attention(),
      mlp=_make_dense_mlp(),
      dropout_factory=_make_dropout,
      layer_norm_factory=_make_layer_norm,
      relative_position_bias_factory=_make_relative_position_bias,
      shared_relative_position_bias=shared_relative_position_bias,
      scanned=scanned)


class MoeArchitectureTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='default', extra_mlp=False),
      dict(testcase_name='extra_mlp', extra_mlp=True))
  def test_moe_architecture(self, extra_mlp):
    batch_size = 2
    seq_length = 4
    num_embeddings = 64

    encoder_factory = functools.partial(
        moe_architecture.SparseEncoder,
        num_layers=2,
        num_sparse_layers=1,
        sparse_layout=LayerLayout.MIXED,
        sparse_layer_factory=functools.partial(
            _make_sparse_encoder_layer, extra_mlp=extra_mlp),
        layer_factory=_make_dense_encoder_layer,
        input_dropout_factory=_make_dropout,
        output_dropout_factory=_make_dropout,
        layer_norm_factory=_make_layer_norm,
    )
    decoder_factory = functools.partial(
        moe_architecture.SparseDecoder,
        num_layers=2,
        num_sparse_layers=1,
        sparse_layout=LayerLayout.MIDDLE,
        sparse_layer_factory=functools.partial(
            _make_sparse_decoder_layer, extra_mlp=extra_mlp),
        layer_factory=_make_dense_decoder_layer,
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

    encoder_layer_0 = variables['params']['encoder']['layers_0']
    # First encoder layer should have regular self-attention.
    self.assertIn('attention', encoder_layer_0)
    # And it should contain a dense MLP (because there is only one sparse MLP
    # layer at the top for MIXED layout).
    self.assertIn('wi', encoder_layer_0['mlp'])  # Dense
    self.assertIn('wo', encoder_layer_0['mlp'])  # Dense
    self.assertNotIn('expert', encoder_layer_0['mlp'])  # Sparse

    encoder_layer_1 = variables['params']['encoder']['layers_1']
    # Second encoder layer should have regular self-attention.
    self.assertIn('attention', encoder_layer_1)
    # And it should contain a sparse MoeLayer (from sparse MIXED layout).
    self.assertIn('expert', encoder_layer_1['mlp'])  # Sparse
    if extra_mlp:
      # Check that the additional mlp blocks are added to the sparse layers.
      self.assertIn('extra_mlp', encoder_layer_1)

    decoder_layer_0 = variables['params']['decoder']['layers_0']
    # First decoder layer should have regular encoder-decoder attention.
    self.assertIn('query', decoder_layer_0['encoder_decoder_attention'])
    # It should contain regular self-attention.
    self.assertIn('self_attention', decoder_layer_0)
    # And it should contain a regular dense MLP (sparse layout is MIDDLE, so
    # first layer will be dense MLP).
    self.assertIn('wi', decoder_layer_0['mlp'])  # Dense
    self.assertIn('wo', decoder_layer_0['mlp'])  # Dense
    self.assertNotIn('expert', decoder_layer_0['mlp'])  # Sparse

    decoder_layer_1 = variables['params']['decoder']['layers_1']
    # Second decoder layer should have regular encoder-decoder attention.
    self.assertIn('query', decoder_layer_1['encoder_decoder_attention'])
    # It should contain regular self-attention.
    self.assertIn('self_attention', decoder_layer_1)
    # And it should contain a sparse MLP (sparse layout is MIDDLE, which for a
    # 2 layer decoder results in top layer being sparse).
    self.assertIn('expert', decoder_layer_1['mlp'])  # Sparse
    if extra_mlp:
      # Check that the additional mlp blocks are added to the sparse layers.
      self.assertIn('extra_mlp', decoder_layer_1)

  def test_moe_architecture_logit_mask_propagates(self):
    batch_size = 2
    seq_length = 4
    num_embeddings = 5

    encoder_factory = functools.partial(
        moe_architecture.SparseEncoder,
        num_layers=1,
        num_sparse_layers=1,
        sparse_layout=LayerLayout.MIXED,
        sparse_layer_factory=functools.partial(
            _make_sparse_encoder_layer, ignore_padding_tokens=True),
        layer_factory=_make_dense_encoder_layer,
        input_dropout_factory=_make_dropout,
        output_dropout_factory=_make_dropout,
        layer_norm_factory=_make_layer_norm,
    )
    decoder_factory = functools.partial(
        moe_architecture.SparseDecoder,
        num_layers=1,
        num_sparse_layers=1,
        sparse_layout=LayerLayout.MIDDLE,
        sparse_layer_factory=functools.partial(
            _make_sparse_decoder_layer, ignore_padding_tokens=True),
        layer_factory=_make_dense_decoder_layer,
        dropout_factory=_make_dropout,
        layer_norm_factory=_make_layer_norm,
    )

    transformer = t5_architecture.EncoderDecoder(
        shared_token_embedder_factory=functools.partial(
            _make_token_emb, num_embeddings=num_embeddings),
        encoder_factory=encoder_factory,
        decoder_factory=decoder_factory,
    )

    encoder_input_tokens = jax.random.randint(
        jax.random.PRNGKey(0), (batch_size, seq_length), minval=0, maxval=4)
    decoder_input_tokens = jax.random.randint(
        jax.random.PRNGKey(1), (batch_size, seq_length), minval=0, maxval=4)
    decoder_target_tokens = jax.random.randint(
        jax.random.PRNGKey(2), (batch_size, seq_length), minval=0, maxval=4)
    output, _ = transformer.init_with_output(
        {
            'params': random.PRNGKey(0),
            'dropout': random.PRNGKey(0)
        },
        encoder_input_tokens=encoder_input_tokens,
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        enable_dropout=False,
    )

    # To identify padding vs non-padding tokens in MoE models, we rely on how
    # the Flaxformer T5 architectures constructs and applies the logit mask.
    # See
    # https://github.com/google/flaxformer/blob/9712a16/flaxformer/architectures/t5/t5_architecture.py#L315
    # and
    # https://github.com/google/flaxformer/blob/9712a16/flaxformer/architectures/t5/t5_architecture.py#L603.
    # So here, we test the specific outputs to catch any changes to the
    # underlying T5 architecture logic.
    np.testing.assert_allclose(
        output, [
            [
                [1.0098116, -0.07620703, 2.2726512, 1.5326656, -0.56177557],
                [-0.94259655, 1.7607443, 2.3942919, 0.72459084, -1.0425543],
                [0.02322888, 0.65254104, 4.597586, 1.4008591, -0.45962948],
                [-0.94259655, 1.7607443, 2.3942919, 0.72459084, -1.0425543],
            ],
            [
                [0.01904473, 0.6017947, 4.61797, 1.3143052, -0.33733633],
                [0.04458817, 0.12540922, 2.4217446, 3.548073, 0.06769713],
                [0.01904476, 0.6017947, 4.61797, 1.3143052, -0.33733624],
                [0., 0., 0., 0., 0.],
            ],
        ],
        rtol=1e-6,
        atol=1e-6)

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
        layer_factory=_make_dense_encoder_layer,
        input_dropout_factory=_make_dropout,
        output_dropout_factory=_make_dropout,
        layer_norm_factory=_make_layer_norm,
    )

    decoder_factory = functools.partial(
        moe_architecture.SparseDecoder,
        num_layers=2,
        num_sparse_layers=2,
        sparse_layout=LayerLayout.MIDDLE,
        sparse_layer_factory=_make_sparse_decoder_layer,
        layer_factory=_make_dense_decoder_layer,
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
    encoder = variables['params']['encoder']
    for layer in ['layers_0', 'layers_1']:
      encoder_layer = encoder[layer]
      # All encoder layers should have regular attention.
      self.assertIn('attention', encoder_layer)
      # And dense MLPs.
      self.assertIn('wi', encoder_layer['mlp'])  # Dense
      self.assertIn('wo', encoder_layer['mlp'])  # Dense
      self.assertNotIn('expert', encoder_layer['mlp'])  # Sparse

    # For degenerate cases, we only have one underlying Decoder.
    decoder = variables['params']['decoder']
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

  def test_scan_architecture(self):
    batch_size = 2
    seq_length = 4
    num_embeddings = 9

    encoder_factory = functools.partial(
        moe_architecture.SparseEncoder,
        num_layers=4,
        num_sparse_layers=2,
        sparse_layout=LayerLayout.MIXED,
        # Individual layers in MoE models are never scanned; only blocks.
        sparse_layer_factory=functools.partial(
            _make_sparse_encoder_layer, scanned=False),
        layer_factory=functools.partial(
            _make_dense_encoder_layer, scanned=False),
        input_dropout_factory=_make_dropout,
        output_dropout_factory=_make_dropout,
        layer_norm_factory=_make_layer_norm,
    )
    decoder_factory = functools.partial(
        moe_architecture.SparseDecoder,
        num_layers=3,
        num_sparse_layers=3,
        sparse_layout=LayerLayout.MIXED,
        # Individual layers in MoE models are never scanned; only blocks.
        sparse_layer_factory=functools.partial(
            _make_sparse_decoder_layer, scanned=False),
        layer_factory=functools.partial(
            _make_dense_decoder_layer, scanned=False),
        dropout_factory=_make_dropout,
        layer_norm_factory=_make_layer_norm,
    )

    scanned_transformer = t5_architecture.EncoderDecoder(
        scan_layers=True,
        shared_token_embedder_factory=functools.partial(
            _make_token_emb, num_embeddings=num_embeddings),
        encoder_factory=functools.partial(encoder_factory, scan_layers=True),
        decoder_factory=functools.partial(decoder_factory, scan_layers=True),
    )

    encoder_input_tokens = np.zeros((batch_size, seq_length), dtype=np.float32)
    decoder_input_tokens = np.zeros((batch_size, seq_length), dtype=np.float32)
    decoder_target_tokens = np.zeros((batch_size, seq_length), dtype=np.float32)

    output, variables = scanned_transformer.init_with_output(
        {
            'params': random.PRNGKey(0),
            'dropout': random.PRNGKey(0)
        },
        encoder_input_tokens=encoder_input_tokens,
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        enable_dropout=False)

    with self.subTest(name='init_with_output'):
      self.assertEqual(output.shape, (batch_size, seq_length, num_embeddings))

      # Verify model keys and param shapes for scan.
      self.assertEqual(
          jax.tree.map(jnp.shape, variables['params']['encoder']),
          flax.core.FrozenDict({
              'encoder': {
                  'subblock_0': {
                      'attention': {
                          'key': {
                              # (emb_dim, scan_dim, qkv_features // num_heads)
                              'kernel': (13, 2, 2),
                          },
                          'out': {
                              # (qkv_features, scan_dim, emb_dim)
                              'kernel': (16, 2, 13),
                          },
                          'query': {
                              'kernel': (13, 2, 16),
                          },
                          'value': {
                              'kernel': (13, 2, 2),
                          },
                      },
                      'mlp': {
                          'wi': {
                              # (emb_dim, scan_dim, dense MLP dim)
                              'kernel': (13, 2, 8),
                          },
                          'wo': {
                              'kernel': (8, 2, 13),
                          },
                      },
                      'pre_attention_layer_norm': {
                          'scale': (13, 2),  # (emb_dim, scan_dim)
                      },
                      'pre_mlp_layer_norm': {
                          'scale': (13, 2),
                      },
                      'relpos_bias': {
                          'rel_embedding': (8, 2, 32),
                      },
                  },
                  'subblock_1': {
                      'attention': {
                          'key': {
                              'kernel': (13, 2, 2),
                          },
                          'out': {
                              'kernel': (16, 2, 13),
                          },
                          'query': {
                              'kernel': (13, 2, 16),
                          },
                          'value': {
                              'kernel': (13, 2, 2),
                          },
                      },
                      'mlp': {
                          'expert': {
                              'wi': {
                                  'bias': (4, 2, 16),
                                  # (num_experts, scan_dim, emb_dim, MoE MLP)
                                  'kernel': (4, 2, 13, 16),
                              },
                              'wo': {
                                  'bias': (4, 2, 13),
                                  'kernel': (4, 2, 16, 13),
                              },
                          },
                          'router': {
                              'router_weights': {
                                  'w': {
                                      'bias': (4, 2),
                                      # (emb_dim, scan_dim, num_experts)
                                      'kernel': (13, 2, 4),
                                  },
                              },
                          },
                      },
                      'pre_attention_layer_norm': {
                          'scale': (13, 2),
                      },
                      'pre_mlp_layer_norm': {
                          'scale': (13, 2),
                      },
                      'relpos_bias': {
                          'rel_embedding': (8, 2, 32),
                      },
                  },
              },
              'encoder_norm': {
                  'scale': (13,),
              },
          }))

      self.assertEqual(
          jax.tree.map(jnp.shape, variables['params']['decoder']),
          flax.core.FrozenDict({
              'decoder': {
                  'subblock_0': {
                      'encoder_decoder_attention': {
                          'key': {
                              # (emb_dim, scan_dim, qkv_features // num_heads)
                              'kernel': (13, 3, 2),
                          },
                          'out': {
                              # (qkv_features, scan_dim, emb_dim)
                              'kernel': (16, 3, 13),
                          },
                          'query': {
                              'kernel': (13, 3, 16),
                          },
                          'value': {
                              'kernel': (13, 3, 2),
                          },
                      },
                      'mlp': {
                          'expert': {
                              'wi': {
                                  'bias': (4, 3, 16),
                                  # (num_experts, scan_dim, emb_dim, MoE MLP)
                                  'kernel': (4, 3, 13, 16),
                              },
                              'wo': {
                                  'bias': (4, 3, 13),
                                  'kernel': (4, 3, 16, 13),
                              },
                          },
                          'router': {
                              'router_weights': {
                                  'w': {
                                      'bias': (4, 3),
                                      # (emb_dim, scan_dim, num_experts)
                                      'kernel': (13, 3, 4),
                                  },
                              },
                          },
                      },
                      'pre_cross_attention_layer_norm': {
                          # (emb_dim, scan_dim)
                          'scale': (13, 3),
                      },
                      'pre_mlp_layer_norm': {
                          'scale': (13, 3),
                      },
                      'pre_self_attention_layer_norm': {
                          'scale': (13, 3),
                      },
                      'relpos_bias': {
                          'rel_embedding': (8, 3, 32),
                      },
                      'self_attention': {
                          'key': {
                              'kernel': (13, 3, 2),
                          },
                          'out': {
                              # (qkv_features, scan_dim, emb_dim)
                              'kernel': (16, 3, 13),
                          },
                          'query': {
                              'kernel': (13, 3, 16),
                          },
                          'value': {
                              'kernel': (13, 3, 2),
                          },
                      },
                  },
              },
              'decoder_norm': {
                  'scale': (13,),
              },
          }))

    with self.subTest(name='sow_intermediates'):
      _, modified_variables = scanned_transformer.apply(
          {'params': variables['params']},
          encoder_input_tokens=encoder_input_tokens,
          decoder_input_tokens=decoder_input_tokens,
          decoder_target_tokens=decoder_target_tokens,
          enable_dropout=False,
          mutable=['intermediates'])
      intermediates = modified_variables['intermediates']

      for metric in MOE_METRICS:
        self.assertIn(metric,
                      intermediates['encoder']['encoder']['subblock_1']['mlp'])
        self.assertIn(metric,
                      intermediates['decoder']['decoder']['subblock_0']['mlp'])

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

  def test_scan_block_factory(self):
    scan_block_factory = functools.partial(
        moe_architecture._scan_block_factory,
        dense_layer_factory=_make_dense_encoder_layer,
        sparse_layer_factory=_make_sparse_encoder_layer)

    with self.subTest(name='degenerate_dense'):
      degenerate_dense_block = scan_block_factory(
          num_layers=12, num_sparse_layers=0, sparse_layout=LayerLayout.MIXED)
      self.assertLen(degenerate_dense_block, 1)
      self.assertIsInstance(degenerate_dense_block[0], EncoderLayer)

    with self.subTest(name='degenerate_sparse'):
      degenerate_sparse_block = scan_block_factory(
          num_layers=12, num_sparse_layers=12, sparse_layout=LayerLayout.TOP)
      self.assertLen(degenerate_sparse_block, 1)
      self.assertIsInstance(degenerate_sparse_block[0], SparseEncoderLayer)

    with self.subTest(name='unsupported_layouts'):
      for layer_layout in [
          LayerLayout.BOTTOM, LayerLayout.MIDDLE, LayerLayout.TOP
      ]:
        with self.assertRaisesRegex(ValueError,
                                    'Scan is only supported for MIXED'):
          scan_block_factory(
              num_layers=12, num_sparse_layers=6, sparse_layout=layer_layout)

    with self.subTest(name='mixed_layouts'):
      block = scan_block_factory(
          num_layers=12, num_sparse_layers=3, sparse_layout=LayerLayout.MIXED)
      self.assertLen(block, 4)
      for sublock in range(3):
        self.assertIsInstance(block[sublock], EncoderLayer)
      self.assertIsInstance(block[-1], SparseEncoderLayer)

  def test_num_scan_blocks(self):
    self.assertEqual(
        moe_architecture._num_scan_blocks(
            num_layers=24, num_sparse_layers=3,
            sparse_layout=LayerLayout.MIXED), 3)

if __name__ == '__main__':
  absltest.main()
