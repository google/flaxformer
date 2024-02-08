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

"""Tests for moe_parallel_fused_decoder."""

from absl.testing import absltest
import flax
from flax import linen as nn
import jax
from jax import numpy as jnp
from jax import random
import numpy as np

from flaxformer.architectures.moe import moe_layers
from flaxformer.architectures.moe import moe_parallel_fused_decoder
from flaxformer.architectures.moe import routing
from flaxformer.components import dense
from flaxformer.components import layer_norm
from flaxformer.components import relative_position_biases
from flaxformer.components.attention import dense_attention

FrozenDict = flax.core.FrozenDict
MoeLayer = moe_layers.MoeLayer
SparseParallelFusedDecoderLayer = (
    moe_parallel_fused_decoder.SparseParallelFusedDecoderLayer
)

EMBEDDING_INIT = nn.initializers.normal(stddev=1.0)
RELPOS_BIAS_INIT = nn.initializers.variance_scaling(1.0, 'fan_avg', 'uniform')
ATTENTION_KERNEL_INIT = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal'
)
MLP_KERNEL_INIT = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'truncated_normal'
)
BIAS_INIT = nn.initializers.zeros
DTYPE = jnp.float32

_make_dropout = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,))

_make_layer_norm = layer_norm.T5LayerNorm


def _make_multi_query_attention(
    embed_dim: int, num_heads: int, head_dim
) -> dense_attention.MultiQueryDotProductAttention:
  """Test configuration for attention."""
  return dense_attention.MultiQueryDotProductAttention(
      num_heads=num_heads,
      dtype=DTYPE,
      head_dim=head_dim,
      kernel_init=ATTENTION_KERNEL_INIT,
      bias_init=BIAS_INIT,
      use_bias=False,
      broadcast_dropout=True,
      float32_logits=True,
      rescale_logits=True,
      split_head_kernel=True,
      # Must be specified for constructing fused O-Wo projection.
      out_features=embed_dim,
      use_rotary_embedding=True,
      dropout_rate=0.1,
  )


def _make_relative_position_bias() -> (
    relative_position_biases.RelativePositionBiases
):
  """Test configuration for the position bias."""
  return relative_position_biases.RelativePositionBiases(
      num_buckets=32,
      max_distance=64,
      num_heads=8,
      dtype=DTYPE,
      embedding_init=RELPOS_BIAS_INIT,
  )


def _make_dense_mlp(embed_dim: int) -> dense.MlpBlock:
  """Test configuration for the MLP."""
  return dense.MlpBlock(
      use_bias=False,
      out_dim=embed_dim,  # Project back to embedding dimension
      activations=('swish', 'linear'),
      # MLP block only applies the activation functions.
      precomputed_intermediates=True,
      fuse_kernels=False,
      kernel_init=MLP_KERNEL_INIT,
      bias_init=BIAS_INIT,
      dtype=DTYPE,
  )


def _make_q_wi_fused_projection(
    attention_module: dense_attention.MultiQueryDotProductAttention,
    mlp_module: dense.MlpBlock,
) -> MoeLayer:
  """Returns sparse Q-Wi projection."""
  expert = dense.DenseGeneral(
      axis=-1,
      features=moe_parallel_fused_decoder.compute_fused_q_wi_dims(
          attention_module, mlp_module
      ),
      use_bias=False,
      dtype=DTYPE,
      kernel_init=ATTENTION_KERNEL_INIT,
      bias_init=BIAS_INIT,
      reshape_kernel=False,
      kernel_axis_names=('embed', 'heads', 'q_wi_fused'),
  )
  router = routing.TokensChooseMaskedRouter(
      # Default router weights fine for Q-Wi projection which takes 3D inputs:
      # [batch_size, seq_length, hidden_dim].
      router_weights=routing.RouterWeights(),
      num_selected_experts=1,
      jitter_noise=0.0,
      batch_prioritized_routing=False,
      dtype=DTYPE,
      ignore_padding_tokens=False,
  )
  return MoeLayer(
      num_experts=4,
      num_expert_partitions=4,
      router=router,
      max_group_size=2,
      train_capacity_factor=1.0,
      eval_capacity_factor=1.0,
      expert=expert,
      num_model_partitions=1,
      input_hidden_dims_axes=('embed',),
      output_hidden_dims_axes=('heads', 'mlp'),
      dtype=DTYPE,
  )


def _make_o_wo_fused_projection(
    attention_module: dense_attention.MultiQueryDotProductAttention,
) -> MoeLayer:
  """Returns sparse O-Wo projection."""
  expert = dense.DenseGeneral(
      axis=(-2, -1),
      features=moe_parallel_fused_decoder.compute_fused_o_wo_dims(
          attention_module
      ),
      use_bias=False,
      dtype=DTYPE,
      kernel_init=ATTENTION_KERNEL_INIT,
      bias_init=BIAS_INIT,
      reshape_kernel=False,
      # o_wo_fused = mlp//heads + head_dim.
      kernel_axis_names=('heads', 'o_wo_fused', 'embed'),
  )
  router = routing.TokensChooseMaskedRouter(
      # Specialized router weights required for O-Wo projection which takes 4D
      # inputs: [batch_size, seq_length, heads, o_wo_fused].
      router_weights=routing.RouterWeights(
          axis=(-2, -1),  # ('heads', 'o_wo_fused') projection
          kernel_axis_names=('heads', 'o_wo_fused', 'unmodeled'),
          reshape_kernel=False,
      ),
      num_selected_experts=1,
      jitter_noise=0.0,
      batch_prioritized_routing=False,
      dtype=DTYPE,
      ignore_padding_tokens=False,
  )
  return MoeLayer(
      num_experts=4,
      num_expert_partitions=4,
      router=router,
      max_group_size=2,
      train_capacity_factor=1.0,
      eval_capacity_factor=1.0,
      expert=expert,
      num_model_partitions=1,
      input_hidden_dims_axes=('heads', 'mlp'),
      output_hidden_dims_axes=('embed',),
      dtype=DTYPE,
  )


def _make_kv_fused_projection(
    attention_module: dense_attention.MultiQueryDotProductAttention,
) -> dense.DenseGeneral:
  """Returns dense KV projection."""
  return dense.DenseGeneral(
      axis=-1,
      features=moe_parallel_fused_decoder.compute_fused_kv_dims(
          attention_module
      ),
      use_bias=False,
      dtype=DTYPE,
      kernel_init=ATTENTION_KERNEL_INIT,
      bias_init=BIAS_INIT,
      reshape_kernel=False,
      kernel_axis_names=('embed', 'multiquery_heads', 'kv_fused'),
  )


class ParallelFusedDecoderOnlyTest(absltest.TestCase):

  def test_layer(self):
    batch_size = 3
    seq_length = 8
    embed_dim = 32
    num_heads = 8
    head_dim = 4

    attention_module = _make_multi_query_attention(
        embed_dim, num_heads, head_dim
    )
    mlp_module = _make_dense_mlp(embed_dim)

    decoder_layer = SparseParallelFusedDecoderLayer(
        self_attention=attention_module,
        mlp=mlp_module,
        # Attention and MLP modules are indirectly used to construct the fused
        # projections.
        q_wi_fused=_make_q_wi_fused_projection(attention_module, mlp_module),
        o_wo_fused=_make_o_wo_fused_projection(attention_module),
        kv_fused=_make_kv_fused_projection(attention_module),
        dropout_factory=_make_dropout,
        layer_norm_factory=_make_layer_norm,
        relative_position_bias_factory=_make_relative_position_bias,
    )

    decoder_target_tokens = np.zeros(
        (batch_size, seq_length, embed_dim), dtype=np.float32
    )
    output, variables = decoder_layer.init_with_output(
        {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(0)},
        targets=decoder_target_tokens,
        encoded=None,
        enable_dropout=True,
    )

    self.assertEqual(output.shape, (batch_size, seq_length, embed_dim))

    self.assertEqual(
        jax.tree_util.tree_map(jnp.shape, variables['params']),
        FrozenDict({
            'kv_fused': {
                # [embed_dim, 1, 2 * head_dim]
                'kernel': (32, 1, 8),
            },
            'layer_norm': {
                'scale': (32,),
            },
            'o_wo_fused': {
                'expert': {
                    # [num_experts, heads, mlp//heads + head_dim, embed_dim]
                    'kernel': (4, 8, 260, 32),
                },
                'router': {
                    'router_weights': {
                        'w': {
                            'bias': (4,),
                            # [heads, mlp//heads + head_dim, num_experts]
                            'kernel': (8, 260, 4),
                        },
                    },
                },
            },
            'q_wi_fused': {
                'expert': {
                    # [num_experts, embed_dim, num_heads,
                    #  mlp//heads * n_act + head_dim]
                    'kernel': (4, 32, 8, 516),
                },
                'router': {
                    'router_weights': {
                        'w': {
                            'bias': (4,),
                            # [embed_dim, num_experts]
                            'kernel': (32, 4),
                        },
                    },
                },
            },
            'relpos_bias': {
                'rel_embedding': (8, 32),
            },
        }),
    )

  def test_projection_dims(self):
    embed_dim = 8
    num_heads = 8
    head_dim = 2

    attention_module = _make_multi_query_attention(
        embed_dim, num_heads, head_dim
    )
    mlp_module = _make_dense_mlp(embed_dim)

    with self.subTest(name='fused_o_wo_dims'):
      self.assertEqual(
          attention_module.out_features,
          moe_parallel_fused_decoder.compute_fused_o_wo_dims(attention_module),
      )

    with self.subTest(name='fused_kv_dims'):
      self.assertEqual(
          (1, 2 * head_dim),
          moe_parallel_fused_decoder.compute_fused_kv_dims(attention_module),
      )

    with self.subTest(name='fused_q_wi_dims'):
      self.assertEqual(
          (num_heads, 514),
          moe_parallel_fused_decoder.compute_fused_q_wi_dims(
              attention_module, mlp_module
          ),
      )


if __name__ == '__main__':
  absltest.main()
