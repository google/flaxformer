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

"""Tests for t5_architecture."""

from typing import Any

from absl.testing import absltest
from aqt.jax_legacy.jax import quantization as aqt
from flax import linen as nn
from jax import random
import jax.numpy as jnp
import numpy as np

from flaxformer import testing_utils
from flaxformer.architectures.t5 import parallel_fused_decoder
from flaxformer.architectures.t5 import t5_architecture_test_utils as t5_test_utils
from flaxformer.components import dense
from flaxformer.components import layer_norm
from flaxformer.components import relative_position_biases
from flaxformer.components.attention import dense_attention

expected_files = testing_utils.ExpectedJsonFiles(
    'flaxformer/architectures/t5/testdata'
)
check_params = expected_files.check_params_shapes_only
get_params = expected_files.get_params


class ParallelFusedDecoderOnlyTest(absltest.TestCase):

  def test_decoder_shapes_fused_parallel(self):
    """Tests if the decoder parameter have the expected shapes."""
    decoder = t5_test_utils.make_parallel_fused_transformer_config()
    inputs = np.array(
        [
            # Batch 1.
            [183, 20, 75],
            # Batch 2.
            [392, 19, 7],
        ],
        dtype=np.int32,
    )
    output, variables = decoder.init_with_output(
        random.PRNGKey(0),
        decoder_input_tokens=inputs,
        decoder_target_tokens=inputs,  # used for mask generation
        enable_dropout=False,
    )
    params = variables['params']
    reformatted = decoder.apply({}, params, method=decoder.to_save_format)
    check_params(reformatted, 'decoder_shapes_fused_parallel.json')
    self.assertEqual(output.shape, (2, 3, 4))

    # Convert back to Flax module structure format and test again.
    params2 = decoder.apply({}, reformatted, method=decoder.from_save_format)
    output2 = decoder.apply(
        {'params': params2},
        decoder_input_tokens=inputs,
        decoder_target_tokens=inputs,  # used for mask generation
        enable_dropout=False,
    )
    output = output.astype(np.float32)
    output2 = output2.astype(np.float32)
    np.testing.assert_allclose(output, output2, rtol=1e-8)

  def test_decoder_separate_init(self):
    """Tests if the decoder init can be controlled independently."""
    dtype = jnp.bfloat16
    num_attn_heads = 8
    num_features = 13
    make_dropout = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,))
    make_layer_norm = layer_norm.T5LayerNorm

    bias_init = nn.initializers.normal(stddev=1.0)

    def _make_mq_attention(num_attn_heads, dtype):
      """First test configuration for attention."""
      return dense_attention.MultiQueryDotProductAttention(
          num_heads=num_attn_heads,
          dtype=dtype,
          qkv_features=512,
          out_features=num_features,
          head_dim=None,
          kernel_init=nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'),
          bias_init=bias_init,
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
          kernel_init=nn.initializers.variance_scaling(
              1.0, 'fan_in', 'truncated_normal'
          ),
          bias_init=bias_init,
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
          embedding_init=nn.initializers.variance_scaling(
              1.0, 'fan_avg', 'uniform'
          ),
      )

    decoder_layer = parallel_fused_decoder.ParallelFusedDecoderLayer(
        self_attention=_make_mq_attention(num_attn_heads, dtype),
        mlp=_make_fusion_mlp(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        relative_position_bias_factory=(
            lambda: _make_relative_position_bias(num_attn_heads, dtype)
        ),
        use_aqt=False,
        weight_params=None,
        possibly_use_quantized_vars=False,
        is_quant_finetune_mode=False,
        q_wi_fused_kernel_init=nn.initializers.constant(0),
        kv_fused_kernel_init=nn.initializers.constant(1),
        o_wo_fused_kernel_init=nn.initializers.constant(2),
    )

    batch = 2
    seq_len = 4
    hidden_dim = 13
    inputs = np.ones((batch, seq_len, hidden_dim), dtype=np.float32)

    variables = decoder_layer.init(
        random.PRNGKey(0),
        targets=inputs,
        encoded=None,
        enable_dropout=False,
    )

    q_wi_fused = variables['params']['q_wi_fused']['kernel']
    kv_fused = variables['params']['kv_fused']['kernel']
    o_wo_fused = variables['params']['o_wo_fused']['kernel']

    np.testing.assert_allclose(
        q_wi_fused, np.zeros(q_wi_fused.shape), rtol=1e-8
    )
    np.testing.assert_allclose(kv_fused, np.ones(kv_fused.shape), rtol=1e-8)
    np.testing.assert_allclose(
        o_wo_fused, np.ones(o_wo_fused.shape) * 2, rtol=1e-8
    )

  def test_quantized_decoder_shapes_fused_parallel(self):
    """Tests if the decoder parameter have the expected shapes."""
    weight_params = aqt.QuantOps.WeightParams(
        prec=8, half_shift=False, axis=None
    )
    decoder = t5_test_utils.make_parallel_fused_transformer_config(
        use_aqt=True, weight_params=weight_params
    )
    inputs = np.array(
        [
            # Batch 1.
            [183, 20, 75],
            # Batch 2.
            [392, 19, 7],
        ],
        dtype=np.int32,
    )
    output, variables = decoder.init_with_output(
        random.PRNGKey(0),
        decoder_input_tokens=inputs,
        decoder_target_tokens=inputs,  # used for mask generation
        enable_dropout=False,
    )
    params = variables['params']
    reformatted = decoder.apply({}, params, method=decoder.to_save_format)
    check_params(reformatted, 'decoder_shapes_fused_parallel_quantized.json')
    self.assertEqual(output.shape, (2, 3, 4))

    # Convert back to Flax module structure format and test again.
    params2 = decoder.apply({}, reformatted, method=decoder.from_save_format)
    output2 = decoder.apply(
        {'params': params2},
        decoder_input_tokens=inputs,
        decoder_target_tokens=inputs,  # used for mask generation
        enable_dropout=False,
    )
    output = output.astype(np.float32)
    output2 = output2.astype(np.float32)
    np.testing.assert_allclose(output, output2, rtol=1e-8)

  def test_materialized_decoder_shapes_fused_parallel(self):
    """Tests if the decoder parameter have the expected shapes."""
    weight_params = aqt.QuantOps.WeightParams(
        prec=8, half_shift=False, axis=None
    )
    decoder = t5_test_utils.make_parallel_fused_transformer_config(
        use_aqt=True,
        weight_params=weight_params,
        possibly_use_quantized_vars=True,
        is_quant_finetune_mode=False,
    )
    inputs = np.array(
        [
            # Batch 1.
            [183, 20, 75],
            # Batch 2.
            [392, 19, 7],
        ],
        dtype=np.int32,
    )

    _, params = decoder.init_with_output(
        random.PRNGKey(0),
        decoder_input_tokens=inputs,
        decoder_target_tokens=inputs,  # used for mask generation
        enable_dropout=False,
    )

    self.assertDictEqual(
        testing_utils.param_dtypes_shapes_axes(
            params['params'], params['params_axes']
        ),
        get_params('decoder_params_axes_fused_parallel_quantized.json'),
    )

  def test_dense_general_factory_parallel_fused(self):
    dtype = jnp.bfloat16
    num_attn_heads = 8
    num_features = 13
    make_dropout = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,))
    make_layer_norm = layer_norm.T5LayerNorm

    bias_init = nn.initializers.normal(stddev=1.0)

    def _make_mq_attention(num_attn_heads, dtype):
      """First test configuration for attention."""
      return dense_attention.MultiQueryDotProductAttention(
          num_heads=num_attn_heads,
          dtype=dtype,
          qkv_features=512,
          out_features=num_features,
          head_dim=None,
          kernel_init=nn.initializers.variance_scaling(1.0, 'fan_in', 'normal'),
          bias_init=bias_init,
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
          kernel_init=nn.initializers.variance_scaling(
              1.0, 'fan_in', 'truncated_normal'
          ),
          bias_init=bias_init,
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
          embedding_init=nn.initializers.variance_scaling(
              1.0, 'fan_avg', 'uniform'
          ),
      )

    variables_arr = []
    for dense_general_cls_factory in [None, lambda: dense.DenseGeneral]:
      decoder_layer = parallel_fused_decoder.ParallelFusedDecoderLayer(
          self_attention=_make_mq_attention(num_attn_heads, dtype),
          mlp=_make_fusion_mlp(dtype),
          dropout_factory=make_dropout,
          layer_norm_factory=make_layer_norm,
          relative_position_bias_factory=(
              lambda: _make_relative_position_bias(num_attn_heads, dtype)
          ),
          use_aqt=False,
          weight_params=None,
          possibly_use_quantized_vars=False,
          is_quant_finetune_mode=False,
          q_wi_fused_kernel_init=nn.initializers.constant(0),
          kv_fused_kernel_init=nn.initializers.constant(1),
          o_wo_fused_kernel_init=nn.initializers.constant(2),
          dense_general_cls_factory=dense_general_cls_factory,
      )
      batch = 2
      seq_len = 4
      hidden_dim = 13
      inputs = np.ones((batch, seq_len, hidden_dim), dtype=np.float32)

      variables = decoder_layer.init(
          random.PRNGKey(0),
          targets=inputs,
          encoded=None,
          enable_dropout=False,
      )
      variables_arr.append(variables)

    np.testing.assert_allclose(
        variables_arr[0]['params']['q_wi_fused']['kernel'],
        variables_arr[1]['params']['q_wi_fused']['kernel'],
    )
    np.testing.assert_allclose(
        variables_arr[0]['params']['kv_fused']['kernel'],
        variables_arr[1]['params']['kv_fused']['kernel'],
    )
    np.testing.assert_allclose(
        variables_arr[0]['params']['o_wo_fused']['kernel'],
        variables_arr[1]['params']['o_wo_fused']['kernel'],
    )


if __name__ == '__main__':
  absltest.main()
