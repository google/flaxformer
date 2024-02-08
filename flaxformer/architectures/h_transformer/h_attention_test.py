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

"""Tests for h_attention.py."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen
from jax import random
import jax.numpy as jnp
import numpy as onp

from flaxformer import testing_utils
from flaxformer.architectures.h_transformer import h_attention
from flaxformer.architectures.h_transformer import token_hierarchy as th


class HAttention1DTest(parameterized.TestCase):
  """Test cases for h_attention."""

  def setUp(self):
    super().setUp()
    self.batch_size = 2
    self.num_heads = 4
    self.head_dim = 2
    self.feature_size = self.num_heads * self.head_dim

  def test_bad_input_shape(self):
    # Delibrately sets a wrong shape here to trigger the ValueError.
    inputs_q = jnp.ones((1, 1, 8, 2))
    with self.assertRaises(ValueError):
      attention_module = h_attention.OneDimEncoderSelfAttention(
          num_heads=self.num_heads, num_clusters=2, use_rpb=True)
      rng = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
      attention_module.init(rng, inputs_q, padding_mask=None)

  def test_bad_padding_mask_shape(self):
    # Delibrately sets a wrong shape here to trigger the ValueError.
    inputs_q = jnp.ones((self.batch_size, 16, self.feature_size))
    padding_mask = jnp.ones((self.batch_size, 16, 1, 1))
    with self.assertRaises(ValueError):
      attention_module = h_attention.OneDimEncoderSelfAttention(
          num_heads=self.num_heads, num_clusters=2, use_rpb=True)
      rng = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
      attention_module.init(rng, inputs_q, padding_mask=padding_mask)

  def test_large_num_clusters(self):
    # Delibrately sets num_clusters > sequence_length//2. This used to trigger
    # a bug. It has been fixed. So this should always pass.
    num_clusters = 16
    seq_len = 4
    inputs_q = jnp.ones((self.batch_size, seq_len, self.feature_size))
    rng = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    attention_module = h_attention.OneDimEncoderSelfAttention(
        num_heads=self.num_heads, num_clusters=num_clusters, use_rpb=True)
    result, _ = attention_module.init_with_output(
        rng, inputs_q, padding_mask=None)
    expected_shape = inputs_q.shape
    self.assertEqual(result.shape, expected_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name='decoder_no_rpb_singlehead',
          use_rpb=False,
          use_multihead_rpb=False),
      dict(
          testcase_name='decoder_rpb_singlehead',
          use_rpb=True,
          use_multihead_rpb=False),
      dict(
          testcase_name='decoder_rpb_multihead',
          use_rpb=True,
          use_multihead_rpb=True),
  )
  def test_decoder_runs(self, use_rpb, use_multihead_rpb):
    num_clusters = 2
    num_level = 4
    num_block = int(onp.exp2(num_level))
    seq_len = num_clusters * num_block
    inputs_q = jnp.ones((self.batch_size, seq_len, self.feature_size))
    rng = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    attention_module = h_attention.OneDimDecoderSelfAttention(
        num_heads=self.num_heads,
        num_clusters=num_clusters,
        use_rpb=use_rpb,
        use_multihead_rpb=use_multihead_rpb)
    result, _ = attention_module.init_with_output(
        rng, inputs_q, padding_mask=None)
    expected_shape = inputs_q.shape
    self.assertEqual(result.shape, expected_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name='encoder_no_rpb_singlehead',
          use_rpb=False,
          use_multihead_rpb=False),
      dict(
          testcase_name='encoder_rpb_singlehead',
          use_rpb=True,
          use_multihead_rpb=False),
      dict(
          testcase_name='encoder_rpb_multihead',
          use_rpb=True,
          use_multihead_rpb=True),
  )
  def test_encoder_runs(self, use_rpb, use_multihead_rpb):
    num_clusters = 2
    num_level = 4
    num_block = int(onp.exp2(num_level))
    seq_len = num_clusters * num_block
    inputs_q = jnp.ones((self.batch_size, seq_len, self.feature_size))
    rng = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    attention_module = h_attention.OneDimEncoderSelfAttention(
        num_heads=self.num_heads,
        num_clusters=num_clusters,
        use_rpb=use_rpb,
        use_multihead_rpb=use_multihead_rpb)
    result, _ = attention_module.init_with_output(
        rng, inputs_q, padding_mask=None)
    expected_shape = inputs_q.shape
    self.assertEqual(result.shape, expected_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name='cross_attention_no_rpb_singlehead',
          use_rpb=False,
          use_multihead_rpb=False),
      dict(
          testcase_name='cross_attention_rpb_singlehead',
          use_rpb=True,
          use_multihead_rpb=False),
      dict(
          testcase_name='cross_attention_rpb_multihead',
          use_rpb=True,
          use_multihead_rpb=True),
  )
  def test_cross_attention_runs(self, use_rpb, use_multihead_rpb):
    num_clusters = 2
    num_level = 4
    num_block = int(onp.exp2(num_level))
    seq_len = num_clusters * num_block
    inputs_q = jnp.ones((self.batch_size, seq_len, self.feature_size))
    rng = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    attention_module = h_attention.OneDimCrossAttention(
        num_heads=self.num_heads,
        num_clusters=num_clusters,
        use_rpb=use_rpb,
        use_multihead_rpb=use_multihead_rpb)
    result, _ = attention_module.init_with_output(rng, inputs_q, inputs_q)
    expected_shape = inputs_q.shape
    self.assertEqual(result.shape, expected_shape)

  def test_attention_params(self):
    num_block = 16
    num_clusters = 4
    seq_len = num_clusters * num_block
    inputs_q = jnp.ones((self.batch_size, seq_len, self.feature_size))
    rng = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    attention_module = h_attention.OneDimEncoderSelfAttention(
        num_heads=self.num_heads,
        num_clusters=num_clusters,
        use_rpb=True,
        use_multihead_rpb=True,
        split_head_kernel=True,
    )
    result, variables = attention_module.init_with_output(
        rng, inputs_q, padding_mask=None)
    expected_shape = inputs_q.shape
    self.assertEqual(result.shape, expected_shape)

    expected_embed = f'embed={self.feature_size}'
    expected_heads = f'heads={self.num_heads}'
    expected_kv = f'kv={self.head_dim}'
    expected_relpos = f'relpos_buckets={4*num_clusters - 1}'
    # The bias term does not have split head shape. The heads are always merged.
    expected_merged_kv = f'kv={self.feature_size}'
    expected_params = {
        'query_multihead_projection': {
            'bias': ['float32', expected_merged_kv],
            'kernel': ['float32', expected_embed, expected_heads, expected_kv],
        },
        'key_multihead_projection': {
            'bias': ['float32', expected_merged_kv],
            'kernel': ['float32', expected_embed, expected_heads, expected_kv],
        },
        'value_multihead_projection': {
            'bias': ['float32', expected_merged_kv],
            'kernel': ['float32', expected_embed, expected_heads, expected_kv],
        },
        'out': {
            'bias': ['float32', expected_embed],
            'kernel': ['float32', expected_merged_kv, expected_embed],
        },
        '1d_relative_position_bias': {
            '1d_relative_position_bias': [
                'float32', expected_relpos, expected_heads
            ],
        },
    }
    self.assertDictEqual(
        testing_utils.param_dtypes_shapes_axes(variables['params'],
                                               variables['params_axes']),
        expected_params)

  def test_self_attention_output(self):
    x = jnp.array([[
        0.08482573, 0.29561728, 0.33432317, 0.6481298, -0.7824855, 0.6298023,
        -0.3278767, -1.6607414
    ],
                   [
                       1.9097669, 1.1209449, -0.8260815, 1.0434877, -0.453946,
                       0.8152592, -1.1234418, 0.2729053
                   ]],
                  dtype=jnp.float32).T

    x = jnp.expand_dims(x, 0)
    # The H-similarity matrix computed by hand for the above input without
    # projections, using constant interpolation and coarsening:
    s = jnp.array([[
        3.654405017, 2.165819418, -1.549263898, 2.047796353, 0.2592372306,
        0.2592372306, -0.8335717156, -0.8335717156
    ],
                   [
                       2.165819418, 1.343907045, -0.8271601383, 1.361290584,
                       0.2592372306, 0.2592372306, -0.8335717156, -0.8335717156
                   ],
                   [
                       -1.549263898, -0.8271601383, 0.7941826266, -0.6453210751,
                       0.1133933598, -0.4629130414, -0.5346589167, -0.5346589167
                   ],
                   [
                       2.047796353, 1.361290584, -0.6453210751, 1.508938818,
                       -0.9808392381, 1.258906586, -0.5346589167, -0.5346589167
                   ],
                   [
                       0.2592372306, 0.2592372306, 0.1133933598, -0.9808392381,
                       0.8183505286, -0.8628948204, 0.7665406749, 1.175621795
                   ],
                   [
                       0.2592372306, 0.2592372306, -0.4629130414, 1.258906586,
                       -0.8628948204, 1.0612985, -1.122393763, -0.8234501969
                   ],
                   [
                       -0.8335717156, -0.8335717156, -0.5346589167,
                       -0.5346589167, 0.7665406749, -1.122393763, 1.369624608,
                       0.2379251883
                   ],
                   [
                       -0.8335717156, -0.8335717156, -0.5346589167,
                       -0.5346589167, 1.175621795, -0.8234501969, 0.2379251883,
                       2.8325393
                   ]])

    s = s / jnp.sqrt(2)
    a = linen.softmax(s, axis=1)
    target_out = a @ x[0]

    attn = h_attention.OneDimEncoderSelfAttention(
        num_heads=1,
        num_clusters=2,
        out_features=2,
        broadcast_dropout=False,
        dropout_rate=0.0,
        use_rpb=False,
        rescale_logits=True,
        use_mxu=True,
        interpolation_kernel_type=th.ConvKernelType.CONST,
        max_similarity_mode='scan_all',
        use_row_sum=False,
        multihead_projection=False,
        output_projection=False,
    )

    mask = jnp.ones((1, 8, 1))
    key = random.PRNGKey(0)
    variables = attn.init(key, x, mask)
    out = attn.apply(variables, x, mask)

    self.assertTrue(jnp.allclose(out, target_out, rtol=5e-5))


if __name__ == '__main__':
  absltest.main()
