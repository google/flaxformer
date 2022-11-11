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

"""Tests for h_attention.py."""

from absl.testing import absltest
from absl.testing import parameterized
from jax import random
import jax.numpy as jnp
import numpy as onp

from flaxformer import testing_utils
from flaxformer.architectures.h_transformer import h_attention


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
        split_head_kernel=True)
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


if __name__ == '__main__':
  absltest.main()
