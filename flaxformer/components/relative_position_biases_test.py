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

"""Tests for relative_position_biases.

"""
from absl.testing import absltest
import jax
from jax import random
from jax import tree_util
import jax.numpy as jnp
import numpy as np

from flaxformer import sharding
from flaxformer import testing_utils
from flaxformer.components import relative_position_biases

expected_files = testing_utils.ExpectedJsonFiles(
    'flaxformer/components/testdata')


class RelativePositionBiasesTest(absltest.TestCase):

  def setUp(self):
    self.num_heads = 3
    self.query_len = 5
    self.key_len = 7
    self.relative_attention = relative_position_biases.RelativePositionBiases(
        num_buckets=12,
        max_distance=10,
        num_heads=3,
        dtype=jnp.float32,
    )
    super().setUp()

  def test_relative_attention_renamed_head_axis(self):
    """Tests that the head axis renaming is as expected."""
    self.relative_attention = relative_position_biases.RelativePositionBiases(
        num_buckets=12,
        max_distance=10,
        num_heads=3,
        dtype=jnp.float32,
        head_axis_name='relpos_heads')
    variables = self.relative_attention.init(
        random.PRNGKey(0), self.query_len, self.key_len)
    sharding.check_params_and_axis_names_match(variables)
    for axis_names in tree_util.tree_leaves(sharding.get_axis_names(variables)):
      for axis_name in axis_names:
        self.assertIn(axis_name, {'relpos_heads', 'relpos_buckets'})
    expected_files.check_params_and_axes(variables['params'],
                                         variables['params_axes'],
                                         'relpos_bias_renamed_head_axis.json')

  def test_relative_attention_bidirectional_params(self):
    """Tests that bidirectional relative position biases have expected params."""
    params = self.relative_attention.init(
        random.PRNGKey(0),
        self.query_len,
        self.key_len,
        bidirectional=True,
        mutable=['params'])
    param_shapes = jax.tree.map(lambda x: x.shape, params)
    self.assertEqual(param_shapes, {
        'params': {
            'rel_embedding': (3, 12),
        },
    })

  def test_regression_relative_attention_bidirectional_values(self):
    """Tests that bidirectional relative position biases match expected values."""

    outputs, unused_params = self.relative_attention.init_with_output(
        random.PRNGKey(0), self.query_len, self.key_len, bidirectional=True)
    self.assertEqual(outputs.shape,
                     (1, self.num_heads, self.query_len, self.key_len))
    self.assertAlmostEqual(outputs[0, 0, 0, 0], -0.10940, places=5)
    self.assertAlmostEqual(outputs[0, 1, 2, 1], -0.22087, places=5)
    self.assertAlmostEqual(outputs[0, 1, 4, 6], 0.27360, places=5)
    self.assertAlmostEqual(outputs[0, 2, 4, 6], -0.31798, places=5)

  def test_relative_attention_unidirectional_params(self):
    """Tests that unidirectional relative position biases have expected params."""
    params = self.relative_attention.init(
        random.PRNGKey(0),
        self.query_len,
        self.key_len,
        bidirectional=False,
        mutable=['params'])
    param_shapes = jax.tree.map(lambda x: x.shape, params)
    self.assertEqual(param_shapes, {
        'params': {
            'rel_embedding': (3, 12),
        },
    })

  def test_regression_relative_attention_unidirectional_values(self):
    """Tests that unidirectional relative position biases match expected values.

    """
    outputs, unused_params = self.relative_attention.init_with_output(
        random.PRNGKey(0), self.query_len, self.key_len, bidirectional=False)
    self.assertEqual(outputs.shape,
                     (1, self.num_heads, self.query_len, self.key_len))

    self.assertAlmostEqual(outputs[0, 0, 0, 0], -0.10940, places=5)
    self.assertAlmostEqual(outputs[0, 1, 2, 1], -0.22087, places=5)
    self.assertAlmostEqual(outputs[0, 1, 4, 6], -0.18996, places=5)
    self.assertAlmostEqual(outputs[0, 2, 4, 6], 0.3660492, places=5)

  def test_relative_attention_decode_cache_error_with_init(self):
    """Tests that relative embedding init fails with decode == True."""
    with self.assertRaisesRegex(
        ValueError,
        'decode-mode cannot be enabled during init. use model.apply to '
        'initialize the decoding cache.'):
      self.relative_attention.init(
          jax.random.PRNGKey(0),
          self.query_len,
          self.key_len,
          bidirectional=False,
          decode=True)

  def test_relative_attention_decode_cache_errror_with_bidirectional(self):
    """Tests that bidirectional relative embeddings fails when decoding."""
    params = self.relative_attention.init(
        jax.random.PRNGKey(0),
        self.query_len,
        self.key_len,
        bidirectional=False,
        decode=False)

    with self.assertRaisesRegex(
        ValueError,
        'bidirectional RelativePositionBiases are not supported when decode=True.'
    ):
      self.relative_attention.apply(
          params,
          self.query_len,
          self.key_len,
          bidirectional=True,
          decode=True,
          mutable=['cache'])

  def test_relative_attention_decode_cache(self):
    """Tests that relative embeddings are correctly cached when decode=True."""

    params = self.relative_attention.init(
        jax.random.PRNGKey(0),
        self.query_len,
        self.key_len,
        bidirectional=False,
        decode=False)

    # during init, cache is not actually initialized.
    self.assertNotIn('cache', params)

    outputs, state = self.relative_attention.apply(
        params,
        self.query_len,
        self.key_len,
        bidirectional=False,
        decode=True,
        mutable=['cache'])

    self.assertEqual(outputs.shape,
                     (1, self.num_heads, self.query_len, self.key_len))

    self.assertIn('cached_bias', state['cache'])

    cached_bias = state['cache']['cached_bias']

    self.assertAlmostEqual(outputs[0, 0, 0, 0], -0.10940, places=5)
    self.assertAlmostEqual(outputs[0, 1, 2, 1], -0.22087, places=5)
    self.assertAlmostEqual(outputs[0, 1, 4, 6], -0.18996, places=5)
    self.assertAlmostEqual(outputs[0, 2, 4, 6], 0.3660492, places=5)

    np.testing.assert_array_equal(outputs, state['cache']['cached_bias'])

    params_with_cache = {
        **params,
        **state,
    }

    outputs, state = self.relative_attention.apply(
        params_with_cache,
        self.query_len,
        self.key_len,
        bidirectional=False,
        decode=True,
        mutable=['cache'])

    np.testing.assert_array_equal(cached_bias, state['cache']['cached_bias'])



if __name__ == '__main__':
  absltest.main()
