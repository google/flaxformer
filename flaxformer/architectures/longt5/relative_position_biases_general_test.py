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

"""Tests for relative_position_biases_general.

The parameter names and outputs should be identical to the original "dense"
`RelativePositionBiases` when we use `full_att_rp_bucket()`.
"""
from absl.testing import absltest
import jax
from jax import random
import jax.numpy as jnp

from flaxformer.architectures.longt5 import relative_position_biases_general as rp_biases_general


class RelativePositionBiasesGeneralTest(absltest.TestCase):

  def setUp(self):
    self.num_heads = 3
    self.query_len = 5
    self.key_len = 7
    self.relative_attention = rp_biases_general.RelativePositionBiasesGeneral(
        num_buckets=12,
        max_distance=10,
        num_heads=3,
        dtype=jnp.float32,
    )
    super().setUp()

  def test_relative_attention_bidirectional_params(self):
    """Tests that bidirectional relative position biases have expected params."""
    rp_bucket = self.relative_attention.full_att_rp_bucket(
        self.query_len, self.key_len, bidirectional=True)
    params = self.relative_attention.init(
        random.PRNGKey(0), rp_bucket, mutable=['params'])
    param_shapes = jax.tree.map(lambda x: x.shape, params)
    self.assertEqual(param_shapes, {
        'params': {
            'rel_embedding': (3, 12),
        },
    })

  def test_regression_relative_attention_bidirectional_values(self):
    """Tests that bidirectional relative position biases match expected values.

    """
    rp_bucket = self.relative_attention.full_att_rp_bucket(
        self.query_len, self.key_len, bidirectional=True)
    outputs, unused_params = self.relative_attention.init_with_output(
        random.PRNGKey(0), rp_bucket)
    self.assertEqual(outputs.shape,
                     (1, self.num_heads, self.query_len, self.key_len))

    self.assertAlmostEqual(outputs[0, 0, 0, 0], -0.1094, places=5)
    self.assertAlmostEqual(outputs[0, 1, 2, 1], -0.22087, places=5)
    self.assertAlmostEqual(outputs[0, 1, 4, 6], 0.27360, places=5)
    self.assertAlmostEqual(outputs[0, 2, 4, 6], -0.31798, places=5)

  def test_relative_attention_unidirectional_params(self):
    """Tests that unidirectional relative position biases have expected params."""
    rp_bucket = self.relative_attention.full_att_rp_bucket(
        self.query_len, self.key_len, bidirectional=False)
    params = self.relative_attention.init(
        random.PRNGKey(0), rp_bucket, mutable=['params'])
    param_shapes = jax.tree.map(lambda x: x.shape, params)
    self.assertEqual(param_shapes, {
        'params': {
            'rel_embedding': (3, 12),
        },
    })

  def test_regression_relative_attention_unidirectional_values(self):
    """Tests that unidirectional relative position biases match expected values.

    """
    rp_bucket = self.relative_attention.full_att_rp_bucket(
        self.query_len, self.key_len, bidirectional=False)
    outputs, unused_params = self.relative_attention.init_with_output(
        random.PRNGKey(0), rp_bucket)
    self.assertEqual(outputs.shape,
                     (1, self.num_heads, self.query_len, self.key_len))

    self.assertAlmostEqual(outputs[0, 0, 0, 0], -0.109404, places=5)
    self.assertAlmostEqual(outputs[0, 1, 2, 1], -0.220874, places=5)
    self.assertAlmostEqual(outputs[0, 1, 4, 6], -0.189960, places=5)
    self.assertAlmostEqual(outputs[0, 2, 4, 6], 0.366049, places=5)


if __name__ == '__main__':
  absltest.main()
