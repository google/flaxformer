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

"""Tests for hierarchical_relative_position_bias.py."""

from absl.testing import absltest
from absl.testing import parameterized
from jax import random

from flaxformer import testing_utils
from flaxformer.architectures.h_transformer import hierarchical_relative_position_bias as h_rpb


class RpbTest(parameterized.TestCase):
  """Test cases for HierarchicalRelativePositionBias."""

  def setUp(self):
    super().setUp()
    self.num_head = 2
    self.num_cluster = 4

  @parameterized.named_parameters(
      ('left_block', -1),
      ('right_block', 1),
      ('mid_block', 0),
  )
  def test_rpb_1d(self, block_coord: int):
    rpb_1d_module = h_rpb.OneDimHierarchicalRelativePositionBias(
        num_cluster=self.num_cluster,
        num_head=self.num_head,
    )
    rng = random.PRNGKey(0)
    result, variables = rpb_1d_module.init_with_output(rng, block_coord)
    expected_shape = (1, 1, self.num_cluster, self.num_cluster, self.num_head)
    self.assertEqual(result.shape, expected_shape)

    expected_positions = f'relpos_buckets={self.num_cluster*4-1}'
    expected_heads = f'heads={self.num_head}'
    expected_1d_rpb = {
        '1d_relative_position_bias': [
            'float32', expected_positions, expected_heads
        ]
    }
    self.assertDictEqual(
        testing_utils.param_dtypes_shapes_axes(variables['params'],
                                               variables['params_axes']),
        expected_1d_rpb)


if __name__ == '__main__':
  absltest.main()
