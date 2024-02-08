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

"""Tests for scatter_utils."""

from absl.testing import absltest
from jax import numpy as jnp
import numpy as np

from flaxformer.architectures.moe import scatter_utils


class ScatterNdTest(absltest.TestCase):

  def test_scatter_nd_simple(self):
    indices = jnp.array([[0, 1]])
    updates = jnp.array([[1, -2, 3]], dtype=jnp.float32)

    actual_result = scatter_utils.scatter_nd(indices, updates, shape=(1, 2, 3))
    expected_result = jnp.array([[[0, 0, 0], [1, -2, 3]]], dtype=jnp.float32)
    np.testing.assert_allclose(actual_result, expected_result)

  def test_scatter_nd_3d_update(self):
    indices = jnp.array([[[0, 1], [1, 0], [1, 1]]])
    updates = jnp.array([[[1, -1], [2, -2], [3, -3]]], dtype=jnp.int32)

    actual_result = scatter_utils.scatter_nd(indices, updates, shape=(2, 2, 2))
    expected_result = jnp.array([[[0, 0], [1, -1]], [[2, -2], [3, -3]]],
                                dtype=jnp.int32)
    np.testing.assert_allclose(actual_result, expected_result)

  def test_scatter_nd_ignore_outside_indices(self):
    indices = jnp.array([[0, 0], [1, 2], [2, 0]])
    updates = jnp.array([1., 2., 3.])

    actual_result = scatter_utils.scatter_nd(indices, updates, shape=(3, 2))
    expected_result = jnp.array([[1., 0.], [0., 0], [3., 0.]])
    np.testing.assert_allclose(actual_result, expected_result)

  def test_scatter_nd_cumulative_updates(self):
    indices = jnp.array([[1, 1], [1, 1], [1, 1]])
    updates = jnp.array([1., 2., 3.])

    actual_result = scatter_utils.scatter_nd(indices, updates, shape=(3, 2))
    expected_result = jnp.array([[0., 0.], [0., 6.], [0., 0.]])
    np.testing.assert_allclose(actual_result, expected_result)


if __name__ == '__main__':
  absltest.main()
