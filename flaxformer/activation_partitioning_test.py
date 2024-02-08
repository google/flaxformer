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

"""Tests for activation_partitioning."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax.linen import partitioning as flax_partitioning
from jax import numpy as jnp

from flaxformer import activation_partitioning


class ActivationPartitioningTest(parameterized.TestCase):

  def test_with_sharding_migration_dims_unset(self):
    x = jnp.array([1, 2, 3])
    with mock.patch.object(
        flax_partitioning,
        "with_sharding_constraint",
        autospec=True,
    ) as mock_wsc:
      activation_partitioning.with_sharding_migration(
          x=x, activation_partitioning_dims=None, logical_axis_names=("foo",))
    mock_wsc.assert_called_once_with(x, ("foo",))

  def test_with_sharding_migration_dims_1_axes_calls_new(self):
    x = jnp.array([1, 2, 3])
    with mock.patch.object(
        flax_partitioning, "get_axis_rules", return_value=["rule"]):
      with mock.patch.object(
          flax_partitioning,
          "with_sharding_constraint",
          autospec=True,
      ) as mock_wsc:
        activation_partitioning.with_sharding_migration(
            x=x, activation_partitioning_dims=1, logical_axis_names=("foo",))
      mock_wsc.assert_called_once_with(x, ("foo",))

  def test_with_sharding_migration_dims_2_errors(self):
    x = jnp.array([1, 2, 3])
    with mock.patch.object(
        flax_partitioning, "get_axis_rules", return_value=["rule"]):
      with self.assertRaisesRegex(ValueError, "rules.*dims.*present"):
        activation_partitioning.with_sharding_migration(
            x=x, activation_partitioning_dims=2, logical_axis_names=("foo",))

  @parameterized.parameters(1, 2)
  def test_with_sharding_migration_no_logical_axis_rules(self, dims):
    x = jnp.array([1, 2, 3])
    with mock.patch.object(
        flax_partitioning, "get_axis_rules", return_value=()):
      with mock.patch.object(
          activation_partitioning,
          "with_sharding",
          autospec=True,
      ) as mock_ws:
        activation_partitioning.with_sharding_migration(
            x=x, activation_partitioning_dims=dims, logical_axis_names=("foo",))
      mock_ws.assert_called_once_with(x, dims)


if __name__ == "__main__":
  absltest.main()
