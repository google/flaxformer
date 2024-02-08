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

"""Tests for testing_utils."""

from absl.testing import absltest
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
import jax.numpy as jnp

from flaxformer import testing_utils


class TestingUtilsTest(absltest.TestCase):

  def test_format_params_shapes(self):
    result = testing_utils.format_params_shapes({"foo[bar]": ["baz", 1, 2, 3]})
    self.assertEqual(result, """{
  "foo[bar]": ["baz", 1, 2, 3]
}""")

  def test_param_dtypes_shapes_axes(self):
    params = nn.FrozenDict({
        "a": {
            "b": jnp.zeros([3, 7], dtype=jnp.float32),
            "c": {
                "d": jnp.zeros([9], dtype=jnp.float32),
            },
        },
        "b": {
            "c": jnp.zeros([3, 7, 4], dtype=jnp.float32),
        },
    })

    params_axes = nn.FrozenDict({
        "a": {
            "b_axes": nn_partitioning.AxisMetadata(names=("vocab", "embed")),
            "c": {
                "d_axes": nn_partitioning.AxisMetadata(names=("embed",)),
            },
        },
        "b": {
            "c_axes":
                nn_partitioning.AxisMetadata(names=("embed", "mlp", "output")),
        },
    })

    result = testing_utils.format_params_shapes(
        testing_utils.param_dtypes_shapes_axes(params, params_axes))

    self.assertEqual(
        result, """{
  "a": {
    "b": ["float32", "vocab=3", "embed=7"],
    "c": {
      "d": ["float32", "embed=9"]
    }
  },
  "b": {
    "c": ["float32", "embed=3", "mlp=7", "output=4"]
  }
}""")


if __name__ == "__main__":
  absltest.main()
