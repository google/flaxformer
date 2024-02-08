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

"""Tests for sharding."""

from absl.testing import absltest
from flax import linen as nn
from flax.core import unfreeze
from flax.linen import partitioning
import jax
from jax import numpy as jnp

from flaxformer import sharding


class BasicModule(nn.Module):

  def setup(self):
    self.x = self.param("x", nn.initializers.ones, (1, 2))
    self.sow(
        "params_axes",
        "x_axes",
        sharding.axis_names("heads", "unmodeled"),
        reduce_fn=sharding.reduce_fn)

  def __call__(self, z):
    return self.x + z


class ShardingTest(absltest.TestCase):

  def test_axis_names(self):
    self.assertEqual(
        sharding.axis_names("embed", "unsharded"),
        partitioning.AxisMetadata(names=("embed", "unsharded")))

  def test_sowing_reduction(self):
    module = BasicModule()

    # Check the initial axes annotations.
    variables = module.init(jax.random.PRNGKey(0), jnp.array([[6, 7]]))
    self.assertDictEqual(
        unfreeze(variables["params_axes"]),
        {"x_axes": sharding.axis_names("heads", "unmodeled")},
    )

    # Re-run and make sure that axes are the same.
    _, variables = module.apply(variables, jnp.array([[6, 7]]), mutable=True)
    self.assertDictEqual(
        unfreeze(variables["params_axes"]),
        {"x_axes": sharding.axis_names("heads", "unmodeled")},
    )

  def test_check_params_and_axis_names_match_matches(self):
    sharding.check_params_and_axis_names_match(
        variables={
            "params": {
                "foo": {
                    "bar": jnp.array([1, 2, 3])
                }
            },
            "params_axes": {
                "foo": {
                    "bar_axes": sharding.axis_names("unsharded")
                }
            },
        })

  def test_check_params_and_axis_names_missing(self):
    with self.assertRaisesRegex(ValueError, ".*not sow.*foo/bar"):
      sharding.check_params_and_axis_names_match(variables={
          "params": {
              "foo": {
                  "bar": jnp.array([1, 2, 3])
              }
          },
          "params_axes": {},
      })

  def test_check_params_and_axis_names_wrong_size(self):
    with self.assertRaisesRegex(ValueError, ".*foo/bar.*doesn't.*for each"):
      sharding.check_params_and_axis_names_match(
          variables={
              "params": {
                  "foo": {
                      "bar": jnp.array([1, 2, 3])
                  }
              },
              "params_axes": {
                  "foo": {
                      "bar_axes":
                          sharding.axis_names("unsharded", "model", "vocab")
                  }
              },
          })

  def test_get_axis_names(self):
    variables = {
        "params": {
            "foo": {
                # Make sure the method is not distracted by actual parameters.
                "bar": jnp.array([1, 2, 3])
            }
        },
        "params_axes": {
            "foo": {
                "bar_axes": sharding.axis_names("unsharded", "model", "vocab")
            }
        },
    }
    self.assertEqual(
        sharding.get_axis_names(variables),
        {"foo": {
            "bar": sharding.AxisNames(("unsharded", "model", "vocab"))
        }})


if __name__ == "__main__":
  absltest.main()
