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

"""Tests for layer_norm."""

from absl.testing import absltest
import chex
from flax import linen as nn
from flax.core import unfreeze
from jax import numpy as jnp
from jax import random

from flaxformer.components import layer_norm


class T5LayerNormTest(absltest.TestCase):

  def test_layer_norm(self):
    rng = random.PRNGKey(0)
    key1, key2, key3 = random.split(rng, 3)
    x = random.normal(key1, (2, 3, 4))
    model_fn = lambda dtype: layer_norm.T5LayerNorm(dtype=dtype)
    y, _ = model_fn(jnp.float32).init_with_output(key2, x)
    self.assertEqual(x.shape, y.shape)
    self.assertEqual(y.dtype, jnp.float32)

    y, _ = model_fn(jnp.int32).init_with_output(key3, x)
    self.assertEqual(y.dtype, jnp.int32)

  def test_default_axis_name(self):
    module = layer_norm.T5LayerNorm()
    rng = random.PRNGKey(0)
    variables = module.init(rng, jnp.zeros([2, 3, 4], dtype=jnp.float32))
    chex.assert_trees_all_equal_shapes(
        unfreeze(variables["params"]),
        {
            "scale": jnp.zeros([4]),
        },
    )
    chex.assert_trees_all_equal(
        unfreeze(variables["params_axes"]),
        {
            "scale_axes": nn.partitioning.AxisMetadata(names=("embed",)),
        },
    )

if __name__ == "__main__":
  absltest.main()
