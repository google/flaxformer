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

"""Tests for scaling module."""

from absl.testing import absltest
from jax import numpy as jnp
from jax import random
import tensorflow as tf

from flaxformer.architectures.dual_encoder import components


class LearnableScalingTest(tf.test.TestCase):

  def test_logits_get_scaled_by_init_scaling_value_during_training(self):
    rng = random.PRNGKey(0)
    key1, key2, key3 = random.split(rng, 3)
    init_scaling_value = random.uniform(key1).item()
    x = random.normal(key2, (2, 3, 4))

    def _get_learnable_scaling(dtype):
      return components.LearnableScaling(
          dtype=dtype, init_scaling_value=init_scaling_value
      )

    model_fn = _get_learnable_scaling
    y = model_fn(jnp.float32).init_with_output(key3, x, enable_dropout=True)

    self.assertAllClose(y[0], init_scaling_value * x)

  def test_logits_get_scaled_by_scalar_and_bias(self):
    logit_scaling_module = components.LearnableScalingAndBias()
    y, _ = logit_scaling_module.init_with_output(
        random.PRNGKey(0), jnp.array([1, 0])
    )
    self.assertAllEqual(y, [4, -11])

  def test_logits_get_scaled_by_scalar_and_bias_learnable(self):
    rng = random.PRNGKey(0)
    key1, key2, key3 = random.split(rng, 3)
    init_scaling_value = random.uniform(key1).item()
    init_bias = random.uniform(key1).item()
    x = random.normal(key2, (2, 3, 4))

    def _get_learnable_scaling(dtype):
      return components.LearnableScalingAndBias(
          dtype=dtype,
          init_scaling_value=init_scaling_value,
          init_bias=init_bias,
      )

    model_fn = _get_learnable_scaling
    y = model_fn(jnp.float32).init_with_output(key3, x, enable_dropout=True)

    self.assertAllClose(y[0], init_scaling_value * x + init_bias)


if __name__ == '__main__':
  absltest.main()
