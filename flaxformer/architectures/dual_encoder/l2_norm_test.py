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

"""Tests for L2 norm."""
from absl.testing import absltest
from jax import numpy as jnp
from jax import random

from flaxformer.architectures.dual_encoder import l2_norm


class L2NormTest(absltest.TestCase):

  def test_l2_norm(self):
    """Test if the l2 norm layer has correct shapes and types."""
    rng = random.PRNGKey(0)
    key1, key2, key3 = random.split(rng, 3)
    x = random.normal(key1, (2, 3, 4))
    model_fn = lambda dtype: l2_norm.L2Norm(dtype=dtype)
    y, _ = model_fn(jnp.float32).init_with_output(key2, x)
    self.assertEqual(x.shape, y.shape)
    self.assertEqual(y.dtype, jnp.float32)

    y, _ = model_fn(jnp.int32).init_with_output(key3, x)
    self.assertEqual(y.dtype, jnp.int32)


if __name__ == "__main__":
  absltest.main()
