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

"""Tests for flaxformer.initializers."""
from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import numpy as jnp
import numpy as np
from flaxformer.components import initializers


class InitializersTest(parameterized.TestCase):

  def test_sinusoidal_returns_correct_shape(self):
    """Tests that we get the expected shape, starting with a 1 dimension."""
    max_len = 10
    min_scale = 1.0
    max_scale = 10000.0
    init_fn = initializers.sinusoidal(min_scale=min_scale, max_scale=max_scale)
    rng = jax.random.PRNGKey(0)
    features = 5
    shape = (max_len, features)
    result = init_fn(rng, shape, jnp.float32)
    self.assertEqual(result.shape, shape)

  def test_sinusoidal_has_deterministic_outputs(self):
    """Tests that we always get the same sinusoids given the same shape."""
    max_len = 10
    min_scale = 1.0
    max_scale = 10000.0
    init_fn = initializers.sinusoidal(min_scale=min_scale, max_scale=max_scale)
    rng = jax.random.PRNGKey(0)
    features = 5
    shape = (max_len, features)
    result = init_fn(rng, shape, jnp.float32)
    expected = [
        [
            0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 1.0000000e+00,
            0.0000000e+00
        ],
        [
            8.4147096e-01, 9.9999997e-05, 5.4030228e-01, 1.0000000e+00,
            0.0000000e+00
        ],
        [
            9.0929741e-01, 1.9999999e-04, -4.1614684e-01, 1.0000000e+00,
            0.0000000e+00
        ],
        [
            1.4112000e-01, 2.9999999e-04, -9.8999250e-01, 9.9999994e-01,
            0.0000000e+00
        ],
        [
            -7.5680250e-01, 3.9999999e-04, -6.5364361e-01, 9.9999994e-01,
            0.0000000e+00
        ],
        [
            -9.5892429e-01, 4.9999997e-04, 2.8366220e-01, 9.9999988e-01,
            0.0000000e+00
        ],
        [
            -2.7941549e-01, 5.9999997e-04, 9.6017027e-01, 9.9999982e-01,
            0.0000000e+00
        ],
        [
            6.5698659e-01, 6.9999992e-04, 7.5390226e-01, 9.9999976e-01,
            0.0000000e+00
        ],
        [
            9.8935825e-01, 7.9999992e-04, -1.4550003e-01, 9.9999970e-01,
            0.0000000e+00
        ],
        [
            4.1211849e-01, 8.9999987e-04, -9.1113025e-01, 9.9999958e-01,
            0.0000000e+00
        ],
    ]
    np.testing.assert_array_almost_equal(result, expected)

  def test_sinusoidal_raises_exception_for_incorrect_shape(self):
    """Tests that we get a ValueError if max_len does not match the shape."""
    # TODO: Remove this test once max_len is removed from sinusoidal.
    max_len = 10
    init_fn = initializers.sinusoidal()
    rng = jax.random.PRNGKey(0)
    features = 5
    shape = (max_len, features, 1)  # Adding an extra dimension triggers error.
    with self.assertRaises(ValueError):
      init_fn(rng, shape, jnp.float32)

  def test_truncated_normal_returns_correct_shape(self):
    """Tests that truncated normal returns the expected shape."""
    init_fn = initializers.truncated_normal(stddev=1.0, dtype=jnp.float32)
    rng = jax.random.PRNGKey(0)
    num_embeddings = 10
    features = 5
    shape = (num_embeddings, features)
    result = init_fn(rng, shape, jnp.float32)
    self.assertEqual(result.shape, shape)

  def test_truncated_normal_has_deterministic_outputs(self):
    """Tests that truncated normal returns the same outputs with fixed RNG."""
    init_fn = initializers.truncated_normal(stddev=1.0, dtype=jnp.float32)
    rng = jax.random.PRNGKey(0)
    num_embeddings = 10
    features = 5
    shape = (num_embeddings, features)
    result = init_fn(rng, shape, jnp.float32)
    expected = [
        [
            -1.85987040e-01, -1.13764632e+00, -8.63419712e-01, 1.38558254e-01,
            1.73241040e-03
        ],
        [
            -7.08928108e-01, -7.57956028e-01, 1.16832398e-01, -1.67809799e-01,
            9.89689469e-01
        ],
        [
            -2.33572051e-01, -7.33788013e-01, 3.87833655e-01, 5.08138120e-01,
            -1.24911046e+00
        ],
        [
            -1.79018974e+00, -2.59301901e-01, 1.25438678e+00, 2.95449466e-01,
            3.30820709e-01
        ],
        [
            -3.51254076e-01, 2.62031645e-01, 1.12873232e+00, -2.84244955e-01,
            -1.55112541e+00
        ],
        [
            1.40685475e+00, -5.01563549e-01, 1.24033138e-01, -1.18946660e+00,
            -1.26286268e+00
        ],
        [
            5.54490983e-01, 4.36401725e-01, 3.97840403e-02, -5.70072941e-02,
            2.93129623e-01
        ],
        [
            1.57007650e-01, -5.00848331e-02, 1.08628595e+00, 1.52689147e+00,
            3.50468487e-01
        ],
        [
            -8.84301245e-01, -1.06949806e-01, 6.85548604e-01, 8.57080519e-01,
            -9.17811871e-01
        ],
        [
            -3.04965496e-01, -1.29926765e+00, -9.85570103e-02, -8.27740490e-01,
            -3.74757677e-01
        ],
    ]
    np.testing.assert_array_almost_equal(result, expected)

  def test_truncated_normal_returns_correct_mean(self):
    """Tests that truncated normal returns the requested mean."""
    mean = 2.0
    stddev = 1.0
    init_fn = initializers.truncated_normal(
        mean=mean, stddev=stddev, dtype=jnp.float32)
    rng = jax.random.PRNGKey(0)
    num_embeddings = 100
    features = 100
    shape = (num_embeddings, features)
    result = init_fn(rng, shape, jnp.float32)
    np.testing.assert_allclose(np.mean(result), mean, atol=0.01)

  def test_truncated_normal_returns_correct_stddev(self):
    """Tests that truncated normal returns the requested stddev."""
    mean = 0.0
    stddev = 0.05
    init_fn = initializers.truncated_normal(
        mean=mean, stddev=stddev, dtype=jnp.float32)
    rng = jax.random.PRNGKey(0)
    num_embeddings = 100
    features = 100
    shape = (num_embeddings, features)
    result = init_fn(rng, shape, jnp.float32)
    np.testing.assert_allclose(np.std(result), stddev, atol=0.01)


if __name__ == '__main__':
  absltest.main()
