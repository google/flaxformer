# Copyright 2021 Google LLC.
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

"""Tests for dense modules."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
import jax
from jax import random
from jax.nn import initializers
import jax.numpy as jnp
import numpy as np

from flaxformer import sharding
from flaxformer.components import dense

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class DenseTest(parameterized.TestCase):

  def test_dense_general_no_bias(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 3))
    model = dense.DenseGeneral(
        features=4,
        use_bias=False,
        kernel_init=initializers.ones,
    )
    y, _ = model.init_with_output(rng, x)
    self.assertEqual(y.shape, (1, 4))
    np.testing.assert_allclose(y, np.full((1, 4), 3.))

  def test_dense_general_with_bias(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 3))
    model = dense.DenseGeneral(
        features=4,
        use_bias=True,
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, _ = model.init_with_output(rng, x)
    self.assertEqual(y.shape, (1, 4))
    np.testing.assert_allclose(y, np.full((1, 4), 4.))

  def test_dense_general_two_features(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 3))
    model = dense.DenseGeneral(
        features=(2, 2),
        use_bias=False,
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
        kernel_axis_names=('a', 'b', 'c'),
    )
    y, variables = model.init_with_output(rng, x)
    # We transform the last input dimension to two output dimensions (2, 2).
    np.testing.assert_allclose(y, np.full((1, 2, 2), 3.))

    # The output sharding dimensions have been collapsed.
    sharding.check_params_and_axis_names_match(variables)
    self.assertEqual(variables['param_axes']['kernel_axes'],
                     sharding.axis_names('a', 'b * c'))

  def test_dense_general_two_axes(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 2, 2))
    model = dense.DenseGeneral(
        features=3,
        use_bias=False,
        axis=(-2, 2),  # Note: this is the same as (1, 2).
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
        kernel_axis_names=('a', 'b', 'c'),
    )
    y, variables = model.init_with_output(rng, x)
    # We transform the last two input dimensions (2, 2) to one output dimension.
    np.testing.assert_allclose(y, np.full((1, 3), 4.))

    # The input sharding dimensions have been collapsed.
    sharding.check_params_and_axis_names_match(variables)
    self.assertEqual(variables['param_axes']['kernel_axes'],
                     sharding.axis_names('a * b', 'c'))

  def test_mlp_same_out_dim(self):
    module = dense.MlpBlock(
        use_bias=False,
        intermediate_dim=4,
        activations=('relu',),
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        dtype=jnp.float32,
    )
    inputs = np.array(
        [
            # Batch 1.
            [[1, 1], [1, 1], [1, 2]],
            # Batch 2.
            [[2, 2], [3, 1], [2, 2]],
        ],
        dtype=np.float32)
    params = module.init(random.PRNGKey(0), inputs, enable_dropout=False)
    self.assertDictEqual(
        jax.tree_map(lambda a: a.tolist(), params.unfreeze()), {
            'params': {
                'wi': {
                    'kernel': [[
                        -0.8675811290740967, 0.08417510986328125,
                        0.022586345672607422, -0.9124102592468262
                    ],
                               [
                                   -0.19464373588562012, 0.49809837341308594,
                                   0.7808468341827393, 0.9267289638519287
                               ]],
                },
                'wo': {
                    'kernel': [[0.01154780387878418, 0.1397249698638916],
                               [0.974980354309082, 0.5903260707855225],
                               [-0.05997943878173828, 0.616570234298706],
                               [0.2934272289276123, 0.8181164264678955]],
                },
            },
            'param_axes': {
                'wi': {
                    'kernel_axes':
                        sharding.AxisMetadata(names=('embed', 'intermediate'))
                },
                'wo': {
                    'kernel_axes':
                        sharding.AxisMetadata(names=('intermediate', 'embed'))
                },
            }
        })
    result = module.apply(params, inputs, enable_dropout=False)
    np.testing.assert_allclose(
        result.tolist(),
        [[[0.5237172245979309, 0.8508185744285583],
          [0.5237172245979309, 0.8508185744285583],
          [1.2344461679458618, 2.3844780921936035]],
         [[1.0474344491958618, 1.7016371488571167],
          [0.6809444427490234, 0.9663378596305847],
          [1.0474344491958618, 1.7016371488571167]]],
        rtol=1e-6,
    )

  def test_mlp_different_out_dim(self):
    module = dense.MlpBlock(
        use_bias=False,
        intermediate_dim=4,
        activations=('relu',),
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        dtype=jnp.float32,
        out_dim=3,
    )
    inputs = np.array(
        [
            # Batch 1.
            [[1, 1], [1, 1], [1, 2]],
            # Batch 2.
            [[2, 2], [3, 1], [2, 2]],
        ],
        dtype=np.float32)
    params = module.init(
        random.PRNGKey(0), inputs, enable_dropout=False, mutable=['params'])
    self.assertEqual(
        jax.tree_map(lambda a: a.tolist(), params), {
            'params': {
                'wi': {
                    'kernel': [[
                        -0.8675811290740967, 0.08417510986328125,
                        0.022586345672607422, -0.9124102592468262
                    ],
                               [
                                   -0.19464373588562012, 0.49809837341308594,
                                   0.7808468341827393, 0.9267289638519287
                               ]],
                },
                'wo': {
                    'kernel': [[
                        -0.7511187791824341, 0.6027762293815613,
                        0.08945237100124359
                    ],
                               [
                                   0.09322204440832138, 0.7264236211776733,
                                   -0.187033012509346
                               ],
                               [
                                   0.18425928056240082, -0.45850488543510437,
                                   0.06437449157238007
                               ],
                               [
                                   -0.35469120740890503, 0.05190309137105942,
                                   0.28195270895957947
                               ]],
                },
            },
        })
    result = module.apply(params, inputs, enable_dropout=False)
    np.testing.assert_allclose(
        result.tolist(),
        [[[0.19724202156066895, 0.055342353880405426, -0.05314655974507332],
          [0.19724202156066895, 0.055342353880405426, -0.05314655974507332],
          [0.0588514469563961, 0.10725077986717224, 0.1652529537677765]],
         [[0.3944840431213379, 0.11068470776081085, -0.10629311949014664],
          [0.22633817791938782, 0.15618085861206055, -0.08576283603906631],
          [0.3944840431213379, 0.11068470776081085, -0.10629311949014664]]],
        rtol=1e-6,
    )

  def test_fuse_kernels(self):
    x = np.random.randn(2, 3)
    fused = dense.MlpBlock(
        use_bias=False,
        intermediate_dim=4,
        fuse_kernels=True,
        activations=('gelu', 'linear'))
    not_fused = dense.MlpBlock(
        use_bias=False,
        intermediate_dim=4,
        fuse_kernels=False,
        activations=('gelu', 'linear'))

    wi_0 = np.random.randn(3, 4)
    wi_1 = np.random.randn(3, 4)
    wo = np.random.randn(4, 3)

    params_not_fused = {
        'wi_0': {
            'kernel': wi_0
        },
        'wi_1': {
            'kernel': wi_1
        },
        'wo': {
            'kernel': wo
        }
    }
    params_fused = {
        'wi_fused': {
            'kernel':
                np.stack([wi_0, wi_1], axis=1)  # shape: [3, 2, 4]
        },
        'wo': {
            'kernel': wo
        }
    }

    y_fused = fused.apply({'params': params_fused}, x, enable_dropout=False)
    y_not_fused = not_fused.apply({'params': params_not_fused},
                                  x,
                                  enable_dropout=False)
    np.testing.assert_allclose(y_fused, y_not_fused, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
