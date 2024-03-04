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

"""Tests for dense modules."""

import functools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from aqt.jax_legacy.jax import quantization as aqt
import flax
from flax import linen as nn
from flax.linen import partitioning
import jax
from jax import dtypes
from jax import random
from jax.nn import initializers
import jax.numpy as jnp
import numpy as np

from flaxformer import sharding
from flaxformer import testing_utils
from flaxformer.components import dense

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


def assert_same_tree(a, b):
  jax.tree.map(
      functools.partial(np.testing.assert_allclose, atol=1e-6, rtol=1e-6), a, b)


class DenseTest(parameterized.TestCase):

  def _mock_initializer(self, key, shape, dtype=jnp.float_, val=1.0):  # pylint: disable=unused-argument
    return jnp.ones(shape, dtypes.canonicalize_dtype(dtype)) * val

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
    self.assertEqual(variables['params_axes']['kernel_axes'],
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
    self.assertEqual(variables['params_axes']['kernel_axes'],
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

    assert_same_tree(
        flax.core.unfreeze(params['params']),
        {
            'wi': {
                'kernel': [
                    [
                        -0.2650487422943115,
                        -0.9350943565368652,
                        -0.09850478172302246,
                        -0.3685007095336914,
                    ],
                    [
                        0.4673573970794678,
                        0.058478593826293945,
                        -0.5871121883392334,
                        -0.7413773536682129,
                    ],
                ],
            },
            'wo': {
                'kernel': [
                    [-0.7278401851654053, 0.6603918075561523],
                    [-0.4713869094848633, -0.37511157989501953],
                    [-0.15709185600280762, 0.7399897575378418],
                    [-0.7014286518096924, -0.2968623638153076],
                ],
            },
        },
    )

    self.assertDictEqual(
        flax.core.unfreeze(params['params_axes']),
        {
            'wi': {
                'kernel_axes': partitioning.AxisMetadata(names=('embed', 'mlp'))
            },
            'wo': {
                'kernel_axes': partitioning.AxisMetadata(names=('mlp', 'embed'))
            },
        },
    )
    result = module.apply(params, inputs, enable_dropout=False)

    np.testing.assert_allclose(
        result.tolist(),
        [[[-0.14724837243556976, 0.13360297679901123],
          [-0.14724837243556976, 0.13360297679901123],
          [-0.4874098598957062, 0.44224196672439575]],
         [[-0.2944967448711395, 0.26720595359802246], [0.0, 0.0],
          [-0.2944967448711395, 0.26720595359802246]]],
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

    variables = module.init(
        random.PRNGKey(0),
        inputs,
        enable_dropout=False,
        mutable=['params', 'params_axes'])
    assert_same_tree(
        flax.core.unfreeze(variables['params']),
        {
            'wi': {
                'kernel': [
                    [
                        -0.2650487422943115,
                        -0.9350943565368652,
                        -0.09850478172302246,
                        -0.3685007095336914,
                    ],
                    [
                        0.4673573970794678,
                        0.058478593826293945,
                        -0.5871121883392334,
                        -0.7413773536682129,
                    ],
                ],
            },
            'wo': {
                'kernel': [
                    [
                        0.549019455909729,
                        -0.7615442276000977,
                        0.2908056378364563,
                    ],
                    [
                        0.8247717618942261,
                        -0.37039434909820557,
                        0.14754922688007355,
                    ],
                    [
                        -0.4929429590702057,
                        0.34858351945877075,
                        -0.27896377444267273,
                    ],
                    [
                        -0.5565190315246582,
                        -0.8740609288215637,
                        0.6347796320915222,
                    ],
                ],
            },
        },
    )

    self.assertDictEqual(
        flax.core.unfreeze(variables['params_axes']),
        {
            'wi': {
                'kernel_axes': partitioning.AxisMetadata(names=('embed', 'mlp'))
            },
            'wo': {
                'kernel_axes': partitioning.AxisMetadata(names=('mlp', 'embed'))
            },
        },
    )
    result = module.apply(variables, inputs, enable_dropout=False)
    np.testing.assert_allclose(
        result.tolist(),
        [[[0.1110713854432106, -0.1540669947862625, 0.05883249640464783],
          [0.1110713854432106, -0.1540669947862625, 0.05883249640464783],
          [0.36765968799591064, -0.509980320930481, 0.19474266469478607]],
         [[0.2221427708864212, -0.308133989572525, 0.11766499280929565],
          [0.0, 0.0, 0.0],
          [0.2221427708864212, -0.308133989572525, 0.11766499280929565]]],
        rtol=1e-6,
    )

  def test_mlp_input_shapes(self):
    module = dense.MlpBlock(
        use_bias=False,
        intermediate_dim=4,
        activations=('relu',),
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        dtype=jnp.float32,
    )
    axis_rules = [('batch', 'data'), ('embed', None), ('length', None),
                  ('mlp', 'model')]

    # 2D inputs.
    inputs = np.array(
        [
            [1, 2, 3],  # Batch 1.
            [4, 5, 6],  # Batch 2.
        ],
        dtype=np.float32)
    with mock.patch(
        'flax.linen.partitioning._AxisRules.rules',
        new_callable=mock.PropertyMock,
        return_value=axis_rules):
      result, _ = module.init_with_output(
          random.PRNGKey(0), inputs, enable_dropout=False)
    expected_result = [[
        1.1578339338302612, -2.476144552230835, 1.1046674251556396
    ], [2.4860682487487793, -5.988793849945068, 2.46048641204834]]
    np.testing.assert_allclose(
        result.tolist(),
        expected_result,
        rtol=1e-6,
    )

    # 3D inputs
    inputs_with_batch_dim = inputs[np.newaxis, ...]
    with mock.patch(
        'flax.linen.partitioning._AxisRules.rules',
        new_callable=mock.PropertyMock,
        return_value=axis_rules):
      batch_result, _ = module.init_with_output(
          random.PRNGKey(0), inputs_with_batch_dim, enable_dropout=False)
    np.testing.assert_allclose(batch_result, result[np.newaxis, ...])

  def test_user_defined_data_sharding_constraints(self):
    customized_module = dense.MlpBlock(
        use_bias=False,
        intermediate_dim=4,
        activations=('relu',),
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        dtype=jnp.float32,
        data_sharding_constraints=('my_constraint', 'embed'))

    axis_rules = [('embed', None), ('my_constraint', 'model')]
    inputs = np.array(
        [
            [1, 2, 3],  # Batch 1.
            [4, 5, 6],  # Batch 2.
        ],
        dtype=np.float32)

    with mock.patch(
        'flax.linen.partitioning._AxisRules.rules',
        new_callable=mock.PropertyMock,
        return_value=axis_rules):
      result, _ = customized_module.init_with_output(
          random.PRNGKey(0), inputs, enable_dropout=False)

    expected_result = [[
        1.1578339338302612, -2.476144552230835, 1.1046674251556396
    ], [2.4860682487487793, -5.988793849945068, 2.46048641204834]]
    np.testing.assert_allclose(
        result.tolist(),
        expected_result,
        rtol=1e-6,
    )

  def test_quantization_no_params_specified(self):
    module = dense.MlpBlock(
        use_bias=False,
        intermediate_dim=4,
        activations=('relu',),
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        dtype=jnp.float32,
        use_aqt=True,
    )
    inputs = np.array(
        [
            # Batch 1.
            [[1, 1], [1, 1], [1, 2]],
            # Batch 2.
            [[2, 2], [3, 1], [2, 2]],
        ],
        dtype=np.float32)
    with self.assertRaisesRegex(
        ValueError,
        'If use_aqt is True, either of weights or acts quantization'):
      module.init(random.PRNGKey(0), inputs, enable_dropout=False)

  def test_mlp_materialized_weights(self):
    weight_params = aqt.QuantOps.WeightParams(
        prec=8, half_shift=False, axis=None)
    module = dense.MlpBlock(
        use_bias=False,
        intermediate_dim=4,
        activations=('relu',),
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        dtype=jnp.float32,
        use_aqt=True,
        weight_params=weight_params,
        possibly_use_quantized_vars=True,
    )
    # enable_dropout
    inputs = np.array(
        [
            # Batch 1.
            [[1, 1], [1, 1], [1, 2]],
            # Batch 2.
            [[2, 2], [3, 1], [2, 2]],
        ],
        dtype=np.float32)
    result, variables = module.init_with_output(
        random.PRNGKey(0), inputs, enable_dropout=False)
    assert_same_tree(
        flax.core.unfreeze(variables['params']),
        {
            'wi': {
                'qkernel': [[0, 0, 0, 0], [0, 0, 0, 0]],
                'qscale': [[
                    2.818772e-07,
                    -9.838715e-07,
                    1.211104e-06,
                    2.669436e-07,
                ]],
            },
            'wo': {
                'qkernel': [[0, 0], [0, 0], [0, 0], [0, 0]],
                'qscale': [[-1.854524e-06, 1.883966e-06]],
            },
        },
    )
    self.assertDictEqual(
        testing_utils.param_dtypes_shapes_axes(variables['params'],
                                               variables['params_axes']),
        {
            'wi': {
                'qkernel': ['int8', 'embed=2', 'mlp=4'],
                'qscale': ['float32', 'embed_qscale=1', 'mlp=4']
            },
            'wo': {
                'qkernel': ['int8', 'mlp=4', 'embed=2'],
                'qscale': ['float32', 'mlp_qscale=1', 'embed=2']
            }
        })

    np.testing.assert_allclose(
        result.tolist(),
        [[[-0.0, -0.0], [-0.0, -0.0], [-0.0, -0.0]],
         [[-0.0, -0.0], [-0.0, -0.0], [-0.0, -0.0]]],
        rtol=1e-6,
    )

  def test_mlp_quantized_weights(self):
    weight_params = aqt.QuantOps.WeightParams(
        prec=8, half_shift=False, axis=None)
    module = dense.MlpBlock(
        use_bias=False,
        intermediate_dim=4,
        activations=('relu',),
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        dtype=jnp.float32,
        use_aqt=True,
        weight_params=weight_params,
    )
    inputs = np.array(
        [
            # Batch 1.
            [[1, 1], [1, 1], [1, 2]],
            # Batch 2.
            [[2, 2], [3, 1], [2, 2]],
        ],
        dtype=np.float32)
    result, variables = module.init_with_output(
        random.PRNGKey(0), inputs, enable_dropout=False)
    assert_same_tree(
        flax.core.unfreeze(variables['params']),
        {
            'wi': {
                'kernel': [
                    [
                        -0.2650487422943115,
                        -0.9350943565368652,
                        -0.09850478172302246,
                        -0.3685007095336914,
                    ],
                    [
                        0.4673573970794678,
                        0.058478593826293945,
                        -0.5871121883392334,
                        -0.7413773536682129,
                    ],
                ],
            },
            'wo': {
                'kernel': [
                    [-0.7278401851654053, 0.6603918075561523],
                    [-0.4713869094848633, -0.37511157989501953],
                    [-0.15709185600280762, 0.7399897575378418],
                    [-0.7014286518096924, -0.2968623638153076],
                ],
            },
        },
    )
    self.assertDictEqual(
        testing_utils.param_dtypes_shapes_axes(variables['params'],
                                               variables['params_axes']),
        {
            'wi': {
                'kernel': ['float32', 'embed=2', 'mlp=4']
            },
            'wo': {
                'kernel': ['float32', 'mlp=4', 'embed=2']
            }
        })

    np.testing.assert_allclose(
        result.tolist(),
        [[[-0.14731408655643463, 0.1332627385854721],
          [-0.14731408655643463, 0.1332627385854721],
          [-0.48747575283050537, 0.4409784972667694]],
         [[-0.29462817311286926, 0.2665254771709442], [0.0, 0.0],
          [-0.29462817311286926, 0.2665254771709442]]],
        rtol=1e-6,
    )

  def test_fuse_kernels(self):
    x = np.random.randn(2, 3)
    fused = dense.MlpBlock(
        use_bias=False,
        intermediate_dim=4,
        fuse_kernels=True,
        activations=('gelu', 'linear'))

    # Check default axis names.
    variables = fused.init(
        random.PRNGKey(0),
        x,
        enable_dropout=False,
        mutable=['params', 'params_axes'])
    self.assertEqual(
        jax.tree.map(lambda a: a.tolist(), variables['params_axes']), {
            'wi_fused': {
                'kernel_axes':
                    nn.partitioning.AxisMetadata(
                        names=('embed', 'mlp_activations', 'mlp')),
            },
            'wo': {
                'kernel_axes':
                    nn.partitioning.AxisMetadata(names=('mlp', 'embed')),
            },
        })

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

  @parameterized.named_parameters([
      ('fuse_kernel_set_wi_fused_init', True, True),
      ('fuse_kernel_no_set_wi_fused_init', True, False),
      ('no_fuse_kernel_no_set_wi_fused_init', False, False),
      ('no_fuse_kernel_set_wi_fused_init', False, True)
  ])
  def test_fuse_kernels_kernel_init(self, fuse_kernels, set_wi_fused_init):
    module = dense.MlpBlock(
        use_bias=False,
        intermediate_dim=4,
        fuse_kernels=fuse_kernels,
        activations=('relu', 'linear'),
        kernel_init=initializers.ones,
        wi_fused_kernel_init=(functools.partial(
            self._mock_initializer, val=2.0) if set_wi_fused_init else None),
        bias_init=initializers.zeros,
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

    # Construct expected params
    wi_0 = [[1., 1., 1., 1.], [1., 1., 1., 1.]]
    wi_1 = [[1., 1., 1., 1.], [1., 1., 1., 1.]]
    wo = [[1., 1.], [1., 1.], [1., 1.], [1., 1.]]
    if fuse_kernels:
      if set_wi_fused_init:
        wi_0 = [[2., 2., 2., 2.], [2., 2., 2., 2.]]
        wi_1 = [[2., 2., 2., 2.], [2., 2., 2., 2.]]
      expected_params = {
          'wi_fused': {
              'kernel': np.stack([wi_0, wi_1], axis=1).tolist()
          },
          'wo': {
              'kernel': wo
          }
      }
    else:
      expected_params = {
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

    self.assertDictEqual(
        jax.tree.map(
            lambda a: a.tolist(), flax.core.unfreeze(params['params'])
        ),
        expected_params,
    )


if __name__ == '__main__':
  absltest.main()
