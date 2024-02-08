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

"""Tests for param_remapping."""

from typing import Any, List, Mapping, Tuple, Union

from absl.testing import absltest
from absl.testing import parameterized
import flax
from flax import linen as nn
import jax
from jax import numpy as jnp

from flaxformer import testing_utils
from flaxformer.architectures.common import param_remapping


class OldDense(nn.Module):
  shape: Tuple[int, ...]

  def setup(self):
    self.w = self.param('w', nn.initializers.normal(), self.shape)

  def __call__(self, x):
    return self.w @ x


class NewDense(nn.Module, param_remapping.ParameterRemappable):
  shape: Tuple[int, ...]

  def setup(self):
    self.weights = self.param('weights', nn.initializers.normal(), self.shape)

  def __call__(self, x):
    return self.weights @ x

  @nn.nowrap  # exempt from named call decorator.
  @flax.linen.module.wrap_method_once
  def _from_save_format(
      self, params: param_remapping.RecursiveDefaultDict) -> Mapping[str, Any]:
    params.merge('weights', params.pop('w'))
    return params

  @nn.nowrap  # exempt from named call decorator.
  @flax.linen.module.wrap_method_once
  def _to_save_format(
      self, params: param_remapping.RecursiveDefaultDict) -> Mapping[str, Any]:
    params.merge('w', params.pop('weights'))
    return params


class Mlp(nn.Module, param_remapping.ParameterRemappable):
  layers: List[Union[OldDense, NewDense]]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x


class MlpSetupInit(nn.Module, param_remapping.ParameterRemappable):

  def setup(self):
    self.layers = [NewDense(shape=(3, 4)), NewDense(shape=(5, 3))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x


class NestedStructuresMlp(nn.Module, param_remapping.ParameterRemappable):
  layers: List[List[Union[OldDense, NewDense]]]

  def __call__(self, x):
    for inner_layers in self.layers:
      for layer in inner_layers:
        x = layer(x)
    return x


class ParameterRemappableTest(parameterized.TestCase):

  @parameterized.named_parameters([
      ('direct_init', Mlp([NewDense(shape=(3, 4)),
                           NewDense(shape=(5, 3))])),
      ('setup_init', MlpSetupInit()),
  ])
  def test_load_old_checkpoint(self, new_mlp):
    # Instantiate an old model and get its params in order to simulate the
    # existence of an old checkpoint.
    old_mlp = Mlp([OldDense(shape=(3, 4)), OldDense(shape=(5, 3))])
    old_mlp_vars = old_mlp.init(jax.random.PRNGKey(0), jnp.zeros([4]))
    self.assertSameStructure(
        testing_utils.param_shapes(old_mlp_vars),
        {'params': {
            'layers_0': {
                'w': [3, 4]
            },
            'layers_1': {
                'w': [5, 3]
            },
        }}, 'old_mlp_vars = ' + testing_utils.format_params_shapes(
            testing_utils.param_shapes(old_mlp_vars)))

    # Use the new model to remap the old parameters into the new format.
    new_mlp_init_vars = new_mlp.init(jax.random.PRNGKey(0), jnp.zeros([4]))
    new_mlp_remapped_vars = {
        'params':
            new_mlp.apply(
                new_mlp_init_vars,
                old_mlp_vars['params'],
                method=new_mlp.from_save_format)
    }
    self.assertSameStructure(
        testing_utils.param_shapes(new_mlp_remapped_vars), {
            'params': {
                'layers_0': {
                    'weights': [3, 4]
                },
                'layers_1': {
                    'weights': [5, 3]
                },
            }
        }, 'new_mlp_remapped_vars = ' + testing_utils.format_params_shapes(
            testing_utils.param_shapes(new_mlp_remapped_vars)))

    # Map the new model's parameters into the save format.
    new_mlp_saveformat_vars = {
        'params':
            new_mlp.apply(
                new_mlp_remapped_vars,
                new_mlp_remapped_vars['params'],
                method=new_mlp.to_save_format)
    }
    self.assertSameStructure(
        testing_utils.param_shapes(new_mlp_saveformat_vars),
        {'params': {
            'layers_0': {
                'w': [3, 4]
            },
            'layers_1': {
                'w': [5, 3]
            },
        }}, 'new_mlp_saveformat_vars = ' + testing_utils.format_params_shapes(
            testing_utils.param_shapes(new_mlp_saveformat_vars)))

  def test_nested_structures(self):
    old_mlp = NestedStructuresMlp(
        [[OldDense(shape=(3, 4)),
          OldDense(shape=(5, 3))],
         [OldDense(shape=(6, 5)),
          OldDense(shape=(7, 6))]])
    old_mlp_vars = old_mlp.init(jax.random.PRNGKey(0), jnp.zeros([4]))
    self.assertSameStructure(
        testing_utils.param_shapes(old_mlp_vars), {
            'params': {
                'layers_0_0': {
                    'w': [3, 4]
                },
                'layers_0_1': {
                    'w': [5, 3]
                },
                'layers_1_0': {
                    'w': [6, 5]
                },
                'layers_1_1': {
                    'w': [7, 6]
                },
            }
        }, 'old_mlp_vars = ' + testing_utils.format_params_shapes(
            testing_utils.param_shapes(old_mlp_vars)))

    new_mlp = NestedStructuresMlp(
        [[NewDense(shape=(3, 4)),
          NewDense(shape=(5, 3))],
         [NewDense(shape=(6, 5)),
          NewDense(shape=(7, 6))]])

    # Use the new model to remap the old parameters into the new format.
    new_mlp_init_vars = new_mlp.init(jax.random.PRNGKey(0), jnp.zeros([4]))
    new_mlp_remapped_vars = {
        'params':
            new_mlp.apply(
                new_mlp_init_vars,
                old_mlp_vars['params'],
                method=new_mlp.from_save_format)
    }
    self.assertSameStructure(
        testing_utils.param_shapes(new_mlp_remapped_vars), {
            'params': {
                'layers_0_0': {
                    'weights': [3, 4]
                },
                'layers_0_1': {
                    'weights': [5, 3]
                },
                'layers_1_0': {
                    'weights': [6, 5]
                },
                'layers_1_1': {
                    'weights': [7, 6]
                },
            }
        }, 'new_mlp_remapped_vars = ' + testing_utils.format_params_shapes(
            testing_utils.param_shapes(new_mlp_remapped_vars)))

    # Map the new model's parameters into the save format.
    new_mlp_saveformat_vars = {
        'params':
            new_mlp.apply(
                new_mlp_remapped_vars,
                new_mlp_remapped_vars['params'],
                method=new_mlp.to_save_format)
    }
    self.assertSameStructure(
        testing_utils.param_shapes(new_mlp_saveformat_vars), {
            'params': {
                'layers_0_0': {
                    'w': [3, 4]
                },
                'layers_0_1': {
                    'w': [5, 3]
                },
                'layers_1_0': {
                    'w': [6, 5]
                },
                'layers_1_1': {
                    'w': [7, 6]
                },
            }
        }, 'new_mlp_saveformat_vars = ' + testing_utils.format_params_shapes(
            testing_utils.param_shapes(new_mlp_saveformat_vars)))


class RecursiveDefaultDictTest(absltest.TestCase):

  def test_merge_mapping(self):
    d = param_remapping.RecursiveDefaultDict()
    d['a']['b']['c'] = 1
    d['a'].merge('b', {'d': 2})
    self.assertSameStructure(d.to_dict(), {'a': {'b': {'c': 1, 'd': 2}}})

  def test_merge_leaf(self):
    d = param_remapping.RecursiveDefaultDict()
    d['a']['b']['c'] = 1
    d['a']['b'].merge('d', 2)
    self.assertSameStructure(d.to_dict(), {'a': {'b': {'c': 1, 'd': 2}}})

  def test_merge_overwrite_mapping_with_leaf_1(self):
    d = param_remapping.RecursiveDefaultDict()
    d['a']['b']['c'] = 1
    with self.assertRaisesRegex(
        ValueError, "Cannot merge a non-mapping into a mapping; key: 'b'.*"):
      d['a'].merge('b', 2)

  def test_merge_overwrite_mapping_with_leaf_2(self):
    d = param_remapping.RecursiveDefaultDict()
    d['a']['b']['c'] = 1
    with self.assertRaisesRegex(
        ValueError, "Cannot merge a non-mapping into a mapping; key: 'b'.*"):
      d.merge('a', {'b': 2})

  def test_merge_overwrite_leaf_with_mapping_1(self):
    d = param_remapping.RecursiveDefaultDict()
    d['a']['b']['c'] = 1
    with self.assertRaisesWithLiteralMatch(
        ValueError, "Cannot overwrite existing leaf key 'c' via merge"):
      d['a']['b'].merge('c', {'d', 2})

  def test_merge_overwrite_leaf_with_mapping_2(self):
    d = param_remapping.RecursiveDefaultDict()
    d['a']['b']['c'] = 1
    with self.assertRaisesWithLiteralMatch(
        ValueError, "Cannot overwrite existing leaf key 'c' via merge"):
      d['a'].merge('b', {'c': {'d', 2}})

  def test_pop_and_merge(self):
    d = param_remapping.RecursiveDefaultDict()
    d['a']['b']['c'] = 1
    self.assertSameStructure(d.to_dict(), {'a': {'b': {'c': 1}}})

    d2 = param_remapping.RecursiveDefaultDict()
    d2['a'].merge('b2', d['a'].pop('b'))
    self.assertSameStructure(d.to_dict(), {})
    self.assertSameStructure(d2.to_dict(), {'a': {'b2': {'c': 1}}})

  def test_pop_and_update(self):
    d = param_remapping.RecursiveDefaultDict()
    d['a']['b']['c'] = 1
    self.assertSameStructure(d.to_dict(), {'a': {'b': {'c': 1}}})

    d2 = param_remapping.RecursiveDefaultDict()
    d2['a']['b2'].update(d['a'].pop('b'))
    self.assertSameStructure(d.to_dict(), {})
    self.assertSameStructure(d2.to_dict(), {'a': {'b2': {'c': 1}}})


if __name__ == '__main__':
  absltest.main()
