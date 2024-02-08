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

"""Functions for enabling parameter remapping in T5X via Gin configuration."""

from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from absl import logging
from flax import linen as nn
from flax import traverse_util
from flax.core import frozen_dict
from t5x import checkpoints

from flaxformer.architectures.common import param_remapping


# frozen_dict.unfreeze is incorrectly typed, so introduce an alias.
def _unfreeze(x: Mapping[str, Any]) -> Dict[str, Any]:
  return frozen_dict.unfreeze(x)  # pytype: disable=wrong-arg-types


def _flattened_names(state_dict: Mapping[str, Any]) -> Sequence[str]:
  return [
      '/'.join(k)
      for k in traverse_util.flatten_dict(state_dict, keep_empty_nodes=True)
  ]


def _apply_remap_fn(remap_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
                    state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
  result = _unfreeze(state_dict)
  result['state']['param_states'] = remap_fn(result['state']['param_states'])
  result['target'] = remap_fn(result['target'])
  return result


def make_to_save_format_fn(
    module: nn.Module) -> checkpoints.SaveStateTransformationFn:
  """Returns a t5x on-save state transformation function.

  Args:
    module: A Flax module inheriting from param_remapping.ParamRemappable.
  """
  if not isinstance(module, param_remapping.ParameterRemappable):
    raise ValueError('Expected `module` to be a `ParameterRemappable`, but was '
                     f'{type(module)}')

  def remap(params: Mapping[str, Any]) -> Mapping[str, Any]:
    return module.apply({}, params, method=module.to_save_format)

  def to_save_format(
      state_dict: checkpoints.PyTreeDef,
      parameter_infos: checkpoints.PyTreeDef,
  ) -> Tuple[checkpoints.PyTreeDef, checkpoints.PyTreeDef]:
    for name in _flattened_names(state_dict):
      logging.info('to_save_format input state_dict: %s', name)
    for name in _flattened_names(parameter_infos):
      logging.info('to_save_format input parameter_infos: %s', name)

    result_state_dict = _apply_remap_fn(remap, state_dict)
    result_parameter_infos = _apply_remap_fn(remap, parameter_infos)

    flat_parameter_infos = traverse_util.flatten_dict(
        result_parameter_infos, keep_empty_nodes=True)
    result_parameter_infos = traverse_util.unflatten_dict({
        k: (v if k[-1] != param_remapping.VERSION_KEY else None)
        for k, v in flat_parameter_infos.items()
    })

    for name in _flattened_names(result_state_dict):
      logging.info('to_save_format output state_dict: %s', name)
    for name in _flattened_names(result_parameter_infos):
      logging.info('to_save_format output parameter_infos: %s', name)
    return result_state_dict, result_parameter_infos

  return to_save_format


def make_from_save_format_fn(
    module: nn.Module) -> checkpoints.RestoreStateTransformationFn:
  """Returns a t5x on-restore state transformation function.

  Args:
    module: A Flax module inheriting from param_remapping.ParamRemappable.
  """
  if not isinstance(module, param_remapping.ParameterRemappable):
    raise ValueError('Expected `module` to be a `ParameterRemappable`, but was '
                     f'{type(module)}')

  def remap(params: Mapping[str, Any]) -> Mapping[str, Any]:
    return module.apply({}, params, method=module.from_save_format)

  def from_save_format(state_dict: checkpoints.PyTreeDef,
                       target_state_dict: checkpoints.PyTreeDef,
                       *,
                       is_resuming: bool = False) -> checkpoints.PyTreeDef:
    del target_state_dict  # Unused.
    del is_resuming  # Unused.

    for name in _flattened_names(state_dict):
      logging.info('from_save_format input state_dict: %s', name)

    result_state_dict = _apply_remap_fn(remap, state_dict)

    for name in _flattened_names(result_state_dict):
      logging.info('from_save_format output state_dict: %s', name)
    return result_state_dict

  return from_save_format
