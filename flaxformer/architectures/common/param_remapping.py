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

"""APIs and code related to the automatic parameter-remapping strategy.

TODO: Move this out of `architectures` and into a common FF area?
TODO: Write a g3doc with examples once this is all worked out.
TODO: Expand docstrings and include examples.
TODO: Add unit tests that demonstrate the behavior.
"""

from __future__ import annotations

import abc
import collections
import dataclasses
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import flax
from flax import linen as nn

METADATA_KEY = '__save_format_metadata__'
VERSION_KEY = '__version__'


class ParameterRemappable(metaclass=abc.ABCMeta):
  """Interface for components that can load checkpoints from old versions.

  All modules should inherit from this class, even if they do not currently
  require parameter remapping, so that their submodules (and transitive
  submodules) will have their parameters remapped when a checkpoint is loaded.

  When a module's code is modified in a way that changes its parameter tree, the
  `_from_save_format` method should be implemented to handle the conversion from
  the parameter tree of the older version(s) to the new one. If it is desired
  that the checkpoint "save format" should reflect this change, then the value
  of the `save_format_metadata` property method should be updated. If, instead,
  it is desired that the checkpoint "save format" *not* change, then the
  `_to_save_format` method should be implemented to convert from the new
  parameter tree structure to the old structure.
  """

  @property
  def save_format_metadata(self) -> Dict[str, Any]:
    """Returns this module's current version."""
    return {VERSION_KEY: 0}

  @nn.nowrap  # Exempt from named call decorator.
  @flax.linen.module.wrap_method_once
  def from_save_format(self, params: Mapping[str, Any]) -> Dict[str, Any]:
    """Handles remapping `params`, including recursively calling submodules.

    This default implementation iterates through the class's fields to find
    submodules, and recursively calls `from_save_format` on them. This avoids
    a large amount of boilerplate code that would otherwise be needed in each
    module.

    To define additional (custom) logic, implement `_from_save_format`.

    Args:
      params: The parameter tree before remapping. Usually comes from a
        checkpoint (or legacy model). May contain a special key "__version__"
        whose value is the version number (i.e., `save_format_metadata`'s value)
        of the module corresponding to structure of the parameter tree layout.
        If a "__version__" key is missing, it is assumed to be `0`, indicating
        "the parameter structure before the `ParameterRemappable` was added to
        the module".

    Returns:
      The remapped parameter tree.
    """
    # Copy `params` into a more convenient dict type.
    params = RecursiveDefaultDict(params)
    # Apply any custom remapping logic.
    params = self._from_save_format(params)

    # Recursively call `from_save_format` for all submodules.
    params = self._submodules_from_save_format(params)

    return filter_out_metadata(params.to_dict())

  @nn.nowrap  # Exempt from named call decorator.
  @flax.linen.module.wrap_method_once
  def to_save_format(self, params: Mapping[str, Any]) -> Dict[str, Any]:
    """Converts `params` to the format of `save_format_metadata`.

    This default implementation iterates through the class's fields to find
    submodules, and recursively calls `to_save_format` on them. This avoids
    a large amount of boilerplate code that would otherwise be needed in each
    module.

    To define additional (custom) logic implement `_to_save_format`.

    Args:
      params: The parameter tree before remapping.

    Returns:
      The remapped parameter tree, with each module's `save_format_metadata`
      value stored under the special key given by `METADATA_KEY`.
    """
    # Recursively call `to_save_format` for all submodules.
    params = self._submodules_to_save_format(params)

    # Copy `params` and convert to a more convenient dict type.
    params = RecursiveDefaultDict(params)
    # Apply any custom remapping logic.
    params = self._to_save_format(params)

    params[METADATA_KEY] = self.save_format_metadata
    return params.to_dict()

  @nn.nowrap  # Exempt from named call decorator.
  @flax.linen.module.wrap_method_once
  def _from_save_format(self,
                        params: RecursiveDefaultDict) -> Mapping[str, Any]:
    """Clients may override this method to add custom remapping logic."""
    return params

  @nn.nowrap  # Exempt from named call decorator.
  @flax.linen.module.wrap_method_once
  def _to_save_format(self, params: RecursiveDefaultDict) -> Mapping[str, Any]:
    """Clients may override this method to add custom remapping logic."""
    return params

  @nn.nowrap  # Exempt from named call decorator.
  @flax.linen.module.wrap_method_once
  def _submodules_from_save_format(
      self, params: Mapping[str, Any]) -> RecursiveDefaultDict:
    """Recursively calls `from_save_format` for all submodules."""
    # Copy `params` and convert to a more convenient dict type.
    result = RecursiveDefaultDict(params)
    for name, submodule in self._get_remappable_submodules():
      if name in params:
        result[name] = submodule.from_save_format(params[name])
    return result

  @nn.nowrap  # Exempt from named call decorator.
  @flax.linen.module.wrap_method_once
  def _submodules_to_save_format(
      self, params: Mapping[str, Any]) -> RecursiveDefaultDict:
    """Recursively calls `to_save_format` for all submodules."""
    # Copy `params` and convert to a more convenient dict type.
    result = RecursiveDefaultDict(params)
    for name, submodule in self._get_remappable_submodules():
      if name in params:
        result[name] = submodule.to_save_format(params[name])
    return result

  @nn.nowrap  # Exempt from named call decorator.
  @flax.linen.module.wrap_method_once
  def _get_remappable_submodules(self) -> List[Tuple[str, ParameterRemappable]]:
    """Returns the parameter-remappable submodules."""
    return [(e.name, e)
            for e in self._get_submodules()
            if isinstance(e, ParameterRemappable)]

  # TODO: There has got to be a library for doing this.
  @nn.nowrap  # Exempt from named call decorator.
  @flax.linen.module.wrap_method_once
  def _get_submodules(self) -> List[nn.Module]:
    """Returns a list of this object's submodules."""
    field_names: List[str] = []
    if dataclasses.is_dataclass(self):
      for field in dataclasses.fields(self):
        field_names.append(field.name)
    for field_name in vars(self):
      if field_name not in field_names:
        field_names.append(field_name)

    field_values: List[Any] = [
        getattr(self, field_name)
        for field_name in field_names
        if field_name not in ('name', 'parent')
    ]

    all_elements: List[ParameterRemappable] = []
    while field_values:
      field_value = field_values.pop(0)
      if isinstance(field_value, ParameterRemappable):
        all_elements.append(field_value)
      elif isinstance(field_value, (str, bytes)):
        # Explicitly skip strings since they are also instances of `Sequence`.
        continue
      elif isinstance(field_value, Mapping):
        field_values = list(field_value.values()) + field_values
      elif isinstance(field_value, Sequence):
        field_values = list(field_value) + field_values

    return [e for e in all_elements if isinstance(e, nn.Module)]


class RecursiveDefaultDict(collections.defaultdict):
  """A `defaultdict` that allows recursively indexing to an arbitrary depth.

  For example:

      d = RecursiveDefaultDict()
      d['a']['b']['c'] = 1
      d.to_dict()  # {'a': {'b': {'c': 1}}}

      d2 = RecursiveDefaultDict()
      d2['a'].merge('b2', d['a'].pop('b'))
      d.to_dict()   # {}
      d2.to_dict()  # {'a': {'b2': {'c': 1}}}
  """

  def __init__(self, initial_content: Optional[Mapping[Any, Any]] = None):
    super().__init__(RecursiveDefaultDict)
    if initial_content is not None:
      self.update(initial_content)

  def merge(self, key: Any, value: Any) -> None:
    """Adds the key-value pair, but recursively if `value` is a Mapping."""
    # If `key` is not yet present, then there is nothing to merge.
    if key not in self:
      if isinstance(value, Mapping):
        value = RecursiveDefaultDict(value)
      self[key] = value
      return

    # If `self[key]` is a RecursiveDefaultDict, then recursively merge `value`.
    if isinstance(self[key], RecursiveDefaultDict):
      if not isinstance(value, Mapping):
        raise ValueError('Cannot merge a non-mapping into a mapping; '
                         f'key: {key!r}; new value: ({value!r})')
      self[key].update(value)
      return

    # Otherwise `self[key]` is a leaf, and cannot be "merged" over.
    raise ValueError(f'Cannot overwrite existing leaf key {key!r} via merge')

  def update(self, other: Mapping[Any, Any]) -> None:
    """Calls `merge` for each key-value pair in `other`."""
    for k, v in other.items():
      self.merge(k, v)

  def pop(self, key: Any, default: Any = None) -> Any:
    return super().pop(key,
                       RecursiveDefaultDict() if default is None else default)

  def to_dict(self) -> Dict[Any, Any]:
    """Recursively converts this object to a regular `dict`.

    Returns:
      A 'dict' with this object's content. Note that entries with empty values
      are recursively filtered out.
    """
    result = {}
    for k, v in self.items():
      if isinstance(v, RecursiveDefaultDict):
        v = v.to_dict()
        if not v:
          continue
      result[k] = v
    return result


def filter_out_metadata(params: Mapping[str, Any]) -> Dict[str, Any]:
  """Removes "__save_format_metadata__" entries from a parameter tree."""
  result = {}
  for k, v in params.items():
    if k == METADATA_KEY:
      continue
    if isinstance(v, Mapping):
      v = filter_out_metadata(v)
      if not v:
        continue
    result[k] = v
  return result
