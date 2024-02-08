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

"""APIs for emitting sharding annotations from Flaxformer."""

import re

from flax import traverse_util
from flax.core import frozen_dict
from flax.linen import partitioning


class AxisNames(tuple):
  """Tuple of string names for each axis, for use outside of jax.jit.

  We create a separate class for this so JAX's pytree utilities can distinguish
  it from a tuple that should be treated as a pytree, instead treating it as a
  leaf.
  """
  # TODO: Use t5x.partitioning.AxisNames once this is migrated into t5x.
  pass


def axis_names(*names: str) -> partitioning.AxisMetadata:
  """Generates axis name metadata to be sown.

  Args:
    *names: Names for each parameter axis.

  Returns:
    partitioning.AxisMetadata metadata struct.
  """
  return partitioning.AxisMetadata(names=names)


def reduce_fn(x, y):
  """Reduction function for sow() calls.

  Args:
    x: Existing value, or () if there was none.
    y: New axis names sown.

  Returns:
    New axis names.
  """
  if not isinstance(y, partitioning.AxisMetadata):
    raise TypeError(
        "Expected newly sown value to be an partitioning.AxisMetadata")

  if isinstance(x, partitioning.AxisMetadata):
    if x != y:
      raise ValueError("If axis names are sown twice, expected them to match. "
                       f"Got {x} and {y}.")
  elif x:
    # Shouldn't happen, so raise a fairly internal error.
    raise AssertionError(f"Non-initial-or-AxisNames value encountered: {x}")
  return y


def _get_single_sowed_value(value) -> AxisNames:
  """Checks that a sown value is as expected.

  Args:
    value: Pytree leaf node, after calling traverse_util.flatten_dict().

  Returns:
    AxisNames metadata struct.

  Raises:
    TypeError: If any objects are of the wrong type.
  """
  if not isinstance(value, partitioning.AxisMetadata):
    raise TypeError(
        "Expected partitioning.AxisMetadata, please make sure to use "
        "`reduce_fn`. Got {value}")
  return AxisNames(value.names)


def get_axis_names(variables):
  """Gets axis names for variables.

  Args:
    variables: Flax variables struct, either from `model.init(...)` or
      `jax.eval_shape(model.init, ...)`.

  Returns:
    Struct matching `variables` with sown `AxisNames` as leaves.
  """
  variables = frozen_dict.unfreeze(variables)  # pytype: disable=wrong-arg-types
  flat_param_axes = traverse_util.flatten_dict(variables["params_axes"])
  flat_axis_names = {}
  for keys, v in flat_param_axes.items():
    # Remove '_axes' suffix from axis metadata path to match param tree.
    flat_param_key = tuple(re.sub(r"_axes$", "", k) for k in keys)
    flat_axis_names[flat_param_key] = _get_single_sowed_value(v)

  return traverse_util.unflatten_dict(flat_axis_names)


def check_params_and_axis_names_match(variables):
  """Checks that parameters and axis names match.

  This means that every parameter should have axis name metadata associated with
  it. It also checks that each parameter dimension has a name.

  Args:
    variables: Flax variables struct, either from `model.init(...)` or
      `jax.eval_shape(model.init, ...)`.

  Raises:
    ValueError: If axis names don't exist, or don't match the param shape.
  """
  variables = frozen_dict.unfreeze(variables)

  def _flatten_with_joined_names(xs):
    return {"/".join(k): v for k, v in traverse_util.flatten_dict(xs).items()}

  flat_params = _flatten_with_joined_names(variables["params"])
  flat_axis_names = _flatten_with_joined_names(get_axis_names(variables))
  for key, array in flat_params.items():
    if key not in flat_axis_names:
      raise ValueError(f"Axis names were not sow'd for {key}.")
    names = flat_axis_names[key]
    if len(array.shape) != len(names):
      raise ValueError(
          f"For {key}, axis names ({names}) doesn't contain one name "
          "for each parameter dimension (shape {array.shape})")
