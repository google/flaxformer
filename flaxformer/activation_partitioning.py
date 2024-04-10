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

"""APIs to assist with partitioning activations."""
import traceback
from typing import Optional, Tuple, TypeVar

from absl import logging
from flax.linen import partitioning as flax_partitioning
import jax
from jax.interpreters import pxla
from jax.lax import with_sharding_constraint as jax_pjit_wsc


def global_mesh_defined():
  """Checks if global xmap/pjit mesh resource environment is defined."""
  maps_env = pxla.thread_resources.env
  return maps_env.physical_mesh.devices.shape != ()  # pylint: disable=g-explicit-bool-comparison


def with_sharding(x, partitioning_dims: int):
  """Annotate an activation for pjit sharding, no-op on cpu or outside pjit.

  These are sharding annotations for the XLA SPMD automatic partitioner (the
  system described in https://arxiv.org/abs/2105.04663 and exposed in JAX
  through `pjit`, `pmap`, and `xmap`. They are always semantically identity
  functions with regard to the program's output, but they create constraints on
  the sharding assignments that the partitioner can choose.

  Seemingly contradictory code like this:
  ```
  x = foo(x)
  x = with_sharding(x, 1)
  x = with_sharding(x, 2)
  x = bar(x)
  ```
  constrains the partitioner to compute `foo` with 1D (data-parallel) sharding
  and `bar` with 2D (data- and model-parallel sharding), and perform resharding
  of x in between.

  The motivation for adding these annotations inside transformer layers is that
  we want to support two different strategies for sharding the activations saved
  between the forward and backward passes, in the case where we are using both
  data and model parallelism for the model overall (so if you're implementing a
  layer that you don't expect to ever be used with model parallelism, you can
  stop worrying about any of this).

  While some activations will "naturally" be sharded with both data and model
  parallelism (i.e., hidden activations inside the MLP and attention activations
  that have a heads dimension), others (i.e. the 2-3 activations in each
  transformer block that have shape batch by sequence by model) can either be
  data-and-model sharded or just data-sharded, and it's very difficult for the
  partitioner to make a good choice one way or the other.

  Sharding these activations with just data parallelism (`partitioning_dims=1`)
  increases the memory used for storing them by a factor of the model-parallel
  axis size, and also makes layer norm perform redundant computation between
  model-parallel cores. On the other hand, sharding them with both data and
  model parallelism (`partitioning_dims=2`) means extra communication to scatter
  them from their producer ops and gather them for their consumer ops. This kind
  of memory vs. communication tradeoff is currently best left up to the user.

  Some heuristics for choosing when to use `with_sharding` and what value of
  `partitioning_dims` to use:
    * `partitioning_dims=2` is the only way to fit some very largest models.
    * `partitioning_dims=2` can substantially increase the training batch size
      that fits in memory, so it can sometimes be an alternative to gradient
      accumulation.
    * On the flip side, using both `partitioning_dims=2` and gradient
      accumulation at the same time is discouraged and might currently be buggy.
    * `partitioning_dims=1` is essentially always faster for inference/decoding
      than `partitioning_dims=2`.

  Args:
    x: The array to be annotated with pjit sharding constraints.
    partitioning_dims: The number of model-parallel shards to create.

  Returns:
    `x` with the specified pjit constraints.
  """
  if jax.devices()[0].platform == 'cpu' or not global_mesh_defined():
    return x
  else:
    if partitioning_dims == 1:
      return jax_pjit_wsc(x, jax.sharding.PartitionSpec('data'))
    elif partitioning_dims == 2:
      if x.ndim == 3:
        return jax_pjit_wsc(x,
                            jax.sharding.PartitionSpec('data', None, 'model'))  # pytype: disable=wrong-arg-count,wrong-arg-types
      elif x.ndim == 4:
        return jax_pjit_wsc(x,
                            jax.sharding.PartitionSpec('data', None, 'model',
                                                       None))  # pytype: disable=wrong-arg-count,wrong-arg-types
      else:
        raise ValueError(
            f'do not know how to partition array of shape {x.shape}')
    else:
      raise ValueError('only 1D or 2D activation partitioning is supported, '
                       f'got {partitioning_dims}')


T = TypeVar('T')


def with_sharding_migration(
    x: T,
    activation_partitioning_dims: Optional[int],
    logical_axis_names: Tuple[str, ...],
) -> T:
  """Helper function for migrating from old to new sharding annotations.

  Calls to this function were previously `with_sharding(x, dims)` (where the
  latter argument is defaulted to 1), and will become
  `flax_partitioning.with_sharding_constraint(x, logical_axis_names)`.

  Currently, if `activation_partitioning_dims` is unset, then the new logic will
  be used (it effectively does not issue a sharding annotation if there are no
  logical to physical mapping rules). If it is set, then a warning is issued,
  and it is used in all cases except when it is 1, where with standard logical
  axis rules, it is equivalent.

  Therefore, this function _mostly_ preserves the old semantics, but exercises
  the new codepath whenever possible.

  Args:
    x: Input array.
    activation_partitioning_dims: List of activation partitioning dimensions.
    logical_axis_names: List of names for each axis in `x`.

  Returns:
    Version of `x` with sharding annotations attached.
  """
  if activation_partitioning_dims is not None:
    last_tb = traceback.extract_stack()[-2]
    logging.log_first_n(
        logging.WARNING,
        'In %s:%d, activation_partitioning_dims was set, but it '
        'is deprecated and will be removed soon.', 10, last_tb.filename,
        last_tb.lineno)
    if not flax_partitioning.get_axis_rules():
      # If logical axis rules are not present, fall back to old behavior.
      return with_sharding(x, activation_partitioning_dims)
    else:
      if activation_partitioning_dims != 1:
        raise ValueError(
            'Both logical axis rules and activation_partitioning_dims'
            ' were present! This can typically be fixed by setting '
            '`ACTIVATION_PARTITIONING_DIMS = None` in your configuration so '
            'logical axis rules can be used instead.')
      else:
        return flax_partitioning.with_sharding_constraint(x, logical_axis_names)
  else:
    return flax_partitioning.with_sharding_constraint(x, logical_axis_names)
