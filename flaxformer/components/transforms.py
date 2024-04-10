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

"""Utility functions for jax transforms used by flaxformer."""

import dataclasses
import functools
import inspect
import flax
from flax import linen as nn
from flax.core.lift import Out as ScanOut  # pylint: disable=unused-import
from flax.linen import partitioning
import jax
from jax.interpreters import pxla
from jax.lax import with_sharding_constraint as jax_pjit_wsc

# TODO: this file contains JAX transform workarounds to fix/move
# upstream, primarily concerning the JAX checkpoint/remat transform and
# workarounds for issues with XLA SPMD and JAX scan transform.

# Workaround a scan(remat(...)) abstraction bug by manually implementing a
# static_argnums behavior for flax remat via closure before applying jax remat.

ScanIn = partitioning.ScanIn  # used in t5_architecture.py


def core_remat_static(fn,
                      variables=True,
                      rngs=True,
                      prevent_cse=True,
                      static_argnums=(),
                      policy=None):
  """Flax functional core remat version with static_argnums."""
  static_argnums = tuple(sorted(static_argnums))

  def _repack_remat_args(dyn_args, static_args):
    """Remake arg list from static and dynamic args given static_argnums."""
    args = []
    s_cnt, d_cnt = 0, 0
    for i in range(len(dyn_args) + len(static_args)):
      if i in static_argnums:
        args.append(static_args[s_cnt])
        s_cnt += 1
      else:
        args.append(dyn_args[d_cnt])
        d_cnt += 1
    return tuple(args)

  def inner(scope_fn, repack_fn, variable_groups, rng_groups, *args):
    static_args = tuple(x for i, x in enumerate(args) if i in static_argnums)
    dyn_args = tuple(x for i, x in enumerate(args) if i not in static_argnums)

    @functools.partial(jax.remat, prevent_cse=prevent_cse, policy=policy)
    @functools.wraps(fn)
    def rematted(variable_groups, rng_groups, *dyn_args):
      args = _repack_remat_args(dyn_args, static_args)
      scope = scope_fn(variable_groups, rng_groups)
      y = fn(scope, *args)
      return y, repack_fn(scope)

    return rematted(variable_groups, rng_groups, *dyn_args)

  return flax.core.lift.pack(
      inner, (variables,), (variables,), (rngs,), name='remat')


def remat(target,
          variables=True,
          rngs=True,
          prevent_cse=True,
          static_argnums=(),
          policy=None,
          methods=None):
  """Flax lifted remat that supports static_argnums."""
  return flax.linen.transforms.lift_transform(
      core_remat_static,
      target,
      variables=variables,
      rngs=rngs,
      prevent_cse=prevent_cse,
      static_argnums=static_argnums,
      policy=policy,
      methods=methods)


# Allow use of scan/remat on factory functions that return module instances.

# Flaxformer uses keyword-only arguments in its methods, which
# aren't natively supported by most JAX transforms.  We use canonicalizing
# method wrappers to present to jax a pure-positional version of the function.


def canonicalize_arguments(orig_fn):
  """Convert function to use positional arguments only.

  Args:
    orig_fn: callable with signature taking positional and keyword arguments,
      but not variadic *args or **kwargs.

  Returns:
    A version of orig_fn taking only positional arguments, and
    a conversion function that takes the original signature, binds
    it to provided mixed arguments, applies defaults, and returns
    a tuple of positional arguments to use with the transformed
    function.
  """
  sig = inspect.signature(orig_fn)
  params = sig.parameters

  def dekwarged_fn(*args):
    new_args = []
    new_kwargs = {}
    if len(args) != len(params):
      raise ValueError(f'Incorrect number of arguments: '
                       f'got {len(args)}, expected {len(params)}.')
    for i, p in enumerate(params):
      param = params[p]
      if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
        new_args.append(args[i])
      elif param.kind is param.KEYWORD_ONLY:
        new_kwargs[p] = args[i]
      elif param.kind is param.VAR_POSITIONAL:
        new_args.extend(args[i])
      elif param.kind is param.VAR_KEYWORD:
        new_kwargs.update(args[i])
      else:
        raise ValueError('Unknown signature parameter type.')
    return orig_fn(*new_args, **new_kwargs)

  # We don't use functools.wraps because we are changing the signature, but
  # we want function properties preserved.
  dekwarged_fn.__dict__.update(orig_fn.__dict__)

  def convert_to_args(*args, **kwargs):
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return tuple(bound.arguments.values())

  return dekwarged_fn, convert_to_args


def canonicalized_class_transform(trafo, clz, *t_args, **t_kwargs):
  """Applies kwarg canonicalization with flax transform to module class clz.

  NB: This function only handles transforming the __call__ method.

  Args:
    trafo: flax lifted transform (e.g. nn.scan, nn.remat)
    clz: nn.Module class to transform
    *t_args: transform arguments
    **t_kwargs: transform keyword-arguments

  Returns:
    A transformed version of clz whose __call__ function has been transformed,
    additionally handling canonicalization of __call__'s signature to a purely
    positional function before applying the transform.
  """
  # Transform postitional only __call__ form of clz.
  dekwarged_fn, convert_to_args = canonicalize_arguments(clz.__call__)
  trafo_fn = trafo(dekwarged_fn, *t_args, **t_kwargs)

  @functools.wraps(clz.__call__)
  def post_fn(self, *args, **kwargs):
    return trafo_fn(*convert_to_args(self, *args, **kwargs))

  return type(trafo.__name__.capitalize() + clz.__name__, (clz,),
              {'__call__': post_fn})


# Flaxformer uses factory functions instead of partial constructors, we need
# to add a explicit handler for dealing with this case as our usual lifting
# API has no way to distinguish a factory function from a class method.


def apply_transform_to_module_factory(trafo, factory, *args, **kwargs):
  """Fix to apply flax transforms to a module factories via re-instantiation."""

  def new_factory():
    # Create the Module instance from the factory in a disconnected dynamic
    # context solely to collect the construction arguments and class.
    nn.module._context.module_stack.append(None)  # pylint: disable=protected-access
    try:
      inst = factory()
      ctor_args = {
          f.name: object.__getattribute__(inst, f.name)
          for f in dataclasses.fields(inst)
          if f.name not in ('parent', 'name')
      }
    finally:
      nn.module._context.module_stack.pop()  # pylint: disable=protected-access
    # Instantiate the transformed module class with gathered construction args
    # in the current dynamic context.
    return canonicalized_class_transform(trafo, inst.__class__, *args,
                                         **kwargs)(**ctor_args)

  return new_factory


factory_remat = functools.partial(apply_transform_to_module_factory,
                                  partitioning.remat)
factory_scan = functools.partial(apply_transform_to_module_factory,
                                 partitioning.scan_with_axes)
factory_vmap = functools.partial(apply_transform_to_module_factory, nn.vmap)

# Scan inner-function SPMD re-annotation.

# The XLA SPMD subsystem currently "loses" annotations on parameter trees that
# pass through an XLA while loop.  We should investigate fixing this at the XLA
# level, but the workaround for now is to re-apply the known sharding
# information for the scanned layer -inside- the functionalized scan body
# function using pjit's with_sharding_constraint.


def global_mesh_defined():
  """Checks if global xmap/pjit mesh resource environment is defined."""
  maps_env = pxla.thread_resources.env
  return maps_env.physical_mesh.devices.shape != ()  # pylint: disable=g-explicit-bool-comparison


def with_sharding_constraint(x, axis_resources):
  """Wrapper for lax.with_sharding_constraint, no-op on cpu or outside pjit."""
  if jax.devices()[0].platform == 'cpu' or not global_mesh_defined():
    return x
  else:
    return jax_pjit_wsc(x, axis_resources)


def inner_scan_spmd(annotation_tree, scan_axis):
  """Workaround to apply a sharding annotation pytree inside a scan body fn.

  This creates a function to be passed to nn.scan's "data_transform" kwarg.

  Args:
    annotation_tree: pytree of PartitionSpecs
    scan_axis: The axis along which layer parameters were scanned.

  Returns:
    A function to be used with nn.scan's data_transform kwarg to apply the
    SPMD annotations on the inner scan body function.
  """
  if annotation_tree is None:
    return None

  # The annotation tree fed through the model is the scan-expanded one,
  # we need to remove the scan axis from these PartitionSpecs.
  def del_axis(x):
    tmp = list(x)
    tmp.pop(scan_axis)
    return type(x)(*tmp)

  annotation_tree = jax.tree.map(del_axis, annotation_tree)

  def annotate_fn(variable_groups, rng_groups):
    broadcast_vars, carry_vars, *scan_variable_groups = variable_groups

    def maybe_annotate_group(x):
      if tuple(x[0].keys()) == ('params',):
        return ({
            'params': with_sharding_constraint(x[0]['params'], annotation_tree)
        },)
      else:
        return x

    scan_variable_groups = tuple(
        map(maybe_annotate_group, scan_variable_groups))
    variable_groups = (broadcast_vars, carry_vars) + scan_variable_groups
    return variable_groups, rng_groups

  return annotate_fn
