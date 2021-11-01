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

"""T5 Transformer model."""

# pylint: disable=attribute-defined-outside-init,g-bare-generic

import functools
import operator
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

from flax import linen as nn
from flax.linen.linear import default_kernel_init
from jax import lax
import jax.numpy as jnp
import numpy as np

from flaxformer import activation_partitioning
from flaxformer import sharding
from flaxformer.types import Array
from flaxformer.types import DType
from flaxformer.types import Initializer


#------------------------------------------------------------------------------
# Adafactor-compatible DenseGeneral for attention layers.
#------------------------------------------------------------------------------
def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int, ...]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple([ax if ax >= 0 else ndim + ax for ax in axes])


def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


class DenseGeneral(nn.Module):
  """A linear transformation with flexible axes.

    Kernel stored as 2d parameter for compatibility with Adafactor optimizer.

    Attributes:
      features: tuple with numbers of output features.
      use_bias: whether to add a bias to the output (default: False).
      axis: tuple with axes to apply the transformation on.
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
      bias_init: initializer function for the bias.
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_axis_names: logical axis names to use for kernel sharding. Each
        should be one of _VALID_AXIS_NAMES in sharding.py.
      reshape_kernel: whether to reshape the kernel parameter to 2D for
        Adafactor.
  """
  features: Union[Iterable[int], int]
  use_bias: bool
  axis: Union[Iterable[int], int] = -1
  dtype: DType = jnp.float32
  kernel_init: Initializer = default_kernel_init
  bias_init: Initializer = nn.initializers.zeros
  precision: Any = None
  kernel_axis_names: Optional[Sequence[str]] = None
  reshape_kernel: bool = True

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    features = _canonicalize_tuple(self.features)
    axis = _canonicalize_tuple(self.axis)

    inputs = jnp.asarray(inputs, self.dtype)
    axis = _normalize_axes(axis, inputs.ndim)

    kernel_shape = tuple([inputs.shape[ax] for ax in axis]) + features
    if self.reshape_kernel:
      kernel_param_shape = (np.prod([inputs.shape[ax] for ax in axis]),
                            np.prod(features))
    else:
      kernel_param_shape = kernel_shape
    kernel = self.param('kernel', self.kernel_init, kernel_param_shape,
                        jnp.float32)

    # Sow axis names as metadata for partitioning/adafactor rules.
    if self.kernel_axis_names is None:
      kernel_axis_names = ['unmodeled'] * len(kernel_param_shape)
    else:
      kernel_axis_names = self.kernel_axis_names
      if len(kernel_axis_names) != len(kernel_shape):
        raise ValueError(f"Kernel axis names {kernel_axis_names} doesn't match "
                         f'kernel shape {kernel_shape}.')
      if self.reshape_kernel:
        kernel_axis_names = (' * '.join(kernel_axis_names[:len(axis)]),
                             ' * '.join(kernel_axis_names[len(axis):]))
    self.sow(
        'param_axes',
        'kernel_axes',
        sharding.axis_names(*kernel_axis_names),
        reduce_fn=sharding.reduce_fn)

    kernel = jnp.asarray(kernel, self.dtype)
    kernel = jnp.reshape(kernel, kernel_shape)

    contract_ind = tuple(range(0, len(axis)))
    out = lax.dot_general(
        inputs,
        kernel, ((axis, contract_ind), ((), ())),
        precision=self.precision)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (np.prod(features),),
                        jnp.float32)
      self.sow(
          'param_axes',
          'bias_axes',
          sharding.axis_names(kernel_axis_names[-1]),
          reduce_fn=sharding.reduce_fn)
      bias = jnp.asarray(bias, self.dtype)
      bias = jnp.reshape(bias, features)
      # Reshape bias for broadcast.
      expand_dims = sorted(set(range(inputs.ndim)) - set(axis))
      for ax in expand_dims:
        bias = jnp.expand_dims(bias, ax)
      out = out + bias
    return out


def _convert_to_activation_function(
    fn_or_string: Union[str, Callable]) -> Callable:
  """Convert a string to an activation function."""
  if fn_or_string == 'linear':
    return lambda x: x
  elif isinstance(fn_or_string, str):
    return getattr(nn, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    raise ValueError("don't know how to convert %s to an activation function" %
                     (fn_or_string,))


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.


  Attributes:
    use_bias: Whether to use bias in the dense layers.
    intermediate_dim: Shared dimension of hidden layers.
    activations: Type of activations for each layer.  Each element is either
      'linear', a string function name in flax.linen, or a function.
    kernel_init: Kernel function, passed to the dense layers.
    bias_init: Bias initializer.
    enable_dropout: Enables non-deterministic dropout when set to True.
    intermediate_dropout_rate: Dropout rate used after the intermediate layers.
    final_dropout_rate: Dropout rate used after the final layer.
    dtype: Type for the dense layer.
    out_dim: Final dimension of the output. If not set, it will be the same as
      the input dimenion.
    intermediate_conv: Optional module applied to the first factor of the
      intermediate layer, after activation.
    precomputed_intermediates: whether we're using outside W_i and W_o
      computations, merely using this layer for intermediate computations.
    fuse_kernels: whether to fuse the kernels for gated activation.
  """
  use_bias: bool
  intermediate_dim: int = 2048
  activations: Sequence[Union[str, Callable]] = ('relu',)
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)
  intermediate_dropout_rate: float = 0.1
  final_dropout_rate: float = 0.1
  dtype: Any = jnp.float32
  out_dim: Optional[int] = None
  intermediate_conv: Optional[nn.Module] = None
  precomputed_intermediates: bool = False
  fuse_kernels: bool = False

  @nn.compact
  def __call__(self,
               inputs,
               decode: bool = False,
               prefill: bool = False,
               prefill_lengths: Optional[Array] = None,
               *,
               enable_dropout: bool = True):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = (
        inputs.shape[-1] if self.out_dim is None else self.out_dim)

    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('linear', 'gelu') for gated-gelu.
    activations = []
    # TODO: don't bother w/ fusion if only a single input matrix?
    if not self.fuse_kernels:
      if self.precomputed_intermediates:
        for idx, (inpt, act_fn) in enumerate(zip(inputs, self.activations)):
          x = _convert_to_activation_function(act_fn)(inpt)
          if idx == 0 and self.intermediate_conv is not None:
            x = self.intermediate_conv(  # pylint: disable=not-callable
                x,
                decode=decode,
                prefill=prefill,
                prefill_lengths=prefill_lengths)
          activations.append(x)
      else:
        for idx, act_fn in enumerate(self.activations):
          dense_name = 'wi' if len(self.activations) == 1 else f'wi_{idx}'
          x = DenseGeneral(
              self.intermediate_dim,
              use_bias=self.use_bias,
              dtype=self.dtype,
              kernel_init=self.kernel_init,
              bias_init=self.bias_init,
              kernel_axis_names=['embed', 'intermediate'],
              name=dense_name)(
                  inputs)
          x = _convert_to_activation_function(act_fn)(x)
          if idx == 0 and self.intermediate_conv is not None:
            x = self.intermediate_conv(  # pylint: disable=not-callable
                x,
                decode=decode,
                prefill=prefill,
                prefill_lengths=prefill_lengths)
          activations.append(x)
    else:
      if self.precomputed_intermediates:
        if self.out_dim is None:
          raise ValueError('Must specify mlp out_dim when using precomputed '
                           'intermediates.')
        xs = inputs
      else:
        xs = DenseGeneral(
            (len(self.activations), self.intermediate_dim),
            use_bias=self.use_bias,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            reshape_kernel=False,
            kernel_axis_names=['embed', 'unmodeled', 'intermediate'],
            name='wi_fused')(
                inputs)
      for idx, act_fn in enumerate(self.activations):
        x = jnp.squeeze(lax.dynamic_slice_in_dim(xs, idx, 1, -2), -2)
        x = _convert_to_activation_function(act_fn)(x)
        if idx == 0 and self.intermediate_conv is not None:
          x = self.intermediate_conv(  # pylint: disable=not-callable
              x,
              decode=decode,
              prefill=prefill,
              prefill_lengths=prefill_lengths)
        activations.append(x)

    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations)
    # Apply dropout and final dense output projection.
    x = nn.Dropout(
        rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=not enable_dropout)  # Broadcast along length.
    x = activation_partitioning.with_sharding(x, 2)
    if self.precomputed_intermediates:
      # we fuse W_out and attention 'O' matrix outside.
      output = x
    else:
      output = DenseGeneral(
          actual_out_dim,
          use_bias=self.use_bias,
          dtype=self.dtype,
          kernel_init=self.kernel_init,
          bias_init=self.bias_init,
          kernel_axis_names=['intermediate', 'embed'],
          name='wo')(
              x)
      output = nn.Dropout(
          rate=self.final_dropout_rate, broadcast_dims=(-2,))(
              output, deterministic=not enable_dropout)
    return output
