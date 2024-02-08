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

"""T5 Transformer model."""

# pylint: disable=attribute-defined-outside-init,g-bare-generic

import functools
import operator
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, Union

from aqt.jax_legacy.jax import flax_layers as aqt_flax_layers
from aqt.jax_legacy.jax import quant_config as aqt_config
from aqt.jax_legacy.jax import quantization as aqt

from flax import linen as nn
from flax.core import frozen_dict
from flax.linen import partitioning
from flax.linen.linear import default_kernel_init
from jax import lax
import jax.numpy as jnp
import numpy as np

from flaxformer import activation_partitioning
from flaxformer.types import Array
from flaxformer.types import DType
from flaxformer.types import Initializer


# ------------------------------------------------------------------------------
# Adafactor-compatible DenseGeneral for attention layers.
# ------------------------------------------------------------------------------
def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int, ...]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple([ax if ax >= 0 else ndim + ax for ax in axes])


def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


# The Flaxformer sharding API emits some names that are too detailed, so we
# remap them here. Any values that don't match keys here are unchanged.
_RESHAPED_KERNEL_AXIS_NAME_MAP = frozen_dict.freeze({
    'heads * kv': 'joined_kv',
})


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
      reshaped_kernel_axis_name_map: Rules for renaming fused kernel axes. We
        keep this as a separate parameter than kernel_axis_names so that
        experiments can toggle `reshape_kernel` without having to keep
        `kernel_axis_names` in sync.
      reshape_kernel: whether to reshape the kernel parameter to 2D for
        Adafactor.
  """
  features: Union[Iterable[int], int]
  use_bias: bool
  axis: Union[Iterable[int], int] = -1
  dtype: DType = jnp.float32
  kernel_init: Initializer = default_kernel_init  # pytype: disable=annotation-type-mismatch  # jax-types
  bias_init: Initializer = nn.initializers.zeros
  precision: Any = None
  kernel_axis_names: Optional[Sequence[str]] = None
  reshaped_kernel_axis_name_map: Mapping[str, str] = (
      _RESHAPED_KERNEL_AXIS_NAME_MAP)
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

    # Determine axes names metadata for partitioning/adafactor rules.
    if self.kernel_axis_names is None:
      kernel_axis_names = ['unmodeled'] * len(kernel_param_shape)
    else:
      kernel_axis_names = self.kernel_axis_names
      if len(kernel_axis_names) != len(kernel_shape):
        raise ValueError(f"Kernel axis names {kernel_axis_names} doesn't match "
                         f'kernel shape {kernel_shape}.')
      if self.reshape_kernel:

        def _reshaped_axis_names(names):
          result = ' * '.join(names)
          return self.reshaped_kernel_axis_name_map.get(result, result)

        kernel_axis_names = (
            _reshaped_axis_names(kernel_axis_names[:len(axis)]),
            _reshaped_axis_names(kernel_axis_names[len(axis):]),
        )

    kernel = partitioning.param_with_axes(
        'kernel',
        self.kernel_init,
        kernel_param_shape,
        jnp.float32,
        axes=tuple(kernel_axis_names))

    kernel = jnp.asarray(kernel, self.dtype)
    kernel = jnp.reshape(kernel, kernel_shape)

    contract_ind = tuple(range(0, len(axis)))
    out = lax.dot_general(
        inputs,
        kernel, ((axis, contract_ind), ((), ())),
        precision=self.precision)
    if self.use_bias:
      bias = partitioning.param_with_axes(
          'bias',
          self.bias_init, (np.prod(features),),
          jnp.float32,
          axes=(kernel_axis_names[-1],))
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
    wi_fused_kernel_init: Optional wi_fused kernel function, passed to the
      dense layers. If None, then kernel_init will be passed instead.
    bias_init: Bias initializer.
    enable_dropout: Enables non-deterministic dropout when set to True.
    intermediate_dropout_rate: Dropout rate used after the intermediate layers.
    final_dropout_rate: Dropout rate used after the final layer.
    intermediate_dropout: Optional Dropout layer used after the intermediate
      layers.
    final_dropout: Optional Dropout layer used after the final layer.
    dtype: Type for the dense layer.
    out_dim: Final dimension of the output. If not set, it will be the same as
      the input dimension.
    intermediate_conv: Optional module applied to the first factor of the
      intermediate layer, after activation.
    precomputed_intermediates: whether we're using outside W_i and W_o
      computations, merely using this layer for intermediate computations.
    fuse_kernels: whether to fuse the kernels for gated activation.
    input_axis_name: Axis name for input activations.
    activations_axis_name: Axis name for intermediate activations.
    intermediate_axis_name: Axis name for output activations.
    data_sharding_constraints: Sharding constraint for data. If unspecified
      (default), sharding constraints are inferred from the data shape; see
      _get_logical_axes().
    activation_partitioning_dims: Activation partition for the intermediate
      activations.
    use_aqt: Whether to use aqt quantization.
    weight_params: Parameters for weight quantization.
    act_params: Parameters for activation quantization.
  """
  use_bias: bool
  intermediate_dim: int = 2048
  activations: Sequence[Union[str, Callable]] = ('relu',)
  kernel_init: Callable = nn.initializers.xavier_uniform()
  wi_fused_kernel_init: Optional[Callable] = None
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)
  intermediate_dropout_rate: float = 0.1
  final_dropout_rate: float = 0.1
  intermediate_dropout: Optional[nn.Module] = None
  final_dropout: Optional[nn.Module] = None
  dtype: Any = jnp.float32
  out_dim: Optional[int] = None
  intermediate_conv: Optional[nn.Module] = None
  precomputed_intermediates: bool = False
  fuse_kernels: bool = False
  input_axis_name: str = 'embed'
  activations_axis_name: str = 'mlp_activations'
  intermediate_axis_name: str = 'mlp'
  output_axis_name: str = 'embed'
  data_sharding_constraints: Optional[Tuple[str, ...]] = None
  activation_partitioning_dims: Optional[int] = 2
  use_aqt: Optional[bool] = False
  weight_params: Optional[aqt.QuantOps.WeightParams] = None
  act_params: Optional[aqt.QuantOps.ActHParams] = None
  possibly_use_quantized_vars: bool = False
  dense_general_factory: Callable[..., nn.Module] = DenseGeneral

  @nn.compact
  def __call__(self,
               inputs,
               decode: bool = False,
               prefill: bool = False,
               prefill_lengths: Optional[Array] = None,
               *,
               enable_dropout: bool = True):
    """Applies Transformer MlpBlock module."""
    wi_fused_kernel_init = (
        self.wi_fused_kernel_init
        if self.wi_fused_kernel_init is not None else self.kernel_init)

    actual_out_dim = (
        inputs.shape[-1] if self.out_dim is None else self.out_dim)

    def dense(features, name, inputs, kernel_axis_names):
      if self.use_aqt:
        if self.weight_params is None and self.act_params is None:
          raise ValueError(
              'If use_aqt is True, either of weights or acts quantization need '
              'to be specified using arguments `weight_params` or `act_params`.'
          )
        # TODO: Push the "quantized vs not" decision down into the
        # AQT library. Currently we make that decision here, because the AQT
        # library doesn't support DenseGeneral, so there's extra reshapes here
        # whose performance impact I don't know.
        aqt_context = aqt_config.DynamicContext(
            update_bounds=False, collect_acts_stats=False)
        weight_prec = self.weight_params.prec if self.weight_params else None
        half_shift = self.weight_params.half_shift if self.weight_params else False
        aqt_hparams = aqt_flax_layers.DenseAqt.HParams(
            weight_prec=weight_prec,
            weight_half_shift=half_shift,
            quant_act=self.act_params,  # currently supports fixed bounds only.
            quant_type=aqt.QuantType.AQT,
            weight_quant_granularity=aqt_config.QuantGranularity.PER_CHANNEL,
        )
        batch, seq_len, channels = inputs.shape
        inputs = inputs.reshape((batch * seq_len, channels))

        result = aqt_flax_layers.DenseAqt(
            features=features,
            hparams=aqt_hparams,
            train=enable_dropout,
            dynamic_context=aqt_context,
            paxis_name=None,
            # No "cross-replica" reduction expressed in the XLA graph at this
            # stage. Will be imposed later, automatically, by XLA SPMD.
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            kernel_axis_names=kernel_axis_names,
            name=name,
            possibly_use_quantized_vars=self.possibly_use_quantized_vars,
        )(inputs, padding_mask=None)
        return result.reshape((batch, seq_len, features))
      else:
        return self.dense_general_factory(
            features=features,
            use_bias=self.use_bias,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            kernel_axis_names=kernel_axis_names,
            name=name,
        )(inputs)

    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
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
          x = dense(self.intermediate_dim, dense_name, inputs,
                    (self.input_axis_name, self.intermediate_axis_name))
          x = _convert_to_activation_function(act_fn)(x)
          if idx == 0 and self.intermediate_conv is not None:
            x = self.intermediate_conv(  # pylint: disable=not-callable
                x,
                decode=decode,
                prefill=prefill,
                prefill_lengths=prefill_lengths)
          activations.append(x)
    else:
      if self.weight_params is not None or self.act_params is not None:
        # TODO: need to make quantization work with fused kernels.
        raise NotImplementedError('Quantization is not supported yet for ',
                                  'fused kernels.')
      if self.precomputed_intermediates:
        if self.out_dim is None:
          raise ValueError('Must specify mlp out_dim when using precomputed '
                           'intermediates.')
        xs = inputs
      else:
        xs = self.dense_general_factory(
            features=(len(self.activations), self.intermediate_dim),
            use_bias=self.use_bias,
            dtype=self.dtype,
            kernel_init=wi_fused_kernel_init,
            bias_init=self.bias_init,
            reshape_kernel=False,
            kernel_axis_names=(
                self.input_axis_name,
                self.activations_axis_name,
                self.intermediate_axis_name,
            ),
            name='wi_fused',
        )(inputs)
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
    # TODO: Change the `None` branch to not applying dropout
    # instead of fallback to default dropout.
    if self.intermediate_dropout:
      x = self.intermediate_dropout(x, deterministic=not enable_dropout)  # pylint: disable=not-callable
    else:
      x = nn.Dropout(
          rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
              x, deterministic=not enable_dropout)  # Broadcast along length.

    # Note: We don't use `activation_partitioning.with_sharding_migration` here
    # because we do often want this 2D sharded. However, if rules are valid,
    # they should result in 2D sharding. We don't need to raise errors if both
    # result in 2D sharding (which with_sharding_migration does).
    if partitioning.get_axis_rules():
      logical_axis_resources = (
          self.data_sharding_constraints or _get_logical_axes(x))
      x = partitioning.with_sharding_constraint(
          x, logical_axis_resources=logical_axis_resources)
    else:
      x = activation_partitioning.with_sharding(
          x, self.activation_partitioning_dims)

    if self.precomputed_intermediates:
      # we fuse W_out and attention 'O' matrix outside.
      output = x
    else:
      output = dense(actual_out_dim, 'wo', x,
                     (self.intermediate_axis_name, self.output_axis_name))
      # TODO: Change the `None` branch to not applying dropout
      # instead of fallback to default dropout.
      if self.final_dropout:
        output = self.final_dropout(output, deterministic=not enable_dropout)  # pylint: disable=not-callable
      else:
        output = nn.Dropout(
            rate=self.final_dropout_rate, broadcast_dims=(-2,))(
                output, deterministic=not enable_dropout)
    return output


def _get_logical_axes(x: Array) -> Tuple[str, ...]:
  """Returns array-shape-dependent logical axis resources."""
  if x.ndim == 2:
    return ('length', 'mlp')
  elif x.ndim == 3:
    return ('batch', 'length', 'mlp')
  elif x.ndim == 4:
    return ('batch', 'length', 'heads', 'mlp_per_head')
  else:
    raise ValueError(
        f'Unexpected array shape. Cannot partition array of shape {x.shape}')
