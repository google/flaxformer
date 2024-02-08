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

"""Mixture of Experts layer.

"""

import functools
from typing import Optional, Tuple, Union

from absl import logging
from flax import linen as nn
from flax.linen import partitioning as flax_partitioning
import jax
import jax.numpy as jnp

from flaxformer.architectures.moe import routing
from flaxformer.architectures.moe import scatter_utils
from flaxformer.components import dense
from flaxformer.types import Array
from flaxformer.types import DType


class MoeLayer(nn.Module):
  """Sparse MoE SPMD layer with per-token routing.

  Attributes:
    num_experts: Number of available experts (feed-forward modules) in this
      layer.
    max_group_size: The total number of tokens (across the global batch) is
      subdivided into groups of this size, on each device. Router computations
      are then performed on a per-group basis. A larger group size will result
      in slower but more accurate top-k and sorting computations, whereas a
      smaller group size will result in faster but more approximate (and
      potentially less stable) routing choices. Note that actual group size may
      be smaller than max_group_size for consistency with the number of experts
      and tokens; see also `strict_group_size` attribute. In practice, we find
      that imperfect routing choices are tolerable and recommend choosing a
      group size on the order of 4096 tokens, although this number will vary
      based on model configuration and size.
    train_capacity_factor: Scaling factor to increase the expert token capacity
      during training. This factor plays an analogous, but slightly different,
      role depending on the routing assignment algorithm: For "tokens choose"
      routing, the capacity factor only affects the maximum number of tokens
      that an expert will process. It does not affect how many experts a given
      token is routed to; see the num_selected_experts attributes of "tokens
      choose" routers. For "experts choose" routing, because experts always fill
      their buffer, increasing the capacity factor will increase the number of
      tokens that an expert will process AND will indirectly increase the number
      of experts that a given token is routed to.
    eval_capacity_factor: As above, but used during evaluation.
    expert: The actual expert. Only MlpBlock and DenseGeneral are currently
      supported.
    router: Token dispatch router. The router determines which tokens are
      dispatched to which expert, and how the expert outputs are combined.
    num_expert_partitions: Specifies the upper bound for size of the expert
      parallel submesh. This must be <= the number of experts.
    num_model_partitions: Size of the model parallel submesh. Model parallelism
      is used if num_model_partitions > 1.
    min_expert_capacity: Minimum token processing capacity for each expert.
    dropout_rate: Dropout rate for each expert.
    input_hidden_dims_axes: Logical axis names to use for sharding constraints
      applied to hidden dimensions of inputs (before experts are called).
    output_hidden_dims_axes: Logical axis names to use for sharding constraints
      applied to hidden dimensions of outputs (after experts are called).
    dtype: The numeric type (default: bfloat16). We recommend a truncated float
      type (e.g. bfloat16) to reduce all-to-all communication overhead. This
      numeric type is used for all computations, except the router, which always
      uses float32 for stability.
    split_params: Whether or not to initialize each expert's parameters
      independently.
    precision: XLA precision for array computations.
    strict_group_size: If True, fail if unable to set the token group size equal
      to max_group_size. If False (default), the actual group size may be
      smaller than max_group_size for consistency with the number of experts and
      tokens.
  """

  num_experts: int
  max_group_size: int
  # TODO: Switch to a single `capacity_factor` once we are using
  #  Fiddle to build different train vs eval model variants.
  train_capacity_factor: float
  eval_capacity_factor: float
  expert: Union[dense.MlpBlock, dense.DenseGeneral]
  router: routing.Router
  num_expert_partitions: int
  num_model_partitions: int
  min_expert_capacity: int = 4
  dropout_rate: float = 0.1
  input_hidden_dims_axes: Tuple[str, ...] = ('embed',)
  output_hidden_dims_axes: Tuple[str, ...] = ('embed',)
  dtype: DType = jnp.bfloat16
  split_params: bool = True
  precision: jax.lax.Precision = jax.lax.Precision.DEFAULT
  strict_group_size: bool = False

  def setup(self):
    """Verifies that the MoeLayer is correctly configured."""
    if self.num_expert_partitions > self.num_experts:
      raise ValueError(
          f'The number of expert partitions ({self.num_expert_partitions}) '
          f'cannot be greater than the number of experts ({self.num_experts}).'
      )

    self.num_expert_replicas = _num_expert_replicas(
        self.num_expert_partitions, self.num_model_partitions
    )

  @nn.compact
  def __call__(
      self,
      inputs: Array,
      decode: bool = False,
      prefill: bool = False,
      prefill_lengths: Optional[Array] = None,
      *,
      enable_dropout: bool = True,
  ) -> Array:
    """Applies MoeLayer module.

    If the 'intermediates' collection is marked as mutable, this method will sow
    diversity metrics.

    Args:
      inputs: Batch of input embeddings of shape <float>[batch_size, seq_length,
        hidden_dims].
      decode: Whether to prepare and use an autoregressive cache.
      prefill: Whether to run a partial sequence to prefill the cache.
      prefill_lengths: The length of each partial sequence we are filling in the
        cache.
      enable_dropout: Enables dropout if set to True.

    Returns:
      Transformed inputs with same shape as inputs:
      <float>[batch_size, seq_length, hidden_dims].

    Raises:
      ValueError if an unrecognized dispatch algorithm is given.
    """
    original_batch_size, original_seq_length, *hidden_dims = inputs.shape

    padded_inputs = _maybe_pad(
        inputs, self.num_experts, self.num_expert_replicas
    )
    padded_batch_size, padded_seq_length, *_ = padded_inputs.shape

    num_tokens = padded_batch_size * padded_seq_length

    num_groups = _num_groups(
        num_tokens,
        self.max_group_size,
        self.num_experts,
        self.num_expert_replicas,
        self.strict_group_size,
    )
    tokens_per_group = num_tokens // num_groups

    if enable_dropout:  # Training
      capacity_factor = self.train_capacity_factor
    else:  # Eval
      capacity_factor = self.eval_capacity_factor
    # Each group will send expert_capacity tokens to each expert.
    expert_capacity = int(
        round(capacity_factor * tokens_per_group / self.num_experts)
    )
    expert_capacity = max(expert_capacity, self.min_expert_capacity)

    # Reshape batch and sequence/token dimensions for expert routing.
    grouped_inputs = jnp.reshape(
        padded_inputs, (num_groups, tokens_per_group, *hidden_dims)
    )
    grouped_inputs = flax_partitioning.with_sharding_constraint(
        grouped_inputs, ('batch', 'length', *self.input_hidden_dims_axes)
    )

    if isinstance(self.router, routing.ScatterRouter):
      outputs = self._scatter_to_experts(
          grouped_inputs,
          enable_dropout,
          expert_capacity,
          decode=decode,
          prefill=prefill,
          prefill_lengths=prefill_lengths,
      )
    elif isinstance(self.router, routing.MaskedRouter):
      outputs = self._mask_and_dispatch_to_experts(
          grouped_inputs,
          enable_dropout,
          expert_capacity,
          decode=decode,
          prefill=prefill,
          prefill_lengths=prefill_lengths,
      )
    else:
      raise ValueError(f'Unrecognized router type: {self.router}')

    # Return to batched shape.
    result = outputs.reshape(
        (padded_batch_size, padded_seq_length, *outputs.shape[2:])
    )
    if (
        padded_seq_length - original_seq_length > 0
        or padded_batch_size - original_batch_size > 0
    ):
      # Inputs were padded in MoE layer. Slice out non-padding tokens.
      result = result[:original_batch_size, :original_seq_length]
    result = flax_partitioning.with_sharding_constraint(
        result, ('batch', 'length', *self.output_hidden_dims_axes)
    )
    return result

  def _scatter_to_experts(
      self,
      token_inputs: Array,
      enable_dropout: bool,
      expert_capacity: int,
      **kwargs,
  ) -> Array:
    """Wraps expert scatter routing and dispatching algorithm.

    This algorithm takes the following steps:
    (1) Compute expert dispatch indices and combine weights using self.router.
    (2) Scatter inputs to experts based on dispatch indices.
    (3) Recombine individual expert outputs using combine weights.

    Args:
      token_inputs: <float>[num_groups, tokens_per_group, hidden_dims] inputs to
        send to experts.
      enable_dropout: If true, apply jitter noise during routing and dropout
        during expert computation.
      expert_capacity: Each group will send this many tokens to each expert.
      **kwargs: Optional keyword arguments to pass to experts.

    Returns:
      <float>[num_groups, tokens_per_group, hidden_dims] outputs from experts.
    """
    num_groups, tokens_per_group, *hidden_dims = token_inputs.shape

    router_indices = self.router(
        token_inputs,
        self.num_experts,
        expert_capacity,
        apply_jitter=enable_dropout,
    )
    num_selected_experts = self.router.num_selected_experts

    # We need num_selected_experts copies of inputs for dispatching. This is a
    # no-op if num_selected_experts = 1.
    token_inputs = jnp.repeat(token_inputs, num_selected_experts, axis=1)

    # Mask out inputs that should not be routed.
    # Shape: [num_groups, tokens_per_group, num_selected_experts].
    successfully_routed = jnp.logical_and(
        router_indices.dispatch_indices[..., 0] < self.num_experts,
        router_indices.dispatch_indices[..., 1] < expert_capacity,
    )
    successfully_routed = successfully_routed.reshape((num_groups, -1))
    # Shape: [num_groups, tokens_per_group * num_selected_experts, hidden_dims].
    masked_inputs = jnp.einsum(
        'gt...,gt->gt...',
        token_inputs,
        successfully_routed,
        precision=self.precision,
    )

    # Combine tokens_per_group and num_selected_experts axes.
    flattened_dispatch_indices = router_indices.dispatch_indices.reshape(
        num_groups, -1, 2
    )

    # Scatter masked inputs.
    shape = (self.num_experts, expert_capacity, *hidden_dims)
    # Note: scatter_nd can be slow under pjit on TPUs, presumably due to
    # suboptimal SPMD compilations. On TPUs, the recommendation is to use
    # MaskedRouter(s) instead.
    # Shape: [num_groups, num_experts, expert_capacity, hidden_dims].
    expert_inputs = jax.vmap(
        lambda i, x: scatter_utils.scatter_nd(i, x, shape)
    )(flattened_dispatch_indices, masked_inputs)

    expert_outputs = self._call_experts(expert_inputs, enable_dropout, **kwargs)

    # Gather outputs.
    # Shape: [num_groups, tokens_per_group * num_selected_experts, hidden_dims].
    expert_outputs = jax.vmap(lambda i, x: x[i[:, 0], i[:, 1]])(
        flattened_dispatch_indices, expert_outputs
    )
    # Separate out num_selected_experts dimension.
    # Shape: [num_groups, tokens_per_group, num_selected_experts, hidden_dims].
    expert_outputs = expert_outputs.reshape(
        (num_groups, tokens_per_group, num_selected_experts, *hidden_dims)
    )

    # Shape: [num_groups, tokens_per_group, num_selected_experts, hidden_dims].
    # Weighted sum of the outputs from the different experts.
    combined_outputs = jnp.einsum(
        'gtk...,gtk->gt...',
        expert_outputs,
        router_indices.combine_weights,
        precision=self.precision,
    )

    # Gather and sow expert metrics.
    successfully_routed = successfully_routed.reshape(
        (num_groups, tokens_per_group, num_selected_experts)
    )
    # Number of tokens that were dispatched to at least one expert.
    num_tokens_dispatched_somewhere = jnp.max(
        successfully_routed, axis=-1
    ).sum()
    if self.router.ignore_padding_tokens:
      # Only count non-padding tokens.
      # To identify non-padding tokens, we rely on the fact that padding tokens
      # in the inputs have already been zeroed out in the default T5
      # architecture. See
      # https://github.com/google/flaxformer/blob/9712a16/flaxformer/architectures/t5/t5_architecture.py#L315
      # and
      # https://github.com/google/flaxformer/blob/9712a16/flaxformer/architectures/t5/t5_architecture.py#L603.
      num_tokens = jnp.array(
          (jnp.sum(jnp.abs(token_inputs), axis=-1) > 0),
          dtype=token_inputs.dtype,
      ).sum()
    else:
      num_tokens = float(num_groups * tokens_per_group)
    fraction_tokens_left_behind = (
        1.0 - num_tokens_dispatched_somewhere / num_tokens
    )

    # Total number of tokens that were dispatched (one token could be dispatched
    # to multiple experts).
    num_tokens_dispatched = successfully_routed.sum()
    total_expert_capacity = self.num_experts * expert_capacity * num_groups
    expert_usage = num_tokens_dispatched / total_expert_capacity
    # Of the tokens dispatched, how confident was the router in its routing.
    router_confidence = (
        router_indices.combine_weights.sum() / num_tokens_dispatched
    )

    self._sow_expert_metrics(  # pytype: disable=wrong-arg-types  # jax-types
        router_indices.auxiliary_loss,
        router_indices.router_z_loss,
        fraction_tokens_left_behind,
        router_confidence,
        expert_usage,
    )

    return combined_outputs

  def _mask_and_dispatch_to_experts(
      self,
      token_inputs: Array,
      enable_dropout: bool,
      expert_capacity: int,
      **kwargs,
  ) -> Array:
    """Wraps expert masked routing and dispatching algorithm.

    This algorithm takes the following steps:
    (1) Compute dispatch mask and combine array using self.router.
    (2) Dispatch inputs to experts based on dispatch mask.
    (3) Recombine individual expert outputs using combine array.

    Args:
      token_inputs: <float>[num_groups, tokens_per_group, hidden_dims] inputs to
        send to experts.
      enable_dropout: If true, apply jitter noise during routing and dropout
        during expert computation.
      expert_capacity: Each group will send this many tokens to each expert.
      **kwargs: Optional keyword arguments to pass to experts.

    Returns:
      <float>[num_groups, tokens_per_group, hidden_dims] outputs from experts.
    """
    num_groups, tokens_per_group = token_inputs.shape[:2]

    router_mask = self.router(
        token_inputs,
        self.num_experts,
        expert_capacity,
        apply_jitter=enable_dropout,
    )

    # Shape: [num_groups, num_experts, expert_capacity, hidden_dims].
    expert_inputs = jnp.einsum(
        'gt...,gtec->gec...',
        token_inputs,
        router_mask.dispatch_mask,
        precision=self.precision,
    )

    expert_outputs = self._call_experts(expert_inputs, enable_dropout, **kwargs)

    # Shape: [num_groups, tokens_per_group, hidden_dims]
    combined_outputs = jnp.einsum(
        'gec...,gtec->gt...',
        expert_outputs,
        router_mask.combine_array,
        precision=self.precision,
    )

    # Gather and sow expert metrics.
    # Number of tokens that were dispatched to at least one expert.
    num_tokens_dispatched_somewhere = jnp.max(
        router_mask.dispatch_mask, axis=(-1, -2)
    ).sum()
    if self.router.ignore_padding_tokens:
      # Only count non-padding tokens.
      # To identify non-padding tokens, we rely on the fact that padding tokens
      # in the inputs have already been zeroed out in the default T5
      # architecture. See
      # https://github.com/google/flaxformer/blob/9712a16/flaxformer/architectures/t5/t5_architecture.py#L315
      # and
      # https://github.com/google/flaxformer/blob/9712a16/flaxformer/architectures/t5/t5_architecture.py#L603.
      num_tokens = jnp.array(
          (jnp.sum(jnp.abs(token_inputs), axis=-1) > 0),
          dtype=token_inputs.dtype,
      ).sum()
    else:
      num_tokens = float(num_groups * tokens_per_group)
    fraction_tokens_left_behind = (
        1.0 - num_tokens_dispatched_somewhere / num_tokens
    )

    # Total number of tokens that were dispatched (one token could be
    # dispatched to multiple experts).
    num_tokens_dispatched = router_mask.dispatch_mask.sum()
    # Of the tokens dispatched, how confident was the router in its routing?
    router_confidence = router_mask.combine_array.sum() / num_tokens_dispatched

    if isinstance(self.router, routing.ExpertsChooseMaskedRouter):
      expert_usage = 1.0  # Experts fully utilized when "expert choose tokens"
    else:
      total_expert_capacity = self.num_experts * expert_capacity * num_groups
      expert_usage = num_tokens_dispatched / total_expert_capacity

    self._sow_expert_metrics(  # pytype: disable=wrong-arg-types  # jnp-type
        router_mask.auxiliary_loss,
        router_mask.router_z_loss,
        fraction_tokens_left_behind,
        router_confidence,
        expert_usage,
    )

    return combined_outputs

  def _call_experts(
      self, inputs: Array, enable_dropout: bool, **kwargs
  ) -> Array:
    """Sends and receives inputs to experts using pjit induced all-to-all calls.

    Assumes training is distributed using jax.experimental.pjit and induces
    all-to-all calls using reshapes and sharding constraints. We use Flax's
    lifted vmap to apply the expert transformation.

    Input data is ideally partitioned as:
    G_ed ** H_m,
    where G (num groups) is partitioned along the e (expert) and d (data)
    axes, and H (hidden dims) is partitioned along the m (model) axis. "**"
    denotes fully replicated axes. By partitioning H along the model parallel
    axis, we avoid duplicate information transfer in the all-to-alls between
    devices replicating data.

    The entire computation is performed using self.dtype. We recommend a
    truncated float type (e.g. bfloat16) to reduce all-to-all communication
    overhead.

    Args:
      inputs: <float>[num_groups, num_experts, expert_capacity, hidden_dims]
        inputs to be dispatched to experts. Each slice across the first
        dimension is passed to a different expert.
      enable_dropout: Whether or not experts should apply dropout.
      **kwargs: Optional keyword arguments to pass to experts.

    Returns:
      <float>[num_groups, num_experts, expert_capacity, hidden_dims] outputs
      from expert computation.
    """
    num_groups, num_experts, capacity, *hidden_dims = inputs.shape
    inputs_dtype = inputs.dtype
    inputs = jax.lax.convert_element_type(inputs, self.dtype)

    # Send examples to their target devices.

    # Note that the ordering of the logical mesh axes in these sharding
    # constraints should map consistently to the same underlying mesh axes; i.e.
    # 'batch' --> ('expert', 'data') and
    # ('expert', 'expert_replica') --> ('expert', 'data').
    inputs = flax_partitioning.with_sharding_constraint(
        inputs, ('batch', 'unmodeled', 'length', *self.input_hidden_dims_axes)
    )

    if self.num_expert_partitions != num_experts:
      # Explicitly extract dimension of size self.num_expert_partitions, along
      # which to partition experts.
      inputs = inputs.reshape(
          self.num_expert_partitions,
          num_groups // num_experts,
          num_experts // self.num_expert_partitions,
          num_experts,
          capacity,
          *hidden_dims,
      )
      inputs = jnp.swapaxes(inputs, 1, 2)

    # Induce all-to-alls:
    # E_ed ** H_m --> E_ed ** H_m,
    # where E is the number of experts and H is the hidden dimension. e, d, and
    # m denote the expert, data and model axes, respectively.
    inputs = inputs.reshape(
        num_experts,
        num_groups // num_experts,
        num_experts,
        capacity,
        *hidden_dims,
    )
    inputs = flax_partitioning.with_sharding_constraint(
        inputs,
        (
            'expert',
            'expert_replicas',
            'unmodeled',
            'length',
            *self.input_hidden_dims_axes,
        ),
    )
    inputs = jnp.swapaxes(inputs, 0, 2)
    inputs = flax_partitioning.with_sharding_constraint(
        inputs,
        (
            'expert',
            'expert_replicas',
            'unmodeled',
            'length',
            *self.input_hidden_dims_axes,
        ),
    )

    inputs = inputs.reshape(num_experts, num_groups * capacity, *hidden_dims)
    # Perform all-gather here along hidden dimnension (H) axis:
    # E_ed ** H_m --> E_ed ** H.
    inputs = flax_partitioning.with_sharding_constraint(
        inputs, ('expert', 'expert_replicas', 'unmodeled')
    )

    # Apply expert transformation.

    # Vectorize over the 'expert' axis of `inputs`. We use Flax's Lifted vmap
    # to introduce parameters along the mapped `expert` axis.
    # The vmapped MLP operation essentially performs:
    # E_ed ** H --> E_ed ** F_m --> E_ed ** H,
    # where F is the feed-forward dimension.
    @functools.partial(
        flax_partitioning.vmap_with_axes,
        in_axes=(0,),
        variable_axes={'params': 0},  # Each expert has its own parameters
        # Any mapped sharding constraints should be applied along 'expert' axis.
        spmd_axis_name='expert',
        split_rngs={
            # Whether to initialize each expert's params independently.
            'params': self.split_params,
            'dropout': True,  # Always use different dropout key for each expert
        },
        partitioning_axis_names={'params': 'expert'},
    )
    def layer_fn(mapped_expert: nn.Module, expert_inputs: Array) -> Array:
      return self._filter_inputs(
          mapped_expert, expert_inputs, enable_dropout=enable_dropout, **kwargs
      )

    outputs = layer_fn(self.expert, inputs)

    # Send examples back to their original devices.
    output_dims = outputs.shape[2:]
    outputs = outputs.reshape(num_experts, num_groups, capacity, *output_dims)

    # Reshard over along hidden dimension (H) axis:
    # E_ed ** H --> E_ed ** H_m,
    # before performing all-to-alls.
    outputs = flax_partitioning.with_sharding_constraint(
        outputs,
        ('expert', 'expert_replicas', 'length', *self.output_hidden_dims_axes),
    )
    outputs = outputs.reshape(
        num_experts,
        num_groups // num_experts,
        num_experts,
        capacity,
        *output_dims,
    )
    outputs = flax_partitioning.with_sharding_constraint(
        outputs,
        (
            'expert',
            'expert_replicas',
            'unmodeled',
            'length',
            *self.output_hidden_dims_axes,
        ),
    )
    outputs = jnp.swapaxes(outputs, 0, 2)
    outputs = flax_partitioning.with_sharding_constraint(
        outputs,
        (
            'expert',
            'expert_replicas',
            'unmodeled',
            'length',
            *self.output_hidden_dims_axes,
        ),
    )

    if self.num_expert_partitions != num_experts:
      # Explicitly extract dimension of size self.num_expert_partitions, along
      # which to partition experts.
      outputs = outputs.reshape(
          self.num_expert_partitions,
          num_experts // self.num_expert_partitions,
          num_groups // num_experts,
          num_experts,
          capacity,
          *output_dims,
      )
      outputs = jnp.swapaxes(outputs, 1, 2)

    outputs = outputs.reshape(num_groups, num_experts, capacity, *output_dims)
    outputs = flax_partitioning.with_sharding_constraint(
        outputs, ('batch', 'unmodeled', 'length', *self.output_hidden_dims_axes)
    )

    return jax.lax.convert_element_type(outputs, inputs_dtype)

  def _filter_inputs(
      self,
      mapped_expert: nn.Module,
      expert_inputs: Array,
      enable_dropout: bool = True,
      **kwargs,
  ) -> Array:
    """Forwards relevant inputs to `mapped_expert`.

    We support MLP (dense.MlpBlock) and regular dense layers
    (dense.DenseGeneral).

    Args:
      mapped_expert: Expert function that is vmapped.
      expert_inputs: Prepared inputs that are mapped over. Shape:
        <float>[num_experts, num_groups // num_experts, num_experts, capacity,
        hidden_dims]
      enable_dropout: Enables dropout if set to True. Only use for MLP experts.
      **kwargs: Optional keyword arguments to pass to experts. Only passed to
        MLP experts.

    Returns:
      Outputs from expert computation.

    Raises:
      ValueError for unsupported expert classes.
    """
    # TODO: Cleaner way of handling different expert call APIs?
    if isinstance(self.expert, dense.DenseGeneral):
      return mapped_expert(expert_inputs)
    elif isinstance(self.expert, dense.MlpBlock):
      return mapped_expert(
          expert_inputs, enable_dropout=enable_dropout, **kwargs
      )
    else:
      raise ValueError(f'Unsupported expert class: {self.expert}.')

  def _sow_expert_metrics(
      self,
      auxiliary_loss: float,
      router_z_loss: float,
      fraction_tokens_left_behind: float,
      router_confidence: float,
      expert_usage: float,
  ) -> None:
    """Sows metrics to analyze expert routing.

    Args:
      auxiliary_loss: Load balancing loss.
      router_z_loss: Loss to encourage smaller router logits.
      fraction_tokens_left_behind: Fraction of tokens NOT routed to any expert.
      router_confidence: Normalized sum of combine weights of those tokens which
        were routed to experts.
      expert_usage: Fraction of total expert capacity used to process tokens.
    NOTE: We wrap scalar metric values in into a 2D array to play nicely with
      the Flaxformer T5 architecture's scan invocation; see
    https://github.com/google/flaxformer/blob/9712a16/flaxformer/architectures/t5/t5_architecture.py#L742
      and
    https://github.com/google/flaxformer/blob/9712a16/flaxformer/architectures/t5/t5_architecture.py#L973.
    """
    for metric, value in [
        ('auxiliary_loss', auxiliary_loss),
        ('router_z_loss', router_z_loss),
        ('fraction_tokens_left_behind', fraction_tokens_left_behind),
        ('router_confidence', router_confidence),
        ('expert_usage', expert_usage),
    ]:
      wrapped_metric_value = jnp.asarray(value).reshape((1, 1))
      self.sow('intermediates', metric, wrapped_metric_value)


def _num_groups(
    num_tokens: int,
    max_group_size: int,
    num_experts: int,
    num_expert_replicas: int,
    strict_group_size: bool = False,
) -> int:
  """Returns the number of token routing groups.

  Note: For pjit-based training, all quantities are global.

  We select the smallest num_groups such that:
  - num_groups >= num_tokens / max_group_size (ensuring the group size is no
    larger than max_group_size),
  - num_tokens % num_groups = 0 (ensuring that the group size evenly divides
    into the num_tokens),
  - num_groups % (num_expert_replicas * num_experts) = 0 (ensuring that number
    of groups can be split across the total number of experts).

  Args:
    num_tokens: Number of tokens from input batch.
    max_group_size: Maximum size of each token routing group. Actual group size
      may end up being smaller.
    num_experts: Total number of unique experts.
    num_expert_replicas: Number of copies of each expert.
    strict_group_size: If True, fail if unable to set the token group size equal
      to max_group_size.

  Returns:
    Number of token routing groups.

  Raises:
    ValueError if we cannot find a group_size satisfying the above requirements.
  """
  # For pjit-based partitioning, we manipulated arrays globally. The number of
  # experts must evenly divide the number of (global) groups.
  min_num_groups = num_tokens // max_group_size
  min_num_groups = max(min_num_groups, num_expert_replicas * num_experts)

  def viable(n):
    """Returns true iff n is a viable number of groups."""
    return num_tokens % n == 0 and n % (num_expert_replicas * num_experts) == 0

  # Increase the number of groups (and decrease the group size) until we have
  # a viable number of groups.
  num_groups = min_num_groups
  while num_groups < num_tokens and not viable(num_groups):
    num_groups += 1

  if num_tokens % num_groups > 0:
    raise ValueError(
        'Group size and the number of experts must divide evenly into the '
        f'global number of tokens, but num_tokens={num_tokens}, while '
        f'num_groups={num_groups} for max_group_size={max_group_size} '
        f'and num_experts={num_experts}, each with {num_expert_replicas} '
        'replicas. Consider increasing the number of tokens (by increasing the '
        'batch size, sequence length, or beam size), and/or decreasing the '
        'number of expert copies (by increasing the expert parallelism or '
        'decreasing the number of experts).'
    )

  group_size = num_tokens // num_groups
  logging.info(
      (
          'Selected group_size=%d and num_groups=%d for input num_tokens=%d, '
          'max_group_size=%d, num_experts=%d and num_expert_replicas=%d'
      ),
      group_size,
      num_groups,
      num_tokens,
      max_group_size,
      num_experts,
      num_expert_replicas,
  )

  if strict_group_size and group_size != max_group_size:
    raise ValueError(
        f'Selected group_size={group_size} is less than the '
        f'max_group_size={max_group_size}. Exiting because strict mode is '
        'active (strict_group_size=True)'
    )

  return num_groups


def _num_expert_replicas(
    num_expert_partitions: int, num_model_partitions: int
) -> int:
  """Infer the number of expert replicas.

  This computation assumes that the underlying mesh has shape ['data', 'expert',
  'model']. We assume that experts are replicated along the 'data' axis, whose
  dimension is inversely proportional to the size of the expert and model
  parallel submeshes.

  Args:
    num_expert_partitions: Size of expert parallel submesh.
    num_model_partitions: Size of model parallel submesh.

  Returns:
    Number of replicas per expert.
  """
  return max(
      1, jax.device_count() // (num_expert_partitions * num_model_partitions)
  )


def _maybe_pad(
    inputs: Array, num_experts: int, num_expert_replicas: int
) -> Array:
  """Pads input array if number of tokens < number of experts.

  This function pads the input sequence to ensure that the number of tokens is
  divisble by the total number of experts.

  Args:
    inputs: Batch of input embeddings of shape <float>[batch_size, seq_length,
      hidden_dims].
    num_experts: Number of unique experts.
    num_expert_replicas: Number of copies of each expert.

  Returns:
    Input embeddings of shape <float>[batch_size + min_batch_padding, seq_length
    + min_seq_padding, hidden_dims]. Only one of min_batch_padding or
    min_seq_padding can be nonzero; both will be zero if no padding is required.
  """
  batch_size, seq_length, *_ = inputs.shape
  num_tokens = batch_size * seq_length
  total_num_experts = num_expert_replicas * num_experts

  if num_tokens % total_num_experts != 0:
    # Let's see how much padding is required if we pad the batch dimension.
    min_batch_padding = 1
    num_padding_tokens = seq_length
    while (num_tokens + num_padding_tokens) % total_num_experts != 0:
      # This loop will always yield
      # num_padding_tokens <= abs(total_num_experts * seq_length - num_tokens)
      # or, equivalently,
      # min_batch_padding <= abs(total_num_experts - batch_size).
      min_batch_padding += 1
      num_padding_tokens += seq_length

    # Alternatively, we could pad along the sequence dimension.
    min_seq_padding = 1
    num_padding_tokens = batch_size
    while (num_tokens + num_padding_tokens) % total_num_experts != 0:
      # This loop will always yield
      # num_padding_tokens <= abs(total_num_experts * batch_size - num_tokens)
      # or, equivalently,
      # min_seq_padding <= abs(total_num_experts - seq_length).
      min_seq_padding += 1
      num_padding_tokens += batch_size

    # Use the dimension which requires the least padding.
    if min_seq_padding * batch_size > min_batch_padding * seq_length:
      min_seq_padding = 0
    else:
      min_batch_padding = 0

    # TODO: Rather than relying on one of the dimensions, we
    #  should select the minimal amount of padding along a mixture of both of
    #  the sequence and batch dimensions.

    result = jnp.pad(
        inputs,
        ((0, min_batch_padding), (0, min_seq_padding), (0, 0)),
        'constant',
        constant_values=0,
    )

    logging.warning(
        (
            'Efficiency warning: Batch size / sequence length temporarily'
            ' padded by %d tokens in MoE layer to ensure that the total number'
            ' of tokens is divisible by the total number of experts (%d). For'
            ' improved efficiency, consider increasing the number of tokens (by'
            ' increasing the batch size or beam size), and/or decreasing the'
            ' number of expert copies (by increasing the expert parallelism or'
            ' decreasing the number of experts).'
        ),
        min_batch_padding * seq_length + min_seq_padding * batch_size,
        total_num_experts,
    )

    return result
  else:
    return inputs
