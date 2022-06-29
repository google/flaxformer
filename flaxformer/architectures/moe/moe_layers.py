# Copyright 2022 Google LLC.
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
from typing import Optional

from absl import logging
import flax
from flax import linen as nn
from flax.linen import partitioning as flax_partitioning
import jax
import jax.numpy as jnp

from flaxformer.architectures.moe import routing
from flaxformer.architectures.moe import scatter_utils
from flaxformer.components import dense
from flaxformer.types import Array
from flaxformer.types import DType


@flax.struct.dataclass
class DiversityMetrics:
  """Metrics for analyzing diversity among experts in mixture of experts models.

  Attributes:
    auxiliary_loss: Auxiliary load balancing loss.
    router_z_loss: Router z-loss. Encourages router logits to remain small in an
      effort to improve stability.
    fraction_tokens_left_behind: Fraction of tokens NOT processed by any expert.
    expert_usage: Fraction of total capacity, across all experts, used to
      process tokens. Larger expert capacities or non-uniform token routing will
      result in smaller expert usage values.
    router_confidence: How confident the router is about the tokens that it has
      routed.
  """
  auxiliary_loss: float
  router_z_loss: float

  fraction_tokens_left_behind: float
  expert_usage: float
  router_confidence: float

  def __add__(self, other):
    return DiversityMetrics(
        self.auxiliary_loss + other.auxiliary_loss,
        self.router_z_loss + other.router_z_loss,
        self.fraction_tokens_left_behind + other.fraction_tokens_left_behind,
        self.expert_usage + other.expert_usage,
        self.router_confidence + other.router_confidence,
    )


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
      role depending on the routing assignment algorithm:
      - For "tokens choose" routing, the capacity factor only affects the
        maximum number of tokens that an expert will process. It does not affect
        how many experts a given token is routed to; see the
        num_selected_experts attributes of "tokens choose" routers.
      - For "experts choose" routing, because experts always fill their buffer,
        increasing the capacity factor will increase the number of tokens that
        an expert will process AND will indirectly increase the number of
        experts that a given token is routed to.
    eval_capacity_factor: As above, but used during evaluation.
    expert: The actual expert, currently constrained to be an MlpBlock.
    router: Token dispatch router. The router determines which tokens are
      dispatched to which expert, and how the expert outputs are combined.
    num_model_partitions: Size of the model parallel submesh. Model parallelism
      is used if num_model_partitions > 1.
    min_expert_capacity: Minimum token processing capacity for each expert.
    dropout_rate: Dropout rate for each expert.
    dtype: The numeric type (default: bfloat16). We recommend a truncated float
      type (e.g. bfloat16) to reduce all-to-all communication overhead. This
      numeric type is used for all computations, except the router, which always
      uses float32 for stability.
    split_params: Whether or not to initialize each expert's parameters
      independently.
    precision: XLA precision for array computations.
    optimize_model_parallel_communications: EXPERIMENTAL flag. If using
      model-parallelism for experts (experts spanning multiple devices), this
      flag is used to ensure that we do not perform duplicate all-to-all
      communication for experts spread across multiple devices, by partitioning
      the model/hidden dimension along the model-parallel axis for the experts.
      This same partition is used in Mesh Tensorflow:
      https://github.com/tensorflow/mesh/blob/6b31c0fc/mesh_tensorflow/transformer/moe.py#L487-L496
      This flag is experimental because, depending on model size and hardware
      topology, the reduction in all-to-all communication costs may be
      outweighed by the increased costs from the new reshape and all-gather.
      Current recommendation, roughly following Mesh Tensorflow, is only to use
      this flag for large models.
    strict_group_size: If True, fail if unable to set the token group size equal
      to max_group_size. If False (default), the actual group size may be
      smaller than max_group_size for consistency with the number of experts
      and tokens.
  """
  num_experts: int
  max_group_size: int
  # TODO: Switch to a single `capacity_factor` once we are using
  #  Fiddle to build different train vs eval model variants.
  train_capacity_factor: float
  eval_capacity_factor: float
  expert: dense.MlpBlock
  router: routing.Router
  num_model_partitions: int
  min_expert_capacity: int = 4
  dropout_rate: float = 0.1
  dtype: DType = jnp.bfloat16
  split_params: bool = True
  precision: jax.lax.Precision = jax.lax.Precision.DEFAULT
  optimize_model_parallel_communications: bool = False
  strict_group_size: bool = False

  def setup(self):
    """Verifies that the MoeLayer is correctly configured."""
    if self.optimize_model_parallel_communications and self.num_model_partitions <= 1:
      raise ValueError(
          'optimize_model_parallel_communications=True with '
          f'num_model_partitions={self.num_model_partitions} has no effect; '
          f'please set optimize_model_parallel_communications=False.')

    self.num_expert_replicas = _num_expert_replicas(self.num_experts,
                                                    self.num_model_partitions)

  @nn.compact
  def __call__(self,
               inputs,
               decode: bool = False,
               prefill: bool = False,
               prefill_lengths: Optional[Array] = None,
               *,
               enable_dropout: bool = True) -> Array:
    """Applies MoeLayer module.

    If the 'intermediates' collection is marked as mutable, this method will sow
    diversity metrics.

    Args:
      inputs: Batch of input embeddings of shape <float>[batch_size, seq_length,
        hidden_dim].
      decode: Whether to prepare and use an autoregressive cache.
      prefill: Whether to run a partial sequence to prefill the cache.
      prefill_lengths: The length of each partial sequence we are filling in the
        cache.
      enable_dropout: Enables dropout if set to True.

    Returns:
      Transformed inputs with same shape as inputs:
      <float>[batch_size, seq_length, hidden_dim].

    Raises:
      ValueError if an unrecognized dispatch algorithm is given.
    """
    batch_size, seq_length, hidden_dim = inputs.shape
    num_tokens = batch_size * seq_length

    num_groups = _num_groups(num_tokens, self.max_group_size, self.num_experts,
                             self.num_expert_replicas, self.strict_group_size)
    tokens_per_group = num_tokens // num_groups

    if enable_dropout:  # Training
      capacity_factor = self.train_capacity_factor
    else:  # Eval
      capacity_factor = self.eval_capacity_factor
    # Each group will send expert_capacity tokens to each expert.
    expert_capacity = int(
        round(capacity_factor * tokens_per_group / self.num_experts))
    expert_capacity = max(expert_capacity, self.min_expert_capacity)

    # Reshape batch and sequence/token dimensions for expert routing.
    token_inputs = jnp.reshape(inputs,
                               (num_groups, tokens_per_group, hidden_dim))

    if isinstance(self.router, routing.ScatterRouter):
      outputs = self._scatter_to_experts(
          token_inputs,
          enable_dropout,
          expert_capacity,
          decode=decode,
          prefill=prefill,
          prefill_lengths=prefill_lengths)
    elif isinstance(self.router, routing.MaskedRouter):
      outputs = self._mask_and_dispatch_to_experts(
          token_inputs,
          enable_dropout,
          expert_capacity,
          decode=decode,
          prefill=prefill,
          prefill_lengths=prefill_lengths)
    else:
      raise ValueError(f'Unrecognized router type: {self.router}')

    # Return to original input shape.
    result = outputs.reshape((batch_size, seq_length, hidden_dim))
    return result

  def _scatter_to_experts(self, token_inputs: Array, enable_dropout: bool,
                          expert_capacity: int, **kwargs) -> Array:
    """Wraps expert scatter routing and dispatching algorithm.

    This algorithm takes the following steps:
    (1) Compute expert dispatch indices and combine weights using self.router.
    (2) Scatter inputs to experts based on dispatch indices.
    (3) Recombine individual expert outputs using combine weights.

    Args:
      token_inputs: <float>[num_groups, tokens_per_group, hidden_dim] inputs to
        send to experts.
      enable_dropout: If true, apply jitter noise during routing and dropout
        during expert computation.
      expert_capacity: Each group will send this many tokens to each expert.
      **kwargs: Optional keyword arguments to pass to experts.

    Returns:
      <float>[num_groups, tokens_per_group, hidden_dim] outputs from experts.
    """
    num_groups, tokens_per_group, hidden_dim = token_inputs.shape
    num_tokens = num_groups * tokens_per_group

    router_indices = self.router(
        token_inputs,
        self.num_experts,
        expert_capacity,
        apply_jitter=enable_dropout)
    num_selected_experts = self.router.num_selected_experts

    # We need num_selected_experts copies of inputs for dispatching. This is a
    # no-op if num_selected_experts = 1.
    token_inputs = jnp.repeat(token_inputs, num_selected_experts, axis=1)

    # Mask out inputs that should not be routed.
    # Shape: [num_groups, tokens_per_group, num_selected_experts].
    successfully_routed = jnp.logical_and(
        router_indices.dispatch_indices[..., 0] < self.num_experts,
        router_indices.dispatch_indices[..., 1] < expert_capacity)
    successfully_routed = successfully_routed.reshape((num_groups, -1))
    # Shape: [num_groups, tokens_per_group * num_selected_experts, hidden_dim].
    masked_inputs = jnp.einsum(
        '...th,...t->...th',
        token_inputs,
        successfully_routed,
        precision=self.precision)

    # Combine tokens_per_group and num_selected_experts axes.
    flattened_dispatch_indices = router_indices.dispatch_indices.reshape(
        num_groups, -1, 2)

    # Scatter masked inputs.
    shape = (self.num_experts, expert_capacity, hidden_dim)
    # Note: scatter_nd can be slow under pjit on TPUs, presumably due to
    # suboptimal SPMD compilations. On TPUs, the recommendation is to use
    # MaskedRouter(s) instead.
    # Shape: [num_groups, num_experts, expert_capacity, hidden_dim].
    expert_inputs = jax.vmap(
        lambda i, x: scatter_utils.scatter_nd(i, x, shape))(
            flattened_dispatch_indices, masked_inputs)

    expert_outputs = self._call_experts(expert_inputs, enable_dropout, **kwargs)

    # Gather outputs.
    # Shape: [num_groups, tokens_per_group * num_selected_experts, hidden_dim].
    expert_outputs = jax.vmap(lambda i, x: x[i[:, 0], i[:, 1]])(
        flattened_dispatch_indices, expert_outputs)
    # Separate out num_selected_experts dimension.
    # Shape: [num_groups, tokens_per_group, num_selected_experts, hidden_dim].
    expert_outputs = expert_outputs.reshape(
        (num_groups, tokens_per_group, num_selected_experts, hidden_dim))

    # Shape: [num_groups, tokens_per_group, num_selected_experts, hidden_dim].
    # Weighted sum of the outputs from the different experts.
    combined_outputs = jnp.einsum(
        '...tkh,...tk->...th',
        expert_outputs,
        router_indices.combine_weights,
        precision=self.precision)

    # Gather and sow expert metrics.
    successfully_routed = successfully_routed.reshape(
        (num_groups, tokens_per_group, num_selected_experts))
    # Number of tokens that were dispatched to at least one expert.
    num_tokens_dispatched_somewhere = jnp.max(
        successfully_routed, axis=-1).sum()
    fraction_tokens_left_behind = 1.0 - num_tokens_dispatched_somewhere / float(
        num_tokens)
    # Total number of tokens that were dispatched (one token could be dispatched
    # to multiple experts).
    num_tokens_dispatched = successfully_routed.sum()
    total_expert_capacity = self.num_experts * expert_capacity * num_groups
    expert_usage = num_tokens_dispatched / total_expert_capacity
    # Of the tokens dispatched, how confident was the router in its routing.
    router_confidence = (
        router_indices.combine_weights.sum() / num_tokens_dispatched)

    self._sow_expert_metrics(router_indices.auxiliary_loss,
                             router_indices.router_z_loss,
                             fraction_tokens_left_behind, router_confidence,
                             expert_usage)

    return combined_outputs

  def _mask_and_dispatch_to_experts(self, token_inputs: Array,
                                    enable_dropout: bool, expert_capacity: int,
                                    **kwargs) -> Array:
    """Wraps expert masked routing and dispatching algorithm.

    This algorithm takes the following steps:
    (1) Compute dispatch mask and combine array using self.router.
    (2) Dispatch inputs to experts based on dispatch mask.
    (3) Recombine individual expert outputs using combine array.

    Args:
      token_inputs: <float>[num_groups, tokens_per_group, hidden_dim] inputs to
        send to experts.
      enable_dropout: If true, apply jitter noise during routing and dropout
        during expert computation.
      expert_capacity: Each group will send this many tokens to each expert.
      **kwargs: Optional keyword arguments to pass to experts.

    Returns:
      <float>[num_groups, tokens_per_group, hidden_dim] outputs from experts.
    """
    num_groups, tokens_per_group, _ = token_inputs.shape
    num_tokens = num_groups * tokens_per_group

    router_mask = self.router(
        token_inputs,
        self.num_experts,
        expert_capacity,
        apply_jitter=enable_dropout)

    # Shape: [num_groups, num_experts, expert_capacity, hidden_dim].
    expert_inputs = jnp.einsum(
        '...th,...tec->...ech',
        token_inputs,
        router_mask.dispatch_mask,
        precision=self.precision)

    expert_outputs = self._call_experts(expert_inputs, enable_dropout, **kwargs)

    # Shape: [num_groups, tokens_per_group, hidden_dim]
    combined_outputs = jnp.einsum(
        '...ech,...tec->...th',
        expert_outputs,
        router_mask.combine_array,
        precision=self.precision)

    # Gather and sow expert metrics.
    # Number of tokens that were dispatched to at least one expert.
    num_tokens_dispatched_somewhere = jnp.max(
        router_mask.dispatch_mask, axis=(-1, -2)).sum()
    fraction_tokens_left_behind = 1.0 - num_tokens_dispatched_somewhere / float(
        num_tokens)
    # Total number of tokens that were dispatched (one token could be
    # dispatched to multiple experts).
    num_tokens_dispatched = router_mask.dispatch_mask.sum()
    # Of the tokens dispatched, how confident was the router in its routing?
    router_confidence = router_mask.combine_array.sum() / num_tokens_dispatched

    if isinstance(self.router, routing.ExpertsChooseMaskedRouter):
      expert_usage = 1.  # Experts fully utilized when "expert choose tokens"
    else:
      total_expert_capacity = self.num_experts * expert_capacity * num_groups
      expert_usage = num_tokens_dispatched / total_expert_capacity

    self._sow_expert_metrics(router_mask.auxiliary_loss,
                             router_mask.router_z_loss,
                             fraction_tokens_left_behind, router_confidence,
                             expert_usage)

    return combined_outputs

  def _call_experts(self, inputs: Array, enable_dropout: bool,
                    **kwargs) -> Array:
    """Sends and receives inputs to experts using pjit induced all_to_all calls.

    Assumes training is distributed using jax.experimental.pjit and induces
    all_to_all calls using reshapes and sharding constraints. We use Flax's
    lifted vmap to apply the expert transformation.

    The entire computations is performed using self.dtype. We recommend a
    truncated float type (e.g. bfloat16) to reduce all-to-all communication
    overhead.

    Args:
      inputs: <float>[num_groups, num_experts, expert_capacity, hidden_dim]
        inputs to be dispatched to experts. Each slice across the first
        dimension is passed to a different expert.
      enable_dropout: Whether or not experts should apply dropout.
      **kwargs: Optional keyword arguments to pass to experts.

    Returns:
      <float>[num_groups, num_experts, expert_capacity, hidden_dim] outputs from
      expert computation.
    """
    num_groups, num_experts, capacity, hidden_dim = inputs.shape
    inputs_dtype = inputs.dtype
    inputs = jax.lax.convert_element_type(inputs, self.dtype)

    # Send examples to their target devices.
    inputs = flax_partitioning.with_sharding_constraint(
        inputs, ('batch', 'unmodeled', 'length', 'embed'))
    inputs = inputs.reshape(num_experts, num_groups // num_experts, num_experts,
                            capacity, hidden_dim)
    inputs = flax_partitioning.with_sharding_constraint(
        inputs, ('expert', 'expert_replicas', 'unmodeled', 'length', 'embed'))

    if self.optimize_model_parallel_communications:
      # Partition inputs along model parallel submesh axis to reduce duplicate
      # all-to-all communications in model parallelism cases.
      # Shape: [num_experts, num_groups // num_experts, num_experts, capacity,
      # hidden_dim, self.num_model_partitions]
      inputs = self._swapaxes_with_sharding_constraint(inputs, 0, 2, capacity,
                                                       hidden_dim)
    else:
      # Shape: [num_experts, num_groups // num_experts, num_experts, capacity,
      # hidden_dim]
      inputs = jnp.swapaxes(inputs, 0, 2)

    inputs = inputs.reshape(num_experts, num_groups * capacity, hidden_dim)
    inputs = flax_partitioning.with_sharding_constraint(
        inputs, ('expert', 'expert_replicas', 'embed'))

    # Apply expert transformation.

    # Vectorize over the 'expert' axis of `inputs`. We use Flax's Lifted vmap
    # to introduce parameters along the mapped `expert` axis.
    @functools.partial(
        flax_partitioning.vmap_with_axes,
        in_axes=(0,),
        variable_axes={'params': 0},  # Each expert has its own parameters
        split_rngs={
            # Whether or not to initialize each expert's params independently.
            'params': self.split_params,
            'dropout': True  # Always use different dropout key for each expert
        },
        partitioning_axis_names={'params': 'expert'})
    def layer_fn(mapped_expert, expert_inputs):
      return mapped_expert(
          expert_inputs, enable_dropout=enable_dropout, **kwargs)

    outputs = layer_fn(self.expert, inputs)

    # Send examples back to their original devices.
    outputs = outputs.reshape(num_experts, num_groups, capacity, hidden_dim)
    outputs = flax_partitioning.with_sharding_constraint(
        outputs, ('expert', 'expert_replicas', 'length', 'embed'))
    outputs = outputs.reshape(num_experts, num_groups // num_experts,
                              num_experts, capacity, hidden_dim)

    if self.optimize_model_parallel_communications:
      # Shape: [num_experts, num_groups // num_experts, num_experts, capacity,
      # hidden_dim, self.num_model_partitions]
      outputs = self._swapaxes_with_sharding_constraint(outputs, 0, 2, capacity,
                                                        hidden_dim)
    else:
      # Shape: [num_experts, num_groups // num_experts, num_experts, capacity,
      # hidden_dim]
      outputs = jnp.swapaxes(outputs, 0, 2)

    outputs = outputs.reshape(num_groups, num_experts, capacity, hidden_dim)
    outputs = flax_partitioning.with_sharding_constraint(
        outputs, ('batch', 'unmodeled', 'length', 'embed'))

    return jax.lax.convert_element_type(outputs, inputs_dtype)

  def _sow_expert_metrics(self, auxiliary_loss: float, router_z_loss: float,
                          fraction_tokens_left_behind: float,
                          router_confidence: float, expert_usage: float):
    """Sows metrics to analyze expert routing."""
    self.sow(
        'intermediates',
        'diversity_metrics',
        DiversityMetrics(auxiliary_loss, router_z_loss,
                         fraction_tokens_left_behind, expert_usage,
                         router_confidence),
        init_fn=lambda: DiversityMetrics(0., 0., 0., 0., 0.),
        # DiversityMetrics are summed if repeated calls are made.
        reduce_fn=lambda a, b: a + b)

  def _swapaxes_with_sharding_constraint(self, array: Array, axis1: int,
                                         axis2: int, expert_capacity: int,
                                         hidden_dim: int) -> Array:
    """Interchanges two array axes under model-parallel sharding constraints.

    For model-parallel experts (self.num_model_partitions > 1), to ensure that
    multiple devices are not performing all-to-all duplicate communications, we
    partition the model/hidden dimension along the expert model-parallel axis
    ('expert_mlp') before performing the all-to-all transfer. See also:
    https://github.com/tensorflow/mesh/blob/6b31c0fc/mesh_tensorflow/transformer/moe.py#L487-L496


    Args:
      array: Input array.
      axis1: First axis.
      axis2: Second axis.
      expert_capacity: Token capacity per expert.
      hidden_dim: Model/hidden dimension of inputs.

    Returns:
      View or copy of input array with axes swapped.

    Raises:
      ValueError if num_model_partitions is less than or equal to 1.
    """
    if self.num_model_partitions <= 1:
      raise ValueError('Expected num_model_partitions to be > 1 but got: '
                       f'{self.num_model_partitions}')
    array = array.reshape(self.num_experts, -1, self.num_experts,
                          expert_capacity,
                          hidden_dim // self.num_model_partitions,
                          self.num_model_partitions)
    array = flax_partitioning.with_sharding_constraint(
        array, ('expert', 'expert_group', 'unmodeled', 'length', 'embed',
                'expert_mlp'))
    array = jnp.swapaxes(array, axis1, axis2)
    return flax_partitioning.with_sharding_constraint(
        array, ('expert', 'expert_group', 'unmodeled', 'length', 'embed',
                'expert_mlp'))


def _num_groups(num_tokens: int,
                max_group_size: int,
                num_experts: int,
                num_expert_replicas: int,
                strict_group_size: bool = False) -> int:
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
        'replicas')

  group_size = num_tokens // num_groups
  logging.info(
      'Selected group_size=%d and num_groups=%d for input num_tokens=%d, '
      'max_group_size=%d, num_experts=%d and num_expert_replicas=%d',
      group_size, num_groups, num_tokens, max_group_size, num_experts,
      num_expert_replicas)

  if strict_group_size and group_size != max_group_size:
    raise ValueError(
        f'Selected group_size={group_size} is less than the '
        f'max_group_size={max_group_size}. Exiting because strict mode is '
        'active (strict_group_size=True)')

  return num_groups


def _num_expert_replicas(num_experts: int, num_model_partitions: int) -> int:
  """Infer the number of expert replicas.

  This computation assumes that we are using the T5X MoePjitPartitioner. In
  particular, we assume that experts are replicated along the 'data' axis, whose
  dimension is inversely proportional to the number of experts and number of
  model parallel dimensions. See also
  https://github.com/google-research/t5x/blob/bdd3928/t5x/contrib/moe/partitioning.py.

  Args:
    num_experts: Total number of experts, across all devices.
    num_model_partitions: Size of model parallel submesh.

  Returns:
    Number of replicas per expert.
  """
  return max(1, jax.device_count() // (num_experts * num_model_partitions))
