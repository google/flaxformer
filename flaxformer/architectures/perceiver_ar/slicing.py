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

"""Perceiver AR slicing utilities."""

import functools

import jax
from jax import lax
from jax.experimental import shard_map
from jax.interpreters import pxla
import jax.numpy as jnp

from flaxformer.types import Array

shard_map = shard_map.shard_map
P = jax.sharding.PartitionSpec


def get_sequence_lengths(decoder_target_tokens: Array) -> Array:
  """Return non-padding lengths of sequences in the batch."""
  return (decoder_target_tokens > 0).astype(jnp.int32).sum(axis=-1)


def sequence_slice_start(sequence_length: Array, num_latents: int) -> Array:
  """Calculate start index for slicing a sequence."""
  end = jnp.maximum(num_latents, sequence_length)
  start = end - num_latents
  return start


def _slice_sequences(x: Array, sequence_length: Array, num_latents: int,
                     axis: int) -> Array:
  start = sequence_slice_start(
      sequence_length=sequence_length, num_latents=num_latents)

  return lax.dynamic_slice_in_dim(x, start, num_latents, axis=axis)


def slice_sequences_vmap(x: Array, sequence_lengths: Array, num_latents: int,
                         axis_within_vmap: int) -> Array:
  """Slice sequences using vmap for Perceiver AR usage.

  Given the length of sequences and the number of latents, each sequence within
  the batch will be sliced to start at max(num_latents, length) - num_latents
  with a length of num_latents.

  Args:
    x: Array to slice, expected to be of shape [batch, ...].
    sequence_lengths: Length of the supplied sequences with shape [batch].
    num_latents: Number of Perceiver AR latents.
    axis_within_vmap: Axis to slice, from within the vmap where the batch axis
      will be hidden.

  Returns:
    Sliced input array.
  """
  return jax.vmap(
      functools.partial(
          _slice_sequences, num_latents=num_latents,
          axis=axis_within_vmap))(x, sequence_lengths)


def slice_sequences_shard_map(
    x: Array, sequence_lengths: Array, num_latents: int, axis_within_map: int
) -> Array:
  """Slice sequences using shard_map for Perceiver AR usage.

  Given the length of sequences and the number of latents, each sequence within
  the batch will be sliced to start at max(num_latents, length) - num_latents
  with a length of num_latents.

  This method should be used for slicing sequences that are partitioned, such
  as the inputs to self-attention.
  shard_map is used to work around XLA partitioning issues with gathers.
  If regular vmap is used, a bunch of extra allgathers are added.
  TODO: Check if this is still relevant.

  Args:
    x: Array to slice, expected to be of shape [batch, length, embedding].
    sequence_lengths: Length of the supplied sequences with shape [batch].
    num_latents: Number of Perceiver AR latents.
    axis_within_map: Axis to slice, from within the shard_map where the batch
      and embedding axis will be hidden.

  Returns:
    Sliced input array.
  """
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty or jax.devices()[0].platform == 'cpu':
    return slice_sequences_vmap(
        x, sequence_lengths, num_latents, axis_within_map
    )
  return shard_map(
      functools.partial(
          _slice_sequences, num_latents=num_latents, axis=axis_within_map
      ),
      mesh,
      in_specs=(P('data', None, 'model'), P('data')),
      out_specs=P('data', None, 'model'),
  )(x, sequence_lengths)
