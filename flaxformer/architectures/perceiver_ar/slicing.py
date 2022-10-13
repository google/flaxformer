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

"""Perceiver AR slicing utilities."""

import functools

import jax
from jax import lax
from jax.experimental import maps
import jax.numpy as jnp
from t5x import partitioning
from flaxformer.types import Array


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


def slice_sequences_xmap(x: Array, sequence_lengths: Array, num_latents: int,
                         axis_within_xmap: int) -> Array:
  """Slice sequences using xmap for Perceiver AR usage.

  Given the length of sequences and the number of latents, each sequence within
  the batch will be sliced to start at max(num_latents, length) - num_latents
  with a length of num_latents.

  This method should be used for slicing sequences that are partitioned, such
  as the inputs to self-attention.
  xmap is used to work around XLA partitioning issues with gathers.
  If regular vmap is used, a bunch of extra allgathers are added.

  Requires the following flags:
  --experimental_xmap_spmd_lowering=True
  --experimental_xmap_spmd_lowering_manual=True

  Args:
    x: Array to slice, expected to be of shape [batch, length, embedding].
    sequence_lengths: Length of the supplied sequences with shape [batch].
    num_latents: Number of Perceiver AR latents.
    axis_within_xmap: Axis to slice, from within the xmap where the batch and
      embedding axis will be hidden.

  Returns:
    Sliced input array.
  """
  if (jax.devices()[0].platform != 'cpu' and
      partitioning.global_mesh_defined()):
    xmap_axis_resources = {'batch': 'data', 'embed': 'model'}
    xmap_embed_axis = 'embed'
  else:
    xmap_axis_resources = {}
    xmap_embed_axis = None

  return maps.xmap(
      functools.partial(
          _slice_sequences, num_latents=num_latents, axis=axis_within_xmap),
      in_axes=(['batch', None, xmap_embed_axis], ['batch']),
      out_axes=['batch', None, xmap_embed_axis],
      axis_resources=xmap_axis_resources)(x, sequence_lengths)
