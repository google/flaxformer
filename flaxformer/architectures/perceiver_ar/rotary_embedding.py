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

"""A library with rotary embedding functions."""

import functools
from typing import Optional, Tuple

import jax
from jax import numpy as jnp

from flaxformer.types import Array


def rotate_half(x: Array) -> Array:
  """Helper that splits a tensor at last dim into half and rotate it."""
  x1, x2 = jnp.split(x, 2, axis=-1)
  x = jnp.concatenate([-x2, x1], axis=-1)
  return x


@functools.partial(jax.jit, static_argnums=(4,))
def apply_rotary_embedding(
    q: Array,
    k: Array,
    cos: Array,
    sin: Array,
    decode: bool = False,
    q_position_offset: Optional[Array] = None,
    rotary_index: Optional[Array] = None) -> Tuple[Array, Array]:
  """Helper function to apply Rotary Embeddings, supports Q position offset."""
  if len(k.shape) == 3:
    # for multi query attention
    k = jnp.expand_dims(k, 2)
    multiquery = True
  else:
    multiquery = False

  batch, qlen, qheads, d = q.shape
  kbatch, klen, kheads, kd = k.shape
  assert batch == kbatch, f'{batch} != {kbatch}'
  assert d == kd, f'{d} != {kd}'

  # cos: [len, d]
  # sin: [len, d]
  # rotary_index: [batch]
  # q_position_offset: [batch]

  if decode and qlen == 1 and rotary_index is not None:
    # we check qlen == 1 so that we don't do this when initializing cache.
    qcos = cos[rotary_index, :]
    qsin = sin[rotary_index, :]
    # qcos, qsin: [batch, d]
    qcos = jax.lax.broadcast_in_dim(qcos, (batch, qlen, qheads, d), (0, 3))
    qsin = jax.lax.broadcast_in_dim(qsin, (batch, qlen, qheads, d), (0, 3))
    # qcos, qsin: [batch, qlen, qheads, d]
  else:
    if q_position_offset is None:
      qcos, qsin = cos[:qlen, :], sin[:qlen, :]
    else:
      # If q_position_offset is specified, we'll slice per-example after
      # broadcasting to batch size.
      qcos, qsin = cos, sin

    # qcos, qsin: [qlen, d]
    qcos = jax.lax.broadcast_in_dim(qcos, (batch, qcos.shape[0], qheads, d),
                                    (1, 3))
    qsin = jax.lax.broadcast_in_dim(qsin, (batch, qsin.shape[0], qheads, d),
                                    (1, 3))
    # qcos, qsin: [batch, qlen, qheads, d]
    if q_position_offset is not None:
      qcos = jax.vmap(
          functools.partial(
              jax.lax.dynamic_slice_in_dim, slice_size=qlen,
              axis=0))(qcos, q_position_offset)
      qsin = jax.vmap(
          functools.partial(
              jax.lax.dynamic_slice_in_dim, slice_size=qlen,
              axis=0))(qsin, q_position_offset)

  kcos, ksin = cos[:klen, :], sin[:klen, :]
  # kcos, ksin: [klen, d]
  kcos = jax.lax.broadcast_in_dim(kcos, (batch, klen, kheads, d), (1, 3))
  ksin = jax.lax.broadcast_in_dim(ksin, (batch, klen, kheads, d), (1, 3))
  # kcos, ksin: [batch, klen, kheads, d]

  out_q = (q * qcos) + (rotate_half(q) * qsin)
  out_k = (k * kcos) + (rotate_half(k) * ksin)
  if multiquery:
    out_k = jnp.squeeze(out_k, 2)
  return out_q, out_k
