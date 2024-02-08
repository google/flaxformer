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

"""Tests for rotary_embedding."""
from absl.testing import absltest

import jax.numpy as jnp
import numpy as np

from flaxformer.architectures.perceiver_ar import rotary_embedding
from flaxformer.components import embedding


class RotaryTest(absltest.TestCase):

  def test_rotary_embedding(self):
    """Checks the shape of rotary encodings."""
    batch = 2
    qlen = 3
    qheads = 4
    d = 2 * 5
    klen = 6
    kheads = 7
    maxlen = 8

    q = np.ones((batch, qlen, qheads, d))
    k = np.ones((batch, klen, kheads, d))
    cos = np.ones((maxlen, d))
    sin = np.ones((maxlen, d))
    out_q, out_k = rotary_embedding.apply_rotary_embedding(q, k, cos, sin)
    self.assertEqual(out_q.shape, q.shape)
    self.assertEqual(out_k.shape, k.shape)

  def test_rotary_embedding_multiquery(self):
    """Checks the shape of rotary encodings."""
    batch = 2
    qlen = 3
    qheads = 4
    d = 2 * 5
    klen = 6
    maxlen = 8

    q = np.ones((batch, qlen, qheads, d))
    k = np.ones((batch, klen, d))
    cos = np.ones((maxlen, d))
    sin = np.ones((maxlen, d))
    out_q, out_k = rotary_embedding.apply_rotary_embedding(q, k, cos, sin)
    self.assertEqual(out_q.shape, q.shape)
    self.assertEqual(out_k.shape, k.shape)

  def test_rotary_embedding_decode(self):
    """Checks the shape of rotary encodings."""
    batch = 2
    qlen = 1
    qheads = 4
    d = 2 * 5
    klen = 6
    maxlen = 8

    q = np.ones((batch, qlen, qheads, d))
    k = np.ones((batch, klen, d))
    cos = np.ones((maxlen, d))
    sin = np.ones((maxlen, d))
    rotary_index = np.ones((batch,), dtype=np.int32)
    out_q, out_k = rotary_embedding.apply_rotary_embedding(
        q, k, cos, sin, decode=True, rotary_index=rotary_index)
    self.assertEqual(out_q.shape, q.shape)
    self.assertEqual(out_k.shape, k.shape)

  def test_rotary_embedding_q_offset(self):
    """Checks the shape of rotary encodings."""
    batch = 2
    qlen = 3
    qheads = 4
    d = 2 * 5
    klen = 6
    kheads = 7
    maxlen = 8

    sin, cos = embedding.generate_fixed_pos_embedding(
        d, maxlen, max_timescale=maxlen)

    # First, generate with queries as long as keys.
    q = np.ones((batch, klen, qheads, d))
    k = np.ones((batch, klen, kheads, d))

    out_full_q, out_full_k = rotary_embedding.apply_rotary_embedding(
        q, k, cos, sin)
    self.assertEqual(out_full_q.shape, q.shape)
    self.assertEqual(out_full_k.shape, k.shape)

    # Then with shorter queries and an offset.
    short_q = np.ones((batch, qlen, qheads, d))

    out_short_q, out_short_k = rotary_embedding.apply_rotary_embedding(
        short_q, k, cos, sin, q_position_offset=jnp.array([2, 3]))
    self.assertEqual(out_short_q.shape, short_q.shape)
    self.assertEqual(out_short_k.shape, k.shape)
    np.testing.assert_allclose(out_short_k, out_full_k)

    # The shorter queries with offsets should be equivalent to a slice of the
    # full query output.
    np.testing.assert_allclose(out_short_q[0], out_full_q[0, 2:5])
    np.testing.assert_allclose(out_short_q[1], out_full_q[1, 3:])

  def test_rotary_embedding_to_subset(self):
    """Checks the shape of rotary encodings."""
    batch = 2
    qheads = 4
    d = 2 * 6
    qklen = 6
    kheads = 7
    maxlen = 8

    # First, generate with queries as long as keys.
    q = np.ones((batch, qklen, qheads, d))
    k = np.ones((batch, qklen, kheads, d))

    out_halfrot_q, out_halfrot_k = rotary_embedding.apply_rotary_embedding_to_subset(
        q, k, max_timescale=maxlen, fraction_to_rotate=0.5)
    self.assertEqual(out_halfrot_q.shape, q.shape)
    self.assertEqual(out_halfrot_k.shape, k.shape)

    # First half of dims should be rotated and therefore not the same as input.
    # Second half should match input.

    with np.testing.assert_raises(AssertionError):
      np.testing.assert_allclose(out_halfrot_q[..., :d // 2], q[..., :d // 2])
    np.testing.assert_allclose(out_halfrot_q[..., d // 2:], q[..., d // 2:])

    with np.testing.assert_raises(AssertionError):
      np.testing.assert_allclose(out_halfrot_k[..., :d // 2], k[..., :d // 2])
    np.testing.assert_allclose(out_halfrot_k[..., d // 2:], k[..., d // 2:])

if __name__ == '__main__':
  absltest.main()
