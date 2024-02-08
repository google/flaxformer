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

"""Tests for attention."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp

import numpy as np
from flaxformer.architectures.perceiver_ar import attention


class AttentionTest(parameterized.TestCase):

  def test_make_causal_mask_with_padding(self):
    x = jnp.array([[7, 0, 0], [8, 5, 0]])
    sequence_lengths = jnp.array([3, 3])
    y = attention.make_causal_mask(
        x, num_latents=3, sequence_lengths=sequence_lengths)
    self.assertEqual(y.shape, (2, 1, 3, 3))
    # Padding is not treated in a special way. So they need to be zeroed out
    # separately.
    expected_y = jnp.array([[[1., 0., 0.], [1., 1., 0.], [1., 1., 1.]]],
                           jnp.float32)
    np.testing.assert_allclose(y[0], expected_y)
    np.testing.assert_allclose(y[1], expected_y)

  def test_make_causal_mask(self):
    x = jnp.ones((1, 3))
    sequence_lengths = jnp.array([3])
    y = attention.make_causal_mask(
        x, num_latents=3, sequence_lengths=sequence_lengths)
    self.assertEqual(y.shape, (1, 1, 3, 3))
    expected_y = jnp.array([[[[1., 0., 0.], [1., 1., 0.], [1., 1., 1.]]]],
                           jnp.float32)
    np.testing.assert_allclose(y, expected_y)

  def test_make_causal_mask_fewer_latents_vary_sequence_lengths(self):
    x = jnp.ones((4, 3))
    sequence_lengths = jnp.array([0, 1, 2, 3])
    y = attention.make_causal_mask(
        x, num_latents=2, sequence_lengths=sequence_lengths)
    self.assertEqual(y.shape, (4, 1, 2, 3))
    expected_y = jnp.array([
        [[[1., 0., 0.], [1., 1., 0.]]],
        [[[1., 0., 0.], [1., 1., 0.]]],
        [[[1., 0., 0.], [1., 1., 0.]]],
        [[[1., 1., 0.], [1., 1., 1.]]],
    ], jnp.float32)
    np.testing.assert_allclose(y, expected_y)

  def test_make_decoder_mask_lm(self):
    decoder_target_tokens = jnp.array([[6, 7, 3, 0]])
    mask = attention.make_decoder_mask(
        decoder_target_tokens=decoder_target_tokens,
        num_latents=4,
        sequence_lengths=jnp.array([3]),
        dtype=jnp.float32)
    expected_mask = jnp.array([[[[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0],
                                 [0, 0, 0, 0]]]])
    np.testing.assert_array_equal(mask, expected_mask)

  def test_make_decoder_mask_lm_smaller_latents(self):
    decoder_target_tokens = jnp.array([[6, 7, 3, 0]])
    mask = attention.make_decoder_mask(
        decoder_target_tokens=decoder_target_tokens,
        num_latents=2,
        sequence_lengths=jnp.array([3]),
        dtype=jnp.float32)
    expected_mask = jnp.array([[[[1, 1, 0, 0], [1, 1, 1, 0]]]])
    np.testing.assert_array_equal(mask, expected_mask)

  def test_make_decoder_mask_prefix_lm(self):
    decoder_target_tokens = jnp.array([[5, 6, 7, 3, 4, 0]])
    decoder_causal_attention = jnp.array([[1, 1, 1, 0, 0, 0]])
    mask = attention.make_decoder_mask(
        decoder_target_tokens=decoder_target_tokens,
        num_latents=6,
        sequence_lengths=jnp.array([5]),
        dtype=jnp.float32,
        decoder_causal_attention=decoder_causal_attention)
    expected_mask = jnp.array(
        [[[[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0],
           [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0]]]],
        dtype=jnp.float32)
    np.testing.assert_array_equal(mask, expected_mask)

  def test_make_decoder_mask_prefix_lm_smaller_latents(self):
    decoder_target_tokens = jnp.array([[5, 6, 7, 3, 4, 0]])
    decoder_causal_attention = jnp.array([[1, 1, 1, 0, 0, 0]])
    mask = attention.make_decoder_mask(
        decoder_target_tokens=decoder_target_tokens,
        num_latents=2,
        sequence_lengths=jnp.array([5]),
        dtype=jnp.float32,
        decoder_causal_attention=decoder_causal_attention)
    expected_mask = jnp.array([[[[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]]]],
                              dtype=jnp.float32)
    np.testing.assert_array_equal(mask, expected_mask)

  def test_make_decoder_mask_prefix_lm_multiple_elements(self):
    decoder_target_tokens = jnp.array([[6, 7, 3, 0], [4, 5, 0, 0]])
    decoder_causal_attention = jnp.array([[1, 1, 0, 0], [1, 0, 0, 0]])
    mask = attention.make_decoder_mask(
        decoder_target_tokens=decoder_target_tokens,
        num_latents=4,
        sequence_lengths=jnp.array([3, 2]),
        dtype=jnp.float32,
        decoder_causal_attention=decoder_causal_attention)
    expected_mask0 = jnp.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0],
                                [0, 0, 0, 0]])
    expected_mask1 = jnp.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0],
                                [0, 0, 0, 0]])
    self.assertEqual(mask.shape, (2, 1, 4, 4))
    np.testing.assert_array_equal(mask[0, 0], expected_mask0)
    np.testing.assert_array_equal(mask[1, 0], expected_mask1)

  def test_make_decoder_mask_prefix_lm_multiple_elements_smaller_latents(self):
    decoder_target_tokens = jnp.array([[6, 7, 3, 0], [4, 5, 0, 0]])
    decoder_causal_attention = jnp.array([[1, 1, 0, 0], [1, 0, 0, 0]])
    mask = attention.make_decoder_mask(
        decoder_target_tokens=decoder_target_tokens,
        num_latents=2,
        sequence_lengths=jnp.array([3, 2]),
        dtype=jnp.float32,
        decoder_causal_attention=decoder_causal_attention)
    expected_mask0 = jnp.array([[1, 1, 0, 0], [1, 1, 1, 0]])
    expected_mask1 = jnp.array([[1, 0, 0, 0], [1, 1, 0, 0]])
    self.assertEqual(mask.shape, (2, 1, 2, 4))
    np.testing.assert_array_equal(mask[0, 0], expected_mask0)
    np.testing.assert_array_equal(mask[1, 0], expected_mask1)


if __name__ == '__main__':
  absltest.main()
