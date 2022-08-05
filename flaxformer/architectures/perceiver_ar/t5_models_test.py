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

"""Tests for t5_models."""

from typing import Sequence
from unittest import mock

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp

from flaxformer.architectures.perceiver_ar import t5_models


def _mock_randint_minval(key: chex.PRNGKey,
                         shape: Sequence[int],
                         minval: chex.Array,
                         maxval: chex.Array,
                         dtype: chex.ArrayDType = jnp.int_):
  del key, maxval
  return jnp.full(shape, minval, dtype)


def _mock_randint_maxval(key: chex.PRNGKey,
                         shape: Sequence[int],
                         minval: chex.Array,
                         maxval: chex.Array,
                         dtype: chex.ArrayDType = jnp.int_):
  del key, minval
  return jnp.full(shape, maxval - 1, dtype)


class T5ModelsTest(absltest.TestCase):

  def test_no_cropping(self):
    batch = {
        'decoder_target_tokens': jnp.ones([8, 128], jnp.int32),
        'decoder_loss_weights': jnp.ones([8, 128], jnp.int32),
    }
    cropped_batch = t5_models.crop_train_batch(
        jax.random.PRNGKey(0),
        batch=batch.copy(),
        cropping_method=t5_models.CroppingMethod.NONE,
        num_latents=16)
    chex.assert_trees_all_close(batch, cropped_batch)

  def test_full_latents_cropping_min(self):
    batch = {
        'decoder_target_tokens': jnp.ones([8, 128], jnp.int32),
        'decoder_loss_weights': jnp.ones([8, 128], jnp.int32),
    }
    expected_batch = {
        'decoder_target_tokens':
            jnp.concatenate(
                [jnp.ones([8, 16], jnp.int32),
                 jnp.zeros([8, 112], jnp.int32)],
                axis=1),
        'decoder_loss_weights':
            jnp.concatenate(
                [jnp.ones([8, 16], jnp.int32),
                 jnp.zeros([8, 112], jnp.int32)],
                axis=1),
    }
    with mock.patch.object(jax.random, 'randint', new=_mock_randint_minval):
      cropped_batch = t5_models.crop_train_batch(
          jax.random.PRNGKey(0),
          batch={**batch},
          cropping_method=t5_models.CroppingMethod.FULL_LATENTS,
          num_latents=16)
    chex.assert_trees_all_close(expected_batch, cropped_batch)

  def test_full_latents_cropping_max(self):
    batch = {
        'decoder_target_tokens': jnp.ones([8, 128], jnp.int32),
        'decoder_loss_weights': jnp.ones([8, 128], jnp.int32),
    }
    expected_batch = {
        'decoder_target_tokens':
            jnp.ones([8, 128], jnp.int32),
        'decoder_loss_weights':
            jnp.concatenate(
                [jnp.zeros([8, 112], jnp.int32),
                 jnp.ones([8, 16], jnp.int32)],
                axis=1),
    }
    with mock.patch.object(jax.random, 'randint', new=_mock_randint_maxval):
      cropped_batch = t5_models.crop_train_batch(
          jax.random.PRNGKey(0),
          batch={**batch},
          cropping_method=t5_models.CroppingMethod.FULL_LATENTS,
          num_latents=16)
    chex.assert_trees_all_close(expected_batch, cropped_batch)

  def test_equal_position_likelihood_cropping_min(self):
    batch = {
        'decoder_target_tokens': jnp.ones([8, 128], jnp.int32),
        'decoder_loss_weights': jnp.ones([8, 128], jnp.int32),
    }
    expected_batch = {
        'decoder_target_tokens':
            jnp.concatenate(
                [jnp.ones([8, 1], jnp.int32),
                 jnp.zeros([8, 127], jnp.int32)],
                axis=1),
        'decoder_loss_weights':
            jnp.concatenate(
                [jnp.ones([8, 1], jnp.int32),
                 jnp.zeros([8, 127], jnp.int32)],
                axis=1),
    }
    with mock.patch.object(jax.random, 'randint', new=_mock_randint_minval):
      cropped_batch = t5_models.crop_train_batch(
          jax.random.PRNGKey(0),
          batch={**batch},
          cropping_method=t5_models.CroppingMethod.EQUAL_POSITION_LIKELIHOOD,
          num_latents=16)
    chex.assert_trees_all_close(expected_batch, cropped_batch)

  def test_equal_position_likelihood_cropping_max(self):
    batch = {
        'decoder_target_tokens': jnp.ones([8, 128], jnp.int32),
        'decoder_loss_weights': jnp.ones([8, 128], jnp.int32),
    }
    expected_batch = {
        'decoder_target_tokens':
            jnp.ones([8, 128], jnp.int32),
        'decoder_loss_weights':
            jnp.concatenate(
                [jnp.zeros([8, 127], jnp.int32),
                 jnp.ones([8, 1], jnp.int32)],
                axis=1),
    }
    with mock.patch.object(jax.random, 'randint', new=_mock_randint_maxval):
      cropped_batch = t5_models.crop_train_batch(
          jax.random.PRNGKey(0),
          batch={**batch},
          cropping_method=t5_models.CroppingMethod.EQUAL_POSITION_LIKELIHOOD,
          num_latents=16)
    chex.assert_trees_all_close(expected_batch, cropped_batch)


if __name__ == '__main__':
  absltest.main()
