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

"""Tests for t5_models."""

import functools
from typing import Sequence
from unittest import mock

from absl.testing import absltest
import chex
from flax.training import common_utils
import jax
import jax.numpy as jnp
from t5x import losses

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


class T5ModelsCroppingTest(absltest.TestCase):

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
      cropped_batch_full_latents = t5_models.crop_train_batch(
          jax.random.PRNGKey(0),
          batch={**batch},
          cropping_method=t5_models.CroppingMethod.FULL_LATENTS,
          num_latents=16)
      cropped_batch_full_latents_with_prefix = t5_models.crop_train_batch(
          jax.random.PRNGKey(0),
          batch={**batch},
          cropping_method=t5_models.CroppingMethod.FULL_LATENTS_WITH_PREFIX,
          num_latents=16)
    chex.assert_trees_all_close(expected_batch, cropped_batch_full_latents)
    chex.assert_trees_all_close(expected_batch,
                                cropped_batch_full_latents_with_prefix)

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
      cropped_batch_full_latents = t5_models.crop_train_batch(
          jax.random.PRNGKey(0),
          batch={**batch},
          cropping_method=t5_models.CroppingMethod.FULL_LATENTS,
          num_latents=16)
      cropped_batch_full_latents_with_prefix = t5_models.crop_train_batch(
          jax.random.PRNGKey(0),
          batch={**batch},
          cropping_method=t5_models.CroppingMethod.FULL_LATENTS_WITH_PREFIX,
          num_latents=16)
    chex.assert_trees_all_close(expected_batch, cropped_batch_full_latents)
    chex.assert_trees_all_close(expected_batch,
                                cropped_batch_full_latents_with_prefix)

  def test_prefix_seq_full_latents_cropping_min(self):
    batch = {
        'decoder_target_tokens':
            jnp.ones([8, 128], jnp.int32),
        'decoder_loss_weights':
            jnp.concatenate(
                [jnp.zeros([8, 28], jnp.int32),
                 jnp.ones([8, 100], jnp.int32)],
                axis=1),
    }
    expected_batch = {
        'decoder_target_tokens':
            jnp.concatenate(
                [jnp.ones([8, 44], jnp.int32),
                 jnp.zeros([8, 84], jnp.int32)],
                axis=1),
        'decoder_loss_weights':
            jnp.concatenate([
                jnp.zeros([8, 28], jnp.int32),
                jnp.ones([8, 16], jnp.int32),
                jnp.zeros([8, 84], jnp.int32)
            ],
                            axis=1),
    }
    with mock.patch.object(jax.random, 'randint', new=_mock_randint_minval):
      cropped_batch = t5_models.crop_train_batch(
          jax.random.PRNGKey(0),
          batch={**batch},
          cropping_method=t5_models.CroppingMethod.FULL_LATENTS,
          num_latents=16)
    chex.assert_trees_all_close(expected_batch, cropped_batch)

  def test_prefix_seq_full_latents_with_prefix_cropping_min(self):
    batch = {
        'decoder_target_tokens':
            jnp.ones([8, 128], jnp.int32),
        'decoder_loss_weights':
            jnp.concatenate(
                [jnp.zeros([8, 28], jnp.int32),
                 jnp.ones([8, 100], jnp.int32)],
                axis=1),
    }
    expected_batch = {
        'decoder_target_tokens':
            jnp.concatenate(
                [jnp.ones([8, 29], jnp.int32),
                 jnp.zeros([8, 99], jnp.int32)],
                axis=1),
        'decoder_loss_weights':
            jnp.concatenate([
                jnp.zeros([8, 28], jnp.int32),
                jnp.ones([8, 1], jnp.int32),
                jnp.zeros([8, 99], jnp.int32)
            ],
                            axis=1),
    }
    with mock.patch.object(jax.random, 'randint', new=_mock_randint_minval):
      cropped_batch = t5_models.crop_train_batch(
          jax.random.PRNGKey(0),
          batch={**batch},
          cropping_method=t5_models.CroppingMethod.FULL_LATENTS_WITH_PREFIX,
          num_latents=16)
    chex.assert_trees_all_close(expected_batch, cropped_batch)

  def test_prefix_seq_full_latents_cropping_max(self):
    batch = {
        'decoder_target_tokens':
            jnp.ones([8, 128], jnp.int32),
        'decoder_loss_weights':
            jnp.concatenate(
                [jnp.zeros([8, 28], jnp.int32),
                 jnp.ones([8, 100], jnp.int32)],
                axis=1),
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
      cropped_batch_full_latents = t5_models.crop_train_batch(
          jax.random.PRNGKey(0),
          batch={**batch},
          cropping_method=t5_models.CroppingMethod.FULL_LATENTS,
          num_latents=16)
      cropped_batch_full_latents_with_prefix = t5_models.crop_train_batch(
          jax.random.PRNGKey(0),
          batch={**batch},
          cropping_method=t5_models.CroppingMethod.FULL_LATENTS_WITH_PREFIX,
          num_latents=16)
    chex.assert_trees_all_close(expected_batch, cropped_batch_full_latents)
    chex.assert_trees_all_close(expected_batch,
                                cropped_batch_full_latents_with_prefix)

  def test_partial_seq_full_latents_cropping_min(self):
    batch = {
        'decoder_target_tokens':
            jnp.concatenate(
                [jnp.ones([8, 100], jnp.int32),
                 jnp.zeros([8, 28], jnp.int32)],
                axis=1),
        'decoder_loss_weights':
            jnp.concatenate(
                [jnp.ones([8, 100], jnp.int32),
                 jnp.zeros([8, 28], jnp.int32)],
                axis=1),
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
      cropped_batch_full_latents = t5_models.crop_train_batch(
          jax.random.PRNGKey(0),
          batch={**batch},
          cropping_method=t5_models.CroppingMethod.FULL_LATENTS,
          num_latents=16)
      cropped_batch_full_latents_with_prefix = t5_models.crop_train_batch(
          jax.random.PRNGKey(0),
          batch={**batch},
          cropping_method=t5_models.CroppingMethod.FULL_LATENTS_WITH_PREFIX,
          num_latents=16)
    chex.assert_trees_all_close(expected_batch, cropped_batch_full_latents)
    chex.assert_trees_all_close(expected_batch,
                                cropped_batch_full_latents_with_prefix)

  def test_prefix_full_latents_cropping_max(self):
    batch = {
        'decoder_target_tokens':
            jnp.concatenate(
                [jnp.ones([8, 100], jnp.int32),
                 jnp.zeros([8, 28], jnp.int32)],
                axis=1),
        'decoder_loss_weights':
            jnp.concatenate(
                [jnp.ones([8, 100], jnp.int32),
                 jnp.zeros([8, 28], jnp.int32)],
                axis=1),
    }
    expected_batch = {
        'decoder_target_tokens':
            jnp.concatenate(
                [jnp.ones([8, 100], jnp.int32),
                 jnp.zeros([8, 28], jnp.int32)],
                axis=1),
        'decoder_loss_weights':
            jnp.concatenate([
                jnp.zeros([8, 84], jnp.int32),
                jnp.ones([8, 16], jnp.int32),
                jnp.zeros([8, 28], jnp.int32)
            ],
                            axis=1),
    }
    with mock.patch.object(jax.random, 'randint', new=_mock_randint_maxval):
      cropped_batch_full_latents = t5_models.crop_train_batch(
          jax.random.PRNGKey(0),
          batch={**batch},
          cropping_method=t5_models.CroppingMethod.FULL_LATENTS,
          num_latents=16)
      cropped_batch_full_latents_with_prefix = t5_models.crop_train_batch(
          jax.random.PRNGKey(0),
          batch={**batch},
          cropping_method=t5_models.CroppingMethod.FULL_LATENTS_WITH_PREFIX,
          num_latents=16)
    chex.assert_trees_all_close(expected_batch, cropped_batch_full_latents)
    chex.assert_trees_all_close(expected_batch,
                                cropped_batch_full_latents_with_prefix)

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


def _mock_compute_logits(params,
                         batch,
                         dropout_rng,
                         num_latents,
                         vocab_size,
                         mutable=False):
  del params, dropout_rng, mutable
  batch_size = batch['decoder_input_tokens'].shape[0]
  logits = jnp.zeros((batch_size, num_latents, vocab_size), jnp.float32)
  # Assuming input tokens are all 1s except the first position, summing will
  # determine the input_length.
  input_length = batch['decoder_input_tokens'].sum(axis=1) + 1
  # Set vocab position 0 to be the input length * 10 for all sequence positions.
  logits = jax.vmap(lambda x, l: x.at[:, 0].set(l * 10))(logits, input_length)
  # If sequence length is shorter than latents, don't fill positions without
  # inputs.
  logits = jnp.where(
      jnp.arange(logits.shape[1])[jnp.newaxis, :, jnp.newaxis] < input_length,
      logits, 0)
  return logits


def _get_token_scores(logits, target_tokens, weights):
  return -losses.cross_entropy_with_logits(
      logits,
      common_utils.onehot(
          target_tokens, logits.shape[-1], on_value=1, off_value=0),
      z_loss=0.0)[0] * weights


class T5ModelsScoreBatchTest(absltest.TestCase):
  """Tests for score_batch.

  The goal of these tests is to ensure that the striding and logits combining
  is happening as intended. So we use _mock_compute_logits to return fake logits
  that are deterministic based on sequence position. By calculating an expected
  final score based on the logits created by an expected set of strides, we can
  ensure the process is completing as expected.
  """

  def test_score_batch(self):
    num_latents = 4
    vocab_size = 2
    model = t5_models.PerceiverARModel(
        module=None,
        vocabulary=None,
        optimizer_def=None,
        num_latents=num_latents,
        decoding_latent_reset_fill=3)

    batch = {
        'decoder_target_tokens': jnp.ones([2, 8]),
        'decoder_input_tokens': jnp.ones([2, 8]).at[:, 0].set(0),
        'decoder_loss_weights': jnp.ones([2, 8]),
        'decoder_causal_attention': jnp.zeros([2, 8]),
    }

    with mock.patch.object(
        model,
        '_compute_logits',
        new=functools.partial(
            _mock_compute_logits,
            num_latents=num_latents,
            vocab_size=vocab_size)):
      sequence_scores = model.score_batch(params=None, batch=batch)

    expected_logits = jnp.array([[[40., 0.], [40., 0.], [40., 0.], [40., 0.],
                                  [60., 0.], [60., 0.], [80., 0.], [80., 0.]]])
    expected_logits = jnp.tile(expected_logits, (2, 1, 1))

    expected_token_scores = _get_token_scores(expected_logits,
                                              batch['decoder_target_tokens'],
                                              batch['decoder_loss_weights'])
    expected_sequence_scores = expected_token_scores.sum(-1)

    chex.assert_trees_all_close(expected_sequence_scores, sequence_scores)

  def test_score_batch_with_remainder(self):
    num_latents = 4
    vocab_size = 2
    model = t5_models.PerceiverARModel(
        module=None,
        vocabulary=None,
        optimizer_def=None,
        num_latents=num_latents,
        decoding_latent_reset_fill=3)

    batch = {
        'decoder_target_tokens': jnp.ones([2, 9]),
        'decoder_input_tokens': jnp.ones([2, 9]).at[:, 0].set(0),
        'decoder_loss_weights': jnp.ones([2, 9]),
        'decoder_causal_attention': jnp.zeros([2, 9]),
    }

    with mock.patch.object(
        model,
        '_compute_logits',
        new=functools.partial(
            _mock_compute_logits,
            num_latents=num_latents,
            vocab_size=vocab_size)):
      sequence_scores = model.score_batch(params=None, batch=batch)

    expected_logits = jnp.array([[[40., 0.], [40., 0.], [40., 0.], [40., 0.],
                                  [60., 0.], [60., 0.], [80., 0.], [80., 0.],
                                  [90., 0.]]])
    expected_logits = jnp.tile(expected_logits, (2, 1, 1))

    expected_token_scores = _get_token_scores(expected_logits,
                                              batch['decoder_target_tokens'],
                                              batch['decoder_loss_weights'])
    expected_sequence_scores = expected_token_scores.sum(-1)

    chex.assert_trees_all_close(expected_sequence_scores, sequence_scores)

  def test_score_batch_inputs_match_latents(self):
    num_latents = 4
    vocab_size = 2
    model = t5_models.PerceiverARModel(
        module=None,
        vocabulary=None,
        optimizer_def=None,
        num_latents=num_latents,
        decoding_latent_reset_fill=3)

    batch = {
        'decoder_target_tokens': jnp.ones([2, 4]),
        'decoder_input_tokens': jnp.ones([2, 4]).at[:, 0].set(0),
        'decoder_loss_weights': jnp.ones([2, 4]),
        'decoder_causal_attention': jnp.zeros([2, 4]),
    }

    with mock.patch.object(
        model,
        '_compute_logits',
        new=functools.partial(
            _mock_compute_logits,
            num_latents=num_latents,
            vocab_size=vocab_size)):
      sequence_scores = model.score_batch(params=None, batch=batch)

    expected_logits = jnp.array([[[40., 0.], [40., 0.], [40., 0.], [40., 0.]]])
    expected_logits = jnp.tile(expected_logits, (2, 1, 1))

    expected_token_scores = _get_token_scores(expected_logits,
                                              batch['decoder_target_tokens'],
                                              batch['decoder_loss_weights'])
    expected_sequence_scores = expected_token_scores.sum(-1)

    chex.assert_trees_all_close(expected_sequence_scores, sequence_scores)

  def test_score_batch_short_sequence(self):
    num_latents = 4
    vocab_size = 2
    model = t5_models.PerceiverARModel(
        module=None,
        vocabulary=None,
        optimizer_def=None,
        num_latents=num_latents,
        decoding_latent_reset_fill=3)

    batch = {
        'decoder_target_tokens':
            jnp.ones([2, 8]).at[:, 5:].set(0),
        'decoder_input_tokens':
            jnp.ones([2, 8]).at[:, 0].set(0).at[:, 5:].set(0),
        'decoder_loss_weights':
            jnp.ones([2, 8]).at[:, 5:].set(0),
        'decoder_causal_attention':
            jnp.zeros([2, 8]),
    }

    with mock.patch.object(
        model,
        '_compute_logits',
        new=functools.partial(
            _mock_compute_logits,
            num_latents=num_latents,
            vocab_size=vocab_size)):
      sequence_scores = model.score_batch(params=None, batch=batch)

    expected_logits = jnp.array([[[40., 0.], [40., 0.], [40., 0.], [40., 0.],
                                  [50., 0.], [0., 0.], [0., 0.], [0., 0.]]])
    expected_logits = jnp.tile(expected_logits, (2, 1, 1))

    expected_token_scores = _get_token_scores(expected_logits,
                                              batch['decoder_target_tokens'],
                                              batch['decoder_loss_weights'])
    expected_sequence_scores = expected_token_scores.sum(-1)

    chex.assert_trees_all_close(expected_sequence_scores, sequence_scores)

  def test_score_batch_sequence_shorter_than_latents(self):
    num_latents = 8
    vocab_size = 2
    model = t5_models.PerceiverARModel(
        module=None,
        vocabulary=None,
        optimizer_def=None,
        num_latents=num_latents,
        decoding_latent_reset_fill=3)

    batch = {
        'decoder_target_tokens':
            jnp.ones([2, 8]).at[:, 5:].set(0),
        'decoder_input_tokens':
            jnp.ones([2, 8]).at[:, 0].set(0).at[:, 5:].set(0),
        'decoder_loss_weights':
            jnp.ones([2, 8]).at[:, 5:].set(0),
        'decoder_causal_attention':
            jnp.zeros([2, 8]),
    }

    with mock.patch.object(
        model,
        '_compute_logits',
        new=functools.partial(
            _mock_compute_logits,
            num_latents=num_latents,
            vocab_size=vocab_size)):
      sequence_scores = model.score_batch(params=None, batch=batch)

    expected_logits = jnp.array([[[50., 0.], [50., 0.], [50., 0.], [50., 0.],
                                  [50., 0.], [0., 0.], [0., 0.], [0., 0.]]])
    expected_logits = jnp.tile(expected_logits, (2, 1, 1))

    expected_token_scores = _get_token_scores(expected_logits,
                                              batch['decoder_target_tokens'],
                                              batch['decoder_loss_weights'])
    expected_sequence_scores = expected_token_scores.sum(-1)

    chex.assert_trees_all_close(expected_sequence_scores, sequence_scores)

  def test_score_batch_different_lengths(self):
    num_latents = 4
    vocab_size = 2
    model = t5_models.PerceiverARModel(
        module=None,
        vocabulary=None,
        optimizer_def=None,
        num_latents=num_latents,
        decoding_latent_reset_fill=3)

    batch = {
        'decoder_target_tokens':
            jnp.ones([2, 8]).at[0, 5:].set(0),
        'decoder_input_tokens':
            jnp.ones([2, 8]).at[:, 0].set(0).at[0, 5:].set(0),
        'decoder_loss_weights':
            jnp.ones([2, 8]).at[0, 5:].set(0),
        'decoder_causal_attention':
            jnp.zeros([2, 8]),
    }

    with mock.patch.object(
        model,
        '_compute_logits',
        new=functools.partial(
            _mock_compute_logits,
            num_latents=num_latents,
            vocab_size=vocab_size)):
      sequence_scores = model.score_batch(params=None, batch=batch)

    expected_logits = jnp.array([
        [[40., 0.], [40., 0.], [40., 0.], [40., 0.], [50., 0.], [0., 0.],
         [0., 0.], [0., 0.]],
        [[40., 0.], [40., 0.], [40., 0.], [40., 0.], [60., 0.], [60., 0.],
         [80., 0.], [80., 0.]],
    ])

    expected_token_scores = _get_token_scores(expected_logits,
                                              batch['decoder_target_tokens'],
                                              batch['decoder_loss_weights'])
    expected_sequence_scores = expected_token_scores.sum(-1)

    chex.assert_trees_all_close(expected_sequence_scores, sequence_scores)

  def test_score_batch_different_lengths_with_input_prefix(self):
    num_latents = 4
    vocab_size = 2
    model = t5_models.PerceiverARModel(
        module=None,
        vocabulary=None,
        optimizer_def=None,
        num_latents=num_latents,
        decoding_latent_reset_fill=3)

    batch = {
        'decoder_target_tokens':
            jnp.array([
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]),
        'decoder_input_tokens':
            jnp.array([
                [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]),
        'decoder_loss_weights':
            jnp.array([
                [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            ]),
        'decoder_causal_attention':
            jnp.array([
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            ]),
    }

    with mock.patch.object(
        model,
        '_compute_logits',
        new=functools.partial(
            _mock_compute_logits,
            num_latents=num_latents,
            vocab_size=vocab_size)):
      sequence_scores = model.score_batch(params=None, batch=batch)

    expected_logits = jnp.array([
        [[0., 0.], [0., 0.], [60., 0.], [60., 0.], [60., 0.], [60., 0.],
         [70., 0.], [0., 0.], [0., 0.], [0., 0.]],
        [[0., 0.], [0., 0.], [60., 0.], [60., 0.], [60., 0.], [60., 0.],
         [80., 0.], [80., 0.], [100., 0.], [100., 0.]],
    ])

    expected_token_scores = _get_token_scores(expected_logits,
                                              batch['decoder_target_tokens'],
                                              batch['decoder_loss_weights'])
    expected_sequence_scores = expected_token_scores.sum(-1)

    chex.assert_trees_all_close(expected_sequence_scores, sequence_scores)

  def test_score_batch_num_latents_equal_sequence_length(self):
    num_latents = 8
    vocab_size = 2
    model = t5_models.PerceiverARModel(
        module=None,
        vocabulary=None,
        optimizer_def=None,
        num_latents=num_latents,
        decoding_latent_reset_fill=3)

    batch = {
        'decoder_target_tokens': jnp.ones([2, 8]),
        'decoder_input_tokens': jnp.ones([2, 8]).at[:, 0].set(0),
        'decoder_loss_weights': jnp.ones([2, 8]),
        'decoder_causal_attention': jnp.zeros([2, 8]),
    }

    with mock.patch.object(
        model,
        '_compute_logits',
        new=functools.partial(
            _mock_compute_logits,
            num_latents=num_latents,
            vocab_size=vocab_size)):
      sequence_scores = model.score_batch(params=None, batch=batch)

    expected_logits = jnp.array([[[80., 0.], [80., 0.], [80., 0.], [80., 0.],
                                  [80., 0.], [80., 0.], [80., 0.], [80., 0.]]])
    expected_logits = jnp.tile(expected_logits, (2, 1, 1))

    expected_token_scores = _get_token_scores(expected_logits,
                                              batch['decoder_target_tokens'],
                                              batch['decoder_loss_weights'])
    expected_sequence_scores = expected_token_scores.sum(-1)

    chex.assert_trees_all_close(expected_sequence_scores, sequence_scores)

  def test_score_batch_refill_matches_latents(self):
    num_latents = 4
    vocab_size = 2
    model = t5_models.PerceiverARModel(
        module=None,
        vocabulary=None,
        optimizer_def=None,
        num_latents=num_latents,
        decoding_latent_reset_fill=4)

    batch = {
        'decoder_target_tokens': jnp.ones([2, 8]),
        'decoder_input_tokens': jnp.ones([2, 8]).at[:, 0].set(0),
        'decoder_loss_weights': jnp.ones([2, 8]),
        'decoder_causal_attention': jnp.zeros([2, 8]),
    }

    with mock.patch.object(
        model,
        '_compute_logits',
        new=functools.partial(
            _mock_compute_logits,
            num_latents=num_latents,
            vocab_size=vocab_size)):
      sequence_scores = model.score_batch(params=None, batch=batch)

    expected_logits = jnp.array([[[40., 0.], [40., 0.], [40., 0.], [40., 0.],
                                  [50., 0.], [60., 0.], [70., 0.], [80., 0.]]])
    expected_logits = jnp.tile(expected_logits, (2, 1, 1))

    expected_token_scores = _get_token_scores(expected_logits,
                                              batch['decoder_target_tokens'],
                                              batch['decoder_loss_weights'])
    expected_sequence_scores = expected_token_scores.sum(-1)

    chex.assert_trees_all_close(expected_sequence_scores, sequence_scores)


class T5ModelsDecodingLatentResetFillTest(absltest.TestCase):

  def test_get_decoding_latent_reset_fill(self):
    model = t5_models.PerceiverARModel(
        module=None, vocabulary=None, optimizer_def=None, num_latents=2048)

    # Short sequence, use all the latents.
    self.assertEqual(2048,
                     model.get_decoding_latent_reset_fill(input_length=32))

    # Sequence length equal to num_latents, use them all.
    self.assertEqual(2048,
                     model.get_decoding_latent_reset_fill(input_length=2048))

    # Sequence length only 1 longer than num_latents, so use all the latents
    # because this still results in only 2 forward passes.
    self.assertEqual(2048,
                     model.get_decoding_latent_reset_fill(input_length=2049))

    # Sequence now 2 more than num_latents, so use num_latents-1 for 2 passes.
    self.assertEqual(2047,
                     model.get_decoding_latent_reset_fill(input_length=2050))

    # Sequence is very long, use standard num_latents-128.
    self.assertEqual(1920,
                     model.get_decoding_latent_reset_fill(input_length=5000))

    # Ensure we can decode with only 1 latent.
    model_single_latent = t5_models.PerceiverARModel(
        module=None, vocabulary=None, optimizer_def=None, num_latents=1)
    self.assertEqual(
        1,
        model_single_latent.get_decoding_latent_reset_fill(input_length=8192))

  def test_get_decoding_latent_reset_fill_when_configured(self):
    model = t5_models.PerceiverARModel(
        module=None,
        vocabulary=None,
        optimizer_def=None,
        num_latents=2048,
        decoding_latent_reset_fill=2000)

    # Short sequence, use all the latents.
    self.assertEqual(2048,
                     model.get_decoding_latent_reset_fill(input_length=32))

    # Sequence length equal to num_latents, use them all.
    self.assertEqual(2048,
                     model.get_decoding_latent_reset_fill(input_length=2048))

    # Sequence length only 1 longer than num_latents, so use all the latents
    # because this still results in only 2 forward passes.
    self.assertEqual(2048,
                     model.get_decoding_latent_reset_fill(input_length=2049))

    # Sequence now 2 more than num_latents, so use num_latents-1 for 2 passes.
    self.assertEqual(2047,
                     model.get_decoding_latent_reset_fill(input_length=2050))

    # Sequence is very long, use configured value.
    self.assertEqual(2000,
                     model.get_decoding_latent_reset_fill(input_length=5000))

  def test_get_decoding_latent_reset_fill_when_configured_max(self):
    model = t5_models.PerceiverARModel(
        module=None,
        vocabulary=None,
        optimizer_def=None,
        num_latents=2048,
        decoding_latent_reset_fill=2048)

    # Short sequence, use all the latents.
    self.assertEqual(2048,
                     model.get_decoding_latent_reset_fill(input_length=32))

    # Sequence length equal to num_latents, use them all.
    self.assertEqual(2048,
                     model.get_decoding_latent_reset_fill(input_length=2048))

    # Sequence length only 1 longer than num_latents, so use all the latents
    # because this still results in only 2 forward passes.
    self.assertEqual(2048,
                     model.get_decoding_latent_reset_fill(input_length=2049))

    # Sequence now 2 more than num_latents, so use num_latents-1 for 2 passes.
    # Except configured value is higher, so use that.
    self.assertEqual(2048,
                     model.get_decoding_latent_reset_fill(input_length=2050))

    # Sequence is very long, use configured value.
    self.assertEqual(2048,
                     model.get_decoding_latent_reset_fill(input_length=5000))

if __name__ == '__main__':
  absltest.main()
