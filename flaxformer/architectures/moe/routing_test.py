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

"""Tests for routing."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np

from flaxformer.architectures.moe import routing


class RoutingTest(parameterized.TestCase):

  def test_load_balancing_loss(self):
    num_tokens = 5
    num_experts = 2
    num_selected_experts = 1
    rng = jax.random.PRNGKey(0)

    router_probs = jax.random.uniform(
        rng, (num_tokens, num_experts), minval=0, maxval=1)
    expert_indices = jax.random.randint(
        rng, (num_tokens, num_selected_experts), minval=0, maxval=2)

    self.assertEqual(
        routing._load_balancing_loss(router_probs, expert_indices), 0.8741045)

  def test_tokens_choose_one_expert_scatter_router(self):
    num_groups = 2
    tokens_per_group = 4
    hidden_dim = 4
    expert_capacity = 2
    num_experts = 4
    num_selected_experts = 1  # Switch routing case
    rng = jax.random.PRNGKey(0)

    token_inputs = jax.random.uniform(
        rng, (num_groups, tokens_per_group, hidden_dim), minval=0, maxval=1)
    router_indices, _ = routing.TokensChooseScatterRouter(
        router_weights=routing.RouterWeights(name='router_weights'),
        num_selected_experts=num_selected_experts,
        jitter_noise=0.01,
        batch_prioritized_routing=True,
        ignore_padding_tokens=False,
        dtype=jnp.float32).init_with_output(
            {
                'params': jax.random.PRNGKey(0),
                'jitter': jax.random.PRNGKey(0)
            }, token_inputs, num_experts, expert_capacity)

    expected_indices = jnp.array([
        [
            [[2, 1]],
            [[3, 0]],
            [[2, 0]],
            [[2, 2]],
        ],
        [
            [[3, 0]],
            [[2, 1]],
            [[3, 1]],
            [[2, 0]],
        ],
    ],
                                 dtype=jnp.int32)

    np.testing.assert_allclose(router_indices.dispatch_indices,
                               expected_indices)

    expected_weights = jnp.array([
        [[0.25390625], [0.2578125], [0.25585938], [0.]],
        [[0.2578125], [0.25390625], [0.25585938], [0.25585938]],
    ],
                                 dtype=jnp.float32)
    np.testing.assert_allclose(router_indices.combine_weights, expected_weights)

    self.assertEqual(router_indices.auxiliary_loss, 1.0166016)
    self.assertEqual(router_indices.router_z_loss, 1.8994141)

  def test_tokens_choose_one_expert_scatter_router_no_bpr(self):
    num_groups = 2
    tokens_per_group = 4
    hidden_dim = 4
    expert_capacity = 2
    num_experts = 4
    num_selected_experts = 1  # Switch routing case
    rng = jax.random.PRNGKey(0)

    token_inputs = jax.random.uniform(
        rng, (num_groups, tokens_per_group, hidden_dim), minval=0, maxval=1)
    router_indices, _ = routing.TokensChooseScatterRouter(
        router_weights=routing.RouterWeights(name='router_weights'),
        num_selected_experts=num_selected_experts,
        jitter_noise=0.01,
        batch_prioritized_routing=False,
        ignore_padding_tokens=False,
        dtype=jnp.float32).init_with_output(
            {
                'params': jax.random.PRNGKey(0),
                'jitter': jax.random.PRNGKey(0)
            }, token_inputs, num_experts, expert_capacity)

    expected_indices = jnp.array([
        [
            [[2, 0]],
            [[3, 0]],
            [[2, 1]],
            [[2, 2]],
        ],
        [
            [[3, 0]],
            [[2, 0]],
            [[3, 1]],
            [[2, 1]],
        ],
    ],
                                 dtype=jnp.int32)

    np.testing.assert_allclose(router_indices.dispatch_indices,
                               expected_indices)

    expected_weights = jnp.array([
        [[0.25390625], [0.2578125], [0.25585938], [0.]],
        [[0.2578125], [0.25390625], [0.25585938], [0.25585938]],
    ],
                                 dtype=jnp.float32)
    np.testing.assert_allclose(router_indices.combine_weights, expected_weights)

    self.assertEqual(router_indices.auxiliary_loss, 1.0166016)
    self.assertEqual(router_indices.router_z_loss, 1.8994141)

  def test_tokens_choose_multiple_experts_scatter_router(self):
    num_groups = 2
    tokens_per_group = 4
    hidden_dim = 4
    expert_capacity = 2
    num_experts = 4
    num_selected_experts = 2
    rng = jax.random.PRNGKey(0)

    token_inputs = jax.random.uniform(
        rng, (num_groups, tokens_per_group, hidden_dim), minval=0, maxval=1)
    router_indices, _ = routing.TokensChooseScatterRouter(
        router_weights=routing.RouterWeights(name='router_weights'),
        num_selected_experts=num_selected_experts,
        jitter_noise=0.01,
        batch_prioritized_routing=True,
        ignore_padding_tokens=False,
        dtype=jnp.float32).init_with_output(
            {
                'params': jax.random.PRNGKey(0),
                'jitter': jax.random.PRNGKey(0)
            }, token_inputs, num_experts, expert_capacity)

    expected_indices = jnp.array([
        [
            [[2, 1], [3, 2]],
            [[3, 0], [2, 3]],
            [[2, 0], [3, 1]],
            [[2, 2], [0, 0]],
        ],
        [
            [[3, 0], [2, 2]],
            [[2, 1], [3, 3]],
            [[3, 1], [2, 3]],
            [[2, 0], [3, 2]],
        ],
    ],
                                 dtype=jnp.int32)

    np.testing.assert_allclose(router_indices.dispatch_indices,
                               expected_indices)

    expected_weights = jnp.array([
        [
            [0.25390625, 0.],
            [0.2578125, 0.],
            [0.25585938, 0.25390625],
            [0., 0.25],
        ],
        [
            [0.2578125, 0.],
            [0.25390625, 0.],
            [0.25585938, 0.],
            [0.25585938, 0.],
        ],
    ],
                                 dtype=jnp.float32)
    np.testing.assert_allclose(router_indices.combine_weights, expected_weights)

    self.assertEqual(router_indices.auxiliary_loss, 2.0289307)
    self.assertEqual(router_indices.router_z_loss, 1.8994141)

  def test_tokens_choose_one_expert_mask_router(self):
    num_groups = 2
    tokens_per_group = 3
    hidden_dim = 4
    num_experts = 2
    num_selected_experts = 1  # Switch routing case
    expert_capacity = 1  # Total capacity = 2*2*1 = 4 < num_tokens
    rng = jax.random.PRNGKey(0)

    token_inputs = jax.random.uniform(
        rng, (num_groups, tokens_per_group, hidden_dim), minval=0, maxval=1)
    router_mask, _ = routing.TokensChooseMaskedRouter(
        router_weights=routing.RouterWeights(name='router_weights'),
        num_selected_experts=num_selected_experts,
        jitter_noise=0.,
        batch_prioritized_routing=True,
        ignore_padding_tokens=False,
        dtype=jnp.float32).init_with_output(
            jax.random.PRNGKey(0), token_inputs, num_experts, expert_capacity)

    expected_mask = jnp.array([
        [
            [[True], [False]],
            [[False], [True]],
            [[False], [False]],
        ],
        [
            [[True], [False]],
            [[False], [True]],
            [[False], [False]],
        ],
    ],
                              dtype=jnp.bool_)

    np.testing.assert_allclose(router_mask.dispatch_mask, expected_mask)

    expected_weights = jnp.array([
        [
            [[0.5078125], [0.]],
            [[0.], [0.50390625]],
            [[0.], [0.]],
        ],
        [
            [[0.50390625], [0.]],
            [[0.], [0.5078125]],
            [[0.], [0.]],
        ],
    ],
                                 dtype=jnp.float32)
    np.testing.assert_allclose(router_mask.combine_array, expected_weights)

    self.assertEqual(router_mask.auxiliary_loss, 1.0006511)
    self.assertEqual(router_mask.router_z_loss, 0.47721353)

  def test_routers_ignore_padding(self):
    num_groups = 2
    tokens_per_group = 6
    hidden_dim = 2
    num_experts = 2
    num_selected_experts = 2
    expert_capacity = 1  # Total capacity = 2*2*1 = 4 < num_tokens
    rng = jax.random.PRNGKey(0)

    token_inputs = jax.random.uniform(
        rng, (num_groups, tokens_per_group, hidden_dim), minval=0, maxval=1)
    # Simulate masked inputs.
    padding_mask = jax.random.randint(
        rng, (num_groups, tokens_per_group, 1), minval=0, maxval=2)
    token_inputs *= padding_mask

    router_weights = routing.RouterWeights(name='router_weights')

    with self.subTest(name='tokens_choose_masked_router'):
      router_mask, _ = routing.TokensChooseMaskedRouter(
          router_weights=router_weights,
          num_selected_experts=num_selected_experts,
          jitter_noise=0.,
          batch_prioritized_routing=True,
          ignore_padding_tokens=True,
          dtype=jnp.float32).init_with_output(
              jax.random.PRNGKey(0), token_inputs, num_experts, expert_capacity)

      expected_mask = jnp.array([
          [
              [[False], [False]],
              [[True], [True]],
              [[False], [False]],
              [[False], [False]],
              [[False], [False]],
              [[False], [False]],
          ],
          [
              [[False], [False]],
              [[True], [True]],
              [[False], [False]],
              [[False], [False]],
              [[False], [False]],
              [[False], [False]],
          ],
      ],
                                dtype=jnp.bool_)

      np.testing.assert_allclose(router_mask.dispatch_mask, expected_mask)

      expected_weights = jnp.array([
          [
              [[0.], [0.]],
              [[0.50390625], [0.49804688]],
              [[0.0], [0.]],
              [[0.0], [0.]],
              [[0.0], [0.]],
              [[0.0], [0.]],
          ],
          [
              [[0.], [0.]],
              [[0.50390625], [0.49414062]],
              [[0.], [0.]],
              [[0.], [0.]],
              [[0.], [0.]],
              [[0.], [0.]],
          ],
      ],
                                   dtype=jnp.float32)
      np.testing.assert_allclose(router_mask.combine_array, expected_weights)

      self.assertEqual(router_mask.auxiliary_loss, 0.6951497)
      self.assertEqual(router_mask.router_z_loss, 0.48541257)

    with self.subTest(name='tokens_choose_scatter_router'):
      router_indices, _ = routing.TokensChooseScatterRouter(
          router_weights=router_weights,
          num_selected_experts=num_selected_experts,
          jitter_noise=0.,
          batch_prioritized_routing=True,
          ignore_padding_tokens=True,
          dtype=jnp.float32).init_with_output(
              jax.random.PRNGKey(0), token_inputs, num_experts, expert_capacity)

      expected_indices = jnp.array([
          [
              [[0, 4], [1, 4]],
              [[0, 0], [1, 0]],
              [[0, 1], [1, 1]],
              [[0, 5], [1, 5]],
              [[0, 2], [1, 2]],
              [[0, 3], [1, 3]],
          ],
          [
              [[0, 3], [1, 3]],
              [[0, 0], [1, 0]],
              [[0, 1], [1, 1]],
              [[0, 4], [1, 4]],
              [[0, 2], [1, 2]],
              [[0, 5], [1, 5]],
          ],
      ],
                                   dtype=jnp.int32)

      np.testing.assert_allclose(router_indices.dispatch_indices,
                                 expected_indices)

      expected_weights = jnp.array([
          [
              [0., 0.],
              [0.50390625, 0.49804688],
              [0., 0.],
              [0., 0.],
              [0., 0.],
              [0., 0.],
          ],
          [
              [0., 0.],
              [0.50390625, 0.49414062],
              [0., 0.],
              [0., 0.],
              [0., 0.],
              [0., 0.],
          ],
      ],
                                   dtype=jnp.float32)
      np.testing.assert_allclose(router_indices.combine_weights,
                                 expected_weights)

      self.assertEqual(router_indices.auxiliary_loss, 1.1676432)
      self.assertEqual(router_indices.router_z_loss, 0.48541257)

    with self.subTest(name='experts_choose_masked_router'):
      router_mask, _ = routing.ExpertsChooseMaskedRouter(
          router_weights=router_weights,
          jitter_noise=0.,
          ignore_padding_tokens=True,
          dtype=jnp.float32).init_with_output(
              jax.random.PRNGKey(0), token_inputs, num_experts, expert_capacity)

      expected_mask = jnp.array([
          [
              [[0], [0]],
              [[1], [1]],
              [[0], [0]],
              [[0], [0]],
              [[0], [0]],
              [[0], [0]],
          ],
          [
              [[0], [0]],
              [[1], [0]],
              [[0], [1]],
              [[0], [0]],
              [[0], [0]],
              [[0], [0]],
          ],
      ],
                                dtype=jnp.bool_)

      np.testing.assert_allclose(router_mask.dispatch_mask, expected_mask)

      expected_weights = jnp.array([
          [
              [[0.], [0.]],
              [[0.50390625], [0.49804688]],
              [[0.0], [0.]],
              [[0.0], [0.]],
              [[0.0], [0.]],
              [[0.0], [0.]],
          ],
          [
              [[0.], [0.]],
              [[0.50390625], [0.]],
              [[0.], [0.49804688]],
              [[0.], [0.]],
              [[0.], [0.]],
              [[0.], [0.]],
          ],
      ],
                                   dtype=jnp.float32)
      np.testing.assert_allclose(router_mask.combine_array, expected_weights)

      self.assertEqual(router_mask.auxiliary_loss, 0.)
      self.assertEqual(router_mask.router_z_loss, 0.48541257)

  def test_tokens_choose_one_expert_mask_router_no_bpr(self):
    num_groups = 2
    tokens_per_group = 3
    hidden_dim = 4
    num_experts = 2
    num_selected_experts = 1  # Switch routing case
    expert_capacity = 1  # Total capacity = 2*2*1 = 4 < num_tokens
    rng = jax.random.PRNGKey(0)

    token_inputs = jax.random.uniform(
        rng, (num_groups, tokens_per_group, hidden_dim), minval=0, maxval=1)
    router_mask, _ = routing.TokensChooseMaskedRouter(
        router_weights=routing.RouterWeights(name='router_weights'),
        num_selected_experts=num_selected_experts,
        jitter_noise=0.,
        batch_prioritized_routing=False,
        ignore_padding_tokens=False,
        dtype=jnp.float32).init_with_output(
            jax.random.PRNGKey(0), token_inputs, num_experts, expert_capacity)

    expected_mask = jnp.array([
        [
            [[True], [False]],
            [[False], [True]],
            [[False], [False]],
        ],
        [
            [[True], [False]],
            [[False], [True]],
            [[False], [False]],
        ],
    ],
                              dtype=jnp.bool_)

    np.testing.assert_allclose(router_mask.dispatch_mask, expected_mask)

    expected_weights = jnp.array([
        [
            [[0.5078125], [0.]],
            [[0.], [0.50390625]],
            [[0.], [0.]],
        ],
        [
            [[0.50390625], [0.]],
            [[0.], [0.5078125]],
            [[0.], [0.]],
        ],
    ],
                                 dtype=jnp.float32)
    np.testing.assert_allclose(router_mask.combine_array, expected_weights)

    self.assertEqual(router_mask.auxiliary_loss, 1.0006511)
    self.assertEqual(router_mask.router_z_loss, 0.47721353)

  def test_tokens_choose_multiple_experts_mask_router(self):
    num_groups = 2
    tokens_per_group = 4
    hidden_dim = 3
    num_experts = 3
    num_selected_experts = 2
    expert_capacity = 1
    rng = jax.random.PRNGKey(0)

    token_inputs = jax.random.uniform(
        rng, (num_groups, tokens_per_group, hidden_dim), minval=0, maxval=1)
    router_mask, _ = routing.TokensChooseMaskedRouter(
        router_weights=routing.RouterWeights(name='router_weights'),
        num_selected_experts=num_selected_experts,
        jitter_noise=0.01,
        batch_prioritized_routing=True,
        ignore_padding_tokens=False,
        dtype=jnp.float32).init_with_output(
            {
                'params': jax.random.PRNGKey(0),
                'jitter': jax.random.PRNGKey(0)
            }, token_inputs, num_experts, expert_capacity)

    expected_mask = jnp.array([
        [
            [[True], [False], [True]],
            [[False], [True], [False]],
            [[False], [False], [False]],
            [[False], [False], [False]],
        ],
        [
            [[True], [True], [False]],
            [[False], [False], [True]],
            [[False], [False], [False]],
            [[False], [False], [False]],
        ],
    ],
                              dtype=jnp.bool_)

    np.testing.assert_allclose(router_mask.dispatch_mask, expected_mask)

    expected_weights = jnp.array([
        [
            [[0.33203125], [0.], [0.3359375]],
            [[0.], [0.3359375], [0.]],
            [[0.], [0.], [0.]],
            [[0.], [0.], [0.]],
        ],
        [
            [[0.33007812], [0.34179688], [0.]],
            [[0.], [0.], [0.3359375]],
            [[0.], [0.], [0.]],
            [[0.], [0.], [0.]],
        ],
    ],
                                 dtype=jnp.float32)
    np.testing.assert_allclose(router_mask.combine_array, expected_weights)

    self.assertEqual(router_mask.auxiliary_loss, 2.001709)
    self.assertEqual(router_mask.router_z_loss, 1.2714844)

  def test_experts_choose_mask_router(self):
    num_groups = 2
    tokens_per_group = 4
    hidden_dim = 3
    num_experts = 2
    expert_capacity = 2
    rng = jax.random.PRNGKey(0)

    token_inputs = jax.random.uniform(
        rng, (num_groups, tokens_per_group, hidden_dim), minval=0, maxval=1)
    router_mask, _ = routing.ExpertsChooseMaskedRouter(
        router_weights=routing.RouterWeights(name='router_weights'),
        jitter_noise=0.,
        dtype=jnp.float32,
        ignore_padding_tokens=False).init_with_output(
            jax.random.PRNGKey(0), token_inputs, num_experts, expert_capacity)

    expected_mask = jnp.array([
        [
            [[0, 1], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 0]],
        ],
        [
            [[1, 0], [0, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 0], [0, 1]],
        ],
    ],
                              dtype=jnp.int32)

    np.testing.assert_allclose(router_mask.dispatch_mask, expected_mask)

    expected_weights = jnp.array([
        [
            [[0., 0.49609375], [0.50390625, 0.]],
            [[0., 0.], [0., 0.50390625]],
            [[0.49804688, 0.], [0., 0.]],
            [[0., 0.], [0., 0.]],
        ],
        [
            [[0.49804688, 0.], [0., 0.]],
            [[0., 0.49414062], [0., 0.]],
            [[0., 0.], [0.5078125, 0.]],
            [[0., 0.], [0., 0.5078125]],
        ],
    ],
                                 dtype=jnp.float32)
    np.testing.assert_allclose(router_mask.combine_array, expected_weights)

    # Auxiliary loss is always 0. for experts choose tokens routing.
    self.assertEqual(router_mask.auxiliary_loss, 0.)
    self.assertEqual(router_mask.router_z_loss, 0.5041504)

  def test_scatter_and_mask_dispatch_equal(self):
    num_groups = 2
    tokens_per_group = 4
    hidden_dim = 3
    num_experts = 3
    num_selected_experts = 1
    expert_capacity = 2
    rng = jax.random.PRNGKey(0)

    router_weights = routing.RouterWeights(name='router_weights')

    token_inputs = jax.random.uniform(
        rng, (num_groups, tokens_per_group, hidden_dim), minval=0, maxval=1)

    router_mask, _ = routing.TokensChooseMaskedRouter(
        router_weights,
        num_selected_experts=num_selected_experts,
        jitter_noise=0.,
        batch_prioritized_routing=True,
        dtype=jnp.float32,
        ignore_padding_tokens=False).init_with_output(
            jax.random.PRNGKey(0), token_inputs, num_experts, expert_capacity)
    # Manipulate masked router dispatch and combine arrays to match format of
    # scatter router output.
    # Ignore capacity. Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_EXPERTS]
    masked_router_says_dispatched = jnp.max(router_mask.dispatch_mask, axis=-1)
    # Ignore particular expert and capacity for combine array.
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP]
    masked_router_combine_array = jnp.max(
        router_mask.combine_array, axis=(-1, -2))

    router_indices, _ = routing.TokensChooseScatterRouter(
        router_weights,
        num_selected_experts=num_selected_experts,
        jitter_noise=0.,
        batch_prioritized_routing=True,
        dtype=jnp.float32,
        ignore_padding_tokens=False).init_with_output(
            jax.random.PRNGKey(0), token_inputs, num_experts, expert_capacity)
    # Manipulate scatter router dispatch and combine indices to match format of
    # masked router output.
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_SELECTED_EXPERTS]
    successfully_routed = router_indices.dispatch_indices[...,
                                                          1] < expert_capacity
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP, NUM_EXPERTS]
    scatter_router_says_dispatched = successfully_routed * jax.nn.one_hot(
        router_indices.dispatch_indices[..., 0].squeeze(axis=-1), num_experts)
    # Remove trivial selected expert axis.
    # Shape: [NUM_GROUPS, TOKENS_PER_GROUP].
    scatter_router_combine_array = router_indices.combine_weights.squeeze(
        axis=-1)

    np.testing.assert_allclose(masked_router_says_dispatched,
                               scatter_router_says_dispatched)
    np.testing.assert_allclose(masked_router_combine_array,
                               scatter_router_combine_array)
    np.testing.assert_allclose(router_mask.auxiliary_loss,
                               router_indices.auxiliary_loss)
    np.testing.assert_allclose(router_mask.router_z_loss,
                               router_indices.router_z_loss)

  def test_router_z_loss(self):
    num_groups = 2
    num_tokens = 6
    num_experts = 4
    rng = jax.random.PRNGKey(0)

    router_logits = jax.random.uniform(
        rng, (num_groups, num_tokens, num_experts), minval=-5, maxval=5)

    self.assertEqual(routing._router_z_loss(router_logits), 13.786719)


if __name__ == '__main__':
  absltest.main()
