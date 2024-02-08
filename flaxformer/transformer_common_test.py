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

"""Tests for flaxformer.bert_model."""
from absl.testing import absltest
from flax import linen as nn
from jax import random
import jax.numpy as jnp
import numpy as np
from flaxformer import transformer_common as common


class MockDecoderLayer(nn.Module):
  """A test layer that takes two inputs."""
  hidden_size: int

  @nn.compact
  def __call__(self, inputs, encoder_outputs):
    return nn.Dense(self.hidden_size)(
        jnp.concatenate([inputs, encoder_outputs], -1))


class TransformerCommonTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = random.PRNGKey(0)

  def test_encoder_layer_stack_output_shape_ok(self):
    """Tests that an encoder layer stack outputs are of the correct shape."""
    batch_size, seq_len, embedding_size, hidden_size = 2, 3, 4, 5
    input_shape = (batch_size, seq_len, embedding_size)
    inputs = random.uniform(self.rng, input_shape, dtype=jnp.float32)
    num_layers = 3

    def make_layer():
      return nn.Dense(hidden_size)

    model = common.LayerSequence(num_layers=num_layers, make_layer=make_layer)
    result, _ = model.init_with_output(self.rng, inputs)
    expected = (batch_size, seq_len, hidden_size)
    self.assertEqual(expected, result.shape)

  def test_encoder_layer_stack_applies_layer_range_correctly(self):
    """Tests that an encoder layer stack correctly applies a range of layers."""
    batch_size, seq_len, embedding_size, hidden_size = 2, 3, 4, 5
    input_shape = (batch_size, seq_len, embedding_size)
    inputs = random.uniform(self.rng, input_shape, dtype=jnp.float32)
    num_layers = 3

    def make_layer():
      return nn.Dense(hidden_size)

    model = common.LayerSequence(num_layers=num_layers, make_layer=make_layer)
    out, variables = model.init_with_output(self.rng, inputs)

    partial_out = model.apply(
        variables, 0, 1, inputs, method=model.apply_range_of_layers)
    full_out = model.apply(
        variables, 1, None, partial_out, method=model.apply_range_of_layers)
    np.testing.assert_allclose(out, full_out, rtol=1e-5)

  def test_decoder_layer_stack_output_shape_ok(self):
    """Tests that a decoder layer stack outputs are of the correct shape."""
    batch_size, seq_len, embedding_size, hidden_size = 2, 3, 4, 5
    input_shape = (batch_size, seq_len, embedding_size)
    encoder_outputs_shape = (batch_size, seq_len, hidden_size)
    inputs = random.uniform(self.rng, input_shape, dtype=jnp.float32)
    encoder_outputs = random.uniform(
        self.rng, encoder_outputs_shape, dtype=jnp.float32)
    num_layers = 3

    def make_layer():
      return MockDecoderLayer(hidden_size)

    model = common.LayerSequence(num_layers=num_layers, make_layer=make_layer)
    result, _ = model.init_with_output(self.rng, inputs, encoder_outputs)
    expected = (batch_size, seq_len, hidden_size)
    self.assertEqual(expected, result.shape)

  def test_decoder_layer_stack_applies_layer_range_correctly(self):
    """Tests that a decoder layer stack correctly applies a range of layers."""
    batch_size, seq_len, embedding_size, hidden_size = 2, 3, 4, 5
    input_shape = (batch_size, seq_len, embedding_size)
    encoder_outputs_shape = (batch_size, seq_len, hidden_size)
    inputs = random.uniform(self.rng, input_shape, dtype=jnp.float32)
    encoder_outputs = random.uniform(
        self.rng, encoder_outputs_shape, dtype=jnp.float32)
    num_layers = 3

    def make_layer():
      return MockDecoderLayer(hidden_size)

    model = common.LayerSequence(num_layers=num_layers, make_layer=make_layer)
    out, variables = model.init_with_output(self.rng, inputs, encoder_outputs)

    partial_out = model.apply(
        variables,
        0,
        1,
        inputs,
        encoder_outputs,
        method=model.apply_range_of_layers)
    full_out = model.apply(
        variables,
        1,
        None,
        partial_out,
        encoder_outputs,
        method=model.apply_range_of_layers)
    np.testing.assert_allclose(out, full_out, rtol=1e-5)


class TransparentLayerSequenceTest(absltest.TestCase):

  def test_transparent_layer_sequence_equals_regular(self):
    batch_size, seq_len, embedding_size, hidden_size = 2, 3, 4, 5
    input_shape = (batch_size, seq_len, embedding_size)
    encoder_outputs_shape = (batch_size, seq_len, hidden_size)
    inputs = random.uniform(random.PRNGKey(0), input_shape, dtype=jnp.float32)
    encoder_outputs = random.uniform(
        random.PRNGKey(1), encoder_outputs_shape, dtype=jnp.float32)
    num_layers = 3

    def make_layer():
      return MockDecoderLayer(hidden_size)

    model = common.LayerSequence(num_layers=num_layers, make_layer=make_layer)
    out, variables = model.init_with_output(
        random.PRNGKey(2), inputs, encoder_outputs)

    class OuterModule(nn.Module):

      def setup(self):
        self.layers = [make_layer() for _ in range(num_layers)]
        self.layer_sequence = common.TransparentLayerSequence(self.layers)

      def __call__(self, *args, **kwargs):
        return self.layer_sequence(*args, **kwargs)

    model2 = OuterModule()
    out2 = model2.apply(variables, inputs, encoder_outputs)
    np.testing.assert_allclose(out, out2, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
