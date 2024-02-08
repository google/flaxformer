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

"""Tests for t5_architecture."""

import functools

from absl.testing import absltest
import jax
from jax import random
from jax import tree_util
import numpy as np

from flaxformer import sharding
from flaxformer import testing_utils
from flaxformer.architectures.t5 import t5_architecture_test_utils as t5_test_utils

expected_files = testing_utils.ExpectedJsonFiles(
    'flaxformer/architectures/t5/testdata')
check_params = expected_files.check_params_shapes_only


class EncoderDecoderTest(absltest.TestCase):

  def test_encoder_shapes_with_relative_attention_per_layer(self):
    transformer = t5_test_utils.make_config1()
    inputs = np.array(
        [
            # Batch 1.
            [101, 183, 20, 75],
            # Batch 2.
            [101, 392, 19, 7],
        ],
        dtype=np.int32)
    output, variables = transformer.init_with_output(
        random.PRNGKey(0),
        inputs,
        enable_dropout=False,
        method=transformer.encode,
    )

    reformatted = transformer.apply({},
                                    variables['params'],
                                    method=transformer.to_save_format)
    check_params(reformatted, 'encoder_shapes_per_layer_relpos_bias.json')
    self.assertEqual(output.shape, (2, 4, 13))

    # Convert back to Flax module structure format and test again.
    params2 = transformer.apply({},
                                reformatted,
                                method=transformer.from_save_format)
    output2 = transformer.apply(
        {'params': params2},
        inputs,
        enable_dropout=False,
        method=transformer.encode,
    )
    np.testing.assert_allclose(output, output2, rtol=1e-8)

  def test_parallel_transformer_config(self):
    transformer = t5_test_utils.make_parallel_transformer_config()
    inputs = np.array(
        [
            # Batch 1.
            [101, 183, 20, 75],
            # Batch 2.
            [101, 392, 19, 7],
        ],
        dtype=np.int32)
    output, variables = transformer.init_with_output(
        random.PRNGKey(0),
        inputs,
        enable_dropout=False,
        method=transformer.encode,
    )

    reformatted = transformer.apply({},
                                    variables['params'],
                                    method=transformer.to_save_format)
    expected_files.check_params(reformatted,
                                'parallel_transformer_encoder_shapes.json')
    self.assertEqual(output.shape, (2, 4, 13))

    # Convert back to Flax module structure format and test again.
    params2 = transformer.apply({},
                                reformatted,
                                method=transformer.from_save_format)
    output2 = transformer.apply(
        {'params': params2},
        inputs,
        enable_dropout=False,
        method=transformer.encode,
    )
    np.testing.assert_allclose(output, output2, rtol=1e-8)

  def test_encode_shared_relative_position_bias(self):
    transformer = t5_test_utils.make_config2_shared_relative_position_bias()
    inputs = np.array(
        [
            # Batch 1.
            [101, 183, 20, 75],
            # Batch 2.
            [101, 392, 19, 7],
        ],
        dtype=np.int32)
    output, variables = transformer.init_with_output(
        random.PRNGKey(0),
        inputs,
        enable_dropout=False,
        method=transformer.encode,
    )

    reformatted = transformer.apply({},
                                    variables['params'],
                                    method=transformer.to_save_format)
    check_params(reformatted, 'encoder_shapes_shared_relpos_bias.json')
    self.assertEqual(output.shape, (2, 4, 13))

    # Convert back to Flax module structure format and test again.
    params2 = transformer.apply({},
                                reformatted,
                                method=transformer.from_save_format)
    output2 = transformer.apply(
        {'params': params2},
        inputs,
        enable_dropout=False,
        method=transformer.encode,
    )
    np.testing.assert_allclose(output, output2, rtol=1e-8)

  def test_encoder_example_packing(self):
    transformer = t5_test_utils.make_config1()
    encoder_input_tokens = np.array(
        [
            # Batch 1.
            [101, 183, 20, 75],
            # Batch 2.
            [101, 392, 19, 0],
        ],
        dtype=np.int32)
    output, variables = transformer.init_with_output(
        random.PRNGKey(0),
        encoder_input_tokens,
        enable_dropout=False,
        method=transformer.encode,
    )

    encoder_input_tokens_packed = np.array([[101, 183, 20, 75, 101, 392, 19]],
                                           dtype=np.int32)
    encoder_segment_ids = np.array([[0, 0, 0, 0, 1, 1, 1]], dtype=np.int32)
    encoder_input_positions = np.array([[0, 1, 2, 3, 0, 1, 2]], dtype=np.int32)
    output_packed = transformer.apply(
        variables,
        encoder_input_tokens_packed,
        encoder_segment_ids=encoder_segment_ids,
        encoder_positions=encoder_input_positions,
        enable_dropout=False,
        method=transformer.encode,
    )


    # Check that the first element matches, which is entire first batch of the
    # padded setup, and the first 3 "tokens" of the packed example.
    np.testing.assert_allclose(
        output[0, :, :], output_packed[0, 0:4, :], rtol=1e-4)

    # Check that the second element matches, which is the first 3 "tokens" of
    # the padded example's second batch, and the last 3 of tokens the packed
    # example's first batch.
    np.testing.assert_allclose(
        output[1, 0:3, :], output_packed[0, 4:7, :], rtol=1e-4, atol=1e-4)

  def test_scan_and_remat(self):
    """Tests if encoder returns the same output for different scan/remat."""
    encoder_input_tokens = np.array(
        [
            # Batch 1.
            [101, 183, 20, 75],
            # Batch 2.
            [101, 392, 19, 7],
        ],
        dtype=np.int32)

    transformer1 = t5_test_utils.make_config1(
        scan_layers=False, layer_remat='none')
    output1, _ = transformer1.init_with_output(
        random.PRNGKey(0),
        encoder_input_tokens,
        enable_dropout=False,
        method=transformer1.encode,
    )

    transformer2 = t5_test_utils.make_config1(
        scan_layers=False, layer_remat='minimal')
    output2, _ = transformer2.init_with_output(
        random.PRNGKey(0),
        encoder_input_tokens,
        enable_dropout=False,
        method=transformer2.encode,
    )

    transformer3 = t5_test_utils.make_config1(
        scan_layers=False, layer_remat='full')
    output3, _ = transformer3.init_with_output(
        random.PRNGKey(0),
        encoder_input_tokens,
        enable_dropout=False,
        method=transformer3.encode,
    )

    transformer4 = t5_test_utils.make_config1(
        scan_layers=True, layer_remat='minimal')
    output4, _ = transformer4.init_with_output(
        random.PRNGKey(0),
        encoder_input_tokens,
        enable_dropout=False,
        method=transformer4.encode,
    )

    transformer5 = t5_test_utils.make_config1(
        scan_layers=True, layer_remat='full')
    output5, _ = transformer5.init_with_output(
        random.PRNGKey(0),
        encoder_input_tokens,
        enable_dropout=False,
        method=transformer5.encode,
    )

    # Check scan_layers=False results
    np.testing.assert_allclose(output1, output2, rtol=2e-4)
    np.testing.assert_allclose(output1, output3, atol=1e-5, rtol=1.5e-5)
    # Check scan_layers=True results
    np.testing.assert_allclose(output4, output5, rtol=1.5e-5)

  def test_scan_axis_annotations(self):
    encoder_input_tokens = np.array(
        [
            # Batch 1.
            [101, 183, 20, 75],
            # Batch 2.
            [101, 392, 19, 7],
        ],
        dtype=np.int32)
    transformer = t5_test_utils.make_config1(
        scan_layers=True, layer_remat='minimal')
    variables = transformer.init(
        random.PRNGKey(0),
        encoder_input_tokens,
        enable_dropout=False,
        method=transformer.encode,
    )

    # Check that the code can run when `params_axes` is not mutable too.
    transformer.apply(
        variables,
        encoder_input_tokens,
        enable_dropout=False,
        method=transformer.encode,
    )

    sharding.check_params_and_axis_names_match(variables)
    for axis_names in tree_util.tree_leaves(sharding.get_axis_names(variables)):
      for name in axis_names:
        self.assertIn(
            name, {
                'embed', 'joined_kv', 'heads', 'head_dim', 'relpos_buckets',
                'mlp', 'vocab', 'layers'
            },
            msg='unrecognized axis in variable')
    expected_files.check_params_and_axes(
        variables['params'],
        variables['params_axes'],
        'encoder_scanned_per_layer_relpos_bias.json',
    )

  def test_entire_transformer_shared_embeds(self):
    encoder_input_tokens = np.zeros((16, 8), dtype=np.float32)
    decoder_input_tokens = np.zeros((16, 8), dtype=np.float32)
    decoder_target_tokens = np.zeros((16, 8), dtype=np.float32)

    transformer = t5_test_utils.make_config3_shared_token_embedder()
    output, variables = transformer.init_with_output(
        random.PRNGKey(0),
        encoder_input_tokens=encoder_input_tokens,
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        enable_dropout=False,
    )
    params = variables['params']
    reformatted = transformer.apply({},
                                    params,
                                    method=transformer.to_save_format)
    check_params(reformatted, 'encoder_decoder_shared_embedding_shapes.json')
    self.assertEqual(output.shape, (16, 8, 71))

    # Convert back to Flax module structure format and test again.
    params2 = transformer.apply({},
                                reformatted,
                                method=transformer.from_save_format)
    output2 = transformer.apply(
        {'params': params2},
        encoder_input_tokens=encoder_input_tokens,
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        enable_dropout=False,
    )
    np.testing.assert_allclose(output, output2, rtol=1e-8)

  def test_axis_names(self):
    encoder_input_tokens = np.zeros((16, 8), dtype=np.float32)
    decoder_input_tokens = np.zeros((16, 8), dtype=np.float32)
    decoder_target_tokens = np.zeros((16, 8), dtype=np.float32)
    transformer = t5_test_utils.make_config3_shared_token_embedder()
    variables = jax.eval_shape(
        functools.partial(transformer.init, enable_dropout=False),
        random.PRNGKey(0),
        encoder_input_tokens=encoder_input_tokens,
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
    )
    sharding.check_params_and_axis_names_match(variables)
    for axis_names in tree_util.tree_leaves(sharding.get_axis_names(variables)):
      for name in axis_names:
        self.assertIn(
            name, {
                'embed', 'joined_kv', 'heads', 'head_dim', 'relpos_buckets',
                'mlp', 'vocab'
            },
            msg='unrecognized axis in variable')

  def test_sow_intermediates(self):
    """Tests intermediate tracking using `Module.sow` in the EncoderDecoder."""
    encoder_input_tokens = np.zeros((16, 8), dtype=np.float32)
    decoder_input_tokens = np.zeros((16, 8), dtype=np.float32)
    decoder_target_tokens = np.zeros((16, 8), dtype=np.float32)

    transformer = t5_test_utils.make_config3_shared_token_embedder()
    variables = transformer.init(
        random.PRNGKey(0),
        encoder_input_tokens=encoder_input_tokens,
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        enable_dropout=False,
    )

    _, modified_variables = transformer.apply(
        {'params': variables['params']},
        encoder_input_tokens=encoder_input_tokens,
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        enable_dropout=False,
        mutable='intermediates',
    )
    # Note: the 'intermediates' collection must be set to mutable in order to
    # get the tracked values back in `modified_variables`.

    # Check the shape of tracked intermediates.
    intermediates = modified_variables['intermediates']

    encoder_input_tokens = intermediates['encoder']['input_tokens_ids']
    self.assertLen(encoder_input_tokens, 1)
    self.assertEqual(encoder_input_tokens[0].shape, (16, 8))

    final_encoder_outputs = intermediates['encoder']['final_encoder_outputs']
    self.assertLen(final_encoder_outputs, 1)
    self.assertEqual(final_encoder_outputs[0].shape, (16, 8, 13))

    pre_logits = intermediates['decoder']['pre_logits_layer']
    self.assertLen(pre_logits, 1)
    self.assertEqual(pre_logits[0].shape, (16, 8, 13))

    logits = intermediates['decoder']['logits']
    self.assertLen(logits, 1)
    self.assertEqual(logits[0].shape, (16, 8, 71))

    encoder_embedded_inputs = intermediates['encoder']['embedder']['output']
    self.assertLen(encoder_embedded_inputs, 1)
    self.assertEqual(encoder_embedded_inputs[0].shape, (16, 8, 13))

    decoder_embedded_inputs = intermediates['decoder']['embedder']['output']
    self.assertLen(decoder_embedded_inputs, 1)
    self.assertEqual(decoder_embedded_inputs[0].shape, (16, 8, 13))

    encoder_num_layers = 3
    decoder_num_layers = 2

    for i in range(encoder_num_layers):
      activations = intermediates['encoder'][f'layers_{i}']['activations']
      self.assertLen(activations, 1)
      self.assertEqual(activations[0].shape, (16, 8, 13))

    for i in range(decoder_num_layers):
      activations = intermediates['decoder'][f'layers_{i}']['activations']
      self.assertLen(activations, 1)
      self.assertEqual(activations[0].shape, (16, 8, 13))

  def test_sow_intermediates_with_scan_model(self):
    """Tests if we obtain intermediates when using scan."""
    rs = np.random.RandomState(0)
    encoder_input_tokens = rs.randint(0, 71, size=(16, 8), dtype=np.int32)
    decoder_input_tokens = rs.randint(0, 71, size=(16, 7), dtype=np.int32)
    decoder_target_tokens = rs.randint(0, 71, size=(16, 7), dtype=np.int32)
    model = t5_test_utils.make_config1(
        scan_layers=True, layer_remat='full', sow_intermediates=True)
    variables = model.init(
        random.PRNGKey(0),
        encoder_input_tokens=encoder_input_tokens,
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        enable_dropout=False,
    )
    _, modified_variables = model.apply(
        {'params': variables['params']},
        encoder_input_tokens=encoder_input_tokens,
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        enable_dropout=False,
        mutable=['intermediates'])
    intermediates = modified_variables['intermediates']
    encoder_layer_outputs = intermediates['encoder']['encoder']['activations']
    # Shape: [batch_size, seq_len, num_layers, hidden_size]
    self.assertEqual(encoder_layer_outputs[0].shape, (16, 8, 3, 13))
    decoder_layer_outputs = intermediates['decoder']['decoder']['activations']
    # Shape: [batch_size, seq_len, num_layers, hidden_size]
    self.assertEqual(decoder_layer_outputs[0].shape, (16, 7, 2, 13))

  def test_capture_input_gradients(self):
    """Tests that the input grads are captured."""
    rs = np.random.RandomState(0)  # Need nonzero inputs to get nonzero grads.
    encoder_input_tokens = rs.randint(0, 71, size=(16, 8), dtype=np.int32)
    decoder_input_tokens = rs.randint(0, 71, size=(16, 8), dtype=np.int32)
    decoder_target_tokens = rs.randint(0, 71, size=(16, 8), dtype=np.int32)

    transformer = t5_test_utils.make_config3_shared_token_embedder()
    variables = transformer.init(
        random.PRNGKey(0),
        encoder_input_tokens=encoder_input_tokens,
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        enable_dropout=False,
    )

    # On initialization there should be empty grads.
    self.assertContainsSubset(('grads',), variables)

    def fake_loss(variables, encoder_input_tokens, decoder_input_tokens,
                  decoder_target_tokens):
      """Returns a loss."""
      output, _ = transformer.apply(
          variables,
          encoder_input_tokens=encoder_input_tokens,
          decoder_input_tokens=decoder_input_tokens,
          decoder_target_tokens=decoder_target_tokens,
          enable_dropout=False,
          mutable=['grads'])  # Needed to enable gradient capture.
      return output.sum()

    grad_fn = jax.grad(fake_loss)
    grads_variables = grad_fn(variables, encoder_input_tokens,
                              decoder_input_tokens, decoder_target_tokens)
    grads = grads_variables['grads']

    encoder_embedder_grad = grads['encoder']['embedder']['output_grad']
    self.assertEqual(encoder_embedder_grad.shape, (16, 8, 13))
    self.assertNotAlmostEqual(encoder_embedder_grad.sum(), 0.0)

    decoder_embedder_grad = grads['decoder']['embedder']['output_grad']
    self.assertEqual(decoder_embedder_grad.shape, (16, 8, 13))
    self.assertNotAlmostEqual(decoder_embedder_grad.sum(), 0.0)


class DecoderOnlyTest(absltest.TestCase):

  def test_decoder_shapes_per_layer_relpos_bias(self):
    """Tests if the decoder parameter have the expected shapes."""
    decoder = t5_test_utils.test_make_decoder_only1()
    inputs = np.array(
        [
            # Batch 1.
            [183, 20, 75],
            # Batch 2.
            [392, 19, 7],
        ],
        dtype=np.int32)
    output, variables = decoder.init_with_output(
        random.PRNGKey(0),
        decoder_input_tokens=inputs,
        decoder_target_tokens=inputs,  # used for mask generation
        enable_dropout=False,
    )
    params = variables['params']
    reformatted = decoder.apply({}, params, method=decoder.to_save_format)
    check_params(reformatted, 'decoder_shapes_per_layer_relpos_bias.json')
    self.assertEqual(output.shape, (2, 3, 4))

    # Convert back to Flax module structure format and test again.
    params2 = decoder.apply({}, reformatted, method=decoder.from_save_format)
    output2 = decoder.apply(
        {'params': params2},
        decoder_input_tokens=inputs,
        decoder_target_tokens=inputs,  # used for mask generation
        enable_dropout=False,
    )
    np.testing.assert_allclose(output, output2, rtol=1e-8)

  def test_decoder_shapes_fused_parallel(self):
    """Tests if the decoder parameter have the expected shapes."""
    decoder = t5_test_utils.make_parallel_fused_transformer_config()
    inputs = np.array(
        [
            # Batch 1.
            [183, 20, 75],
            # Batch 2.
            [392, 19, 7],
        ],
        dtype=np.int32,
    )
    output, variables = decoder.init_with_output(
        random.PRNGKey(0),
        decoder_input_tokens=inputs,
        decoder_target_tokens=inputs,  # used for mask generation
        enable_dropout=False,
    )
    params = variables['params']
    reformatted = decoder.apply({}, params, method=decoder.to_save_format)
    check_params(reformatted, 'decoder_shapes_fused_parallel.json')
    self.assertEqual(output.shape, (2, 3, 4))

    # Convert back to Flax module structure format and test again.
    params2 = decoder.apply({}, reformatted, method=decoder.from_save_format)
    output2 = decoder.apply(
        {'params': params2},
        decoder_input_tokens=inputs,
        decoder_target_tokens=inputs,  # used for mask generation
        enable_dropout=False,
    )
    output = output.astype(np.float32)
    output2 = output2.astype(np.float32)
    np.testing.assert_allclose(output, output2, rtol=1e-8)

  def test_decoder_shapes_explicit_attention_map(self):
    """Tests if the decoder parameter have the expected shapes."""
    decoder = t5_test_utils.make_parallel_fused_transformer_config()
    inputs = np.array(
        [
            # Batch 1.
            [183, 20, 75],
            # Batch 2.
            [392, 19, 7],
        ],
        dtype=np.int32,
    )
    output, variables = decoder.init_with_output(
        random.PRNGKey(0),
        decoder_input_tokens=inputs,
        decoder_target_tokens=None,  # not needed if attention mask is provided.
        # By specifying the attention mask explicitly we can mix e.g., prefix
        # LM with bidirectional LM, as done below.
        decoder_attention_mask=[
            [[
                [1, 1, 0],
                [1, 1, 0],
                [1, 1, 1],
            ]],
            [[
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]],
        ],
        enable_dropout=False,
    )
    params = variables['params']
    reformatted = decoder.apply({}, params, method=decoder.to_save_format)
    check_params(reformatted, 'decoder_shapes_fused_parallel.json')
    self.assertEqual(output.shape, (2, 3, 4))



if __name__ == '__main__':
  absltest.main()
