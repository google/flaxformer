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

"""Tests for h_transformer_1d_architecture."""

from absl.testing import absltest
from absl.testing import parameterized
from jax import random
import numpy as np

from flaxformer import testing_utils
from flaxformer.architectures.h_transformer import h_transformer_1d_architecture_test_utils as h_transformer_test_utils
from flaxformer.architectures.h_transformer import h_transformer_utils as utils

testdata_dir = 'flaxformer/architectures/h_transformer/testdata'
expected_files = testing_utils.ExpectedJsonFiles(testdata_dir)
check_params = expected_files.check_params_shapes_only


class EncoderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.inputs = np.array([
        [101, 183, 20, 75],
        [101, 392, 19, 7],
    ],
                           dtype=np.int32)
    self.embed_size = 13
    batch, seq_len = self.inputs.shape
    self.expected_output_shape = (batch, seq_len, self.embed_size)
    self.rng_key = random.PRNGKey(0)

  @parameterized.named_parameters(
      dict(testcase_name='scan', scan_layers=True),
      dict(testcase_name='no_scan', scan_layers=False),
  )
  def test_encoder_run(self, scan_layers):
    encoder = h_transformer_test_utils.config_encoder(
        embed_size=self.embed_size, scan_layers=scan_layers)
    output, _ = encoder.init_with_output(
        self.rng_key,
        self.inputs,
        enable_dropout=False,
    )
    self.assertEqual(output.shape, self.expected_output_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name='scan',
          scan_layers=True,
          layer_remat_options=[
              utils.LayerRematOptions.MINIMAL, utils.LayerRematOptions.FULL
          ]),
      dict(
          testcase_name='no_scan',
          scan_layers=False,
          layer_remat_options=[
              utils.LayerRematOptions.NONE, utils.LayerRematOptions.MINIMAL,
              utils.LayerRematOptions.FULL
          ]),
  )
  def test_scan_and_remat(self, scan_layers, layer_remat_options):
    """Tests if encoder returns the same output for different scan/remat."""
    outputs = []
    for layer_remat in layer_remat_options:
      encoder = h_transformer_test_utils.config_encoder(
          embed_size=self.embed_size,
          scan_layers=scan_layers,
          layer_remat=layer_remat)
      output, _ = encoder.init_with_output(
          self.rng_key,
          self.inputs,
          enable_dropout=False,
      )
      outputs.append(output)

    for other_output in outputs[1:]:
      np.testing.assert_allclose(outputs[0], other_output, rtol=1.5e-5)

  def test_encoder_shapes_per_layer(self):
    encoder = h_transformer_test_utils.config_encoder()
    output1, variables = encoder.init_with_output(
        self.rng_key,
        self.inputs,
        enable_dropout=False,
    )

    reformatted = encoder.apply({},
                                variables['params'],
                                method=encoder.to_save_format)
    check_params(reformatted, 'encoder_shapes_per_layer.json')
    self.assertEqual(output1.shape, (2, 4, 13))

    # Convert back to Flax module structure format and test again.
    params2 = encoder.apply({}, reformatted, method=encoder.from_save_format)
    output2 = encoder.apply(
        {'params': params2},
        self.inputs,
        enable_dropout=False,
    )
    np.testing.assert_allclose(output1, output2, rtol=1e-8)


class DecoderOnlyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.inputs = np.array([
        [101, 183, 20, 75, 76, 78, 91, 102, 122, 187, 23, 76, 76, 87, 94, 121],
        [101, 392, 19, 7, 76, 78, 91, 102, 122, 187, 23, 76, 76, 87, 94, 121],
    ],
                           dtype=np.int32)
    self.embed_size = 13
    self.vocab_size = 2000
    (batch, seq_len) = self.inputs.shape
    self.expected_output_shape = (batch, seq_len, self.vocab_size)

  @parameterized.named_parameters(
      dict(testcase_name='scan', scan_layers=True),
      dict(testcase_name='no_scan', scan_layers=False),
  )
  def test_decoder_run(self, scan_layers):
    decoder = h_transformer_test_utils.config_decoder_only(
        embed_size=self.embed_size,
        vocab_size=self.vocab_size,
        scan_layers=scan_layers)
    output, _ = decoder.init_with_output(
        random.PRNGKey(0), self.inputs, enable_dropout=False)
    self.assertEqual(output.shape, self.expected_output_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name='scan',
          scan_layers=True,
          layer_remat_options=[
              utils.LayerRematOptions.MINIMAL, utils.LayerRematOptions.FULL
          ]),
      dict(
          testcase_name='no_scan',
          scan_layers=False,
          layer_remat_options=[
              utils.LayerRematOptions.NONE, utils.LayerRematOptions.MINIMAL,
              utils.LayerRematOptions.FULL
          ]),
  )
  def test_scan_and_remat(self, scan_layers, layer_remat_options):
    """Tests if decoder returns the same output for different scan/remat."""
    outputs = []
    for layer_remat in layer_remat_options:
      decoder = h_transformer_test_utils.config_decoder_only(
          embed_size=self.embed_size,
          vocab_size=self.vocab_size,
          scan_layers=scan_layers,
          layer_remat=layer_remat)
      output, _ = decoder.init_with_output(
          random.PRNGKey(0), self.inputs, enable_dropout=False)
      outputs.append(output)

    for other_output in outputs[1:]:
      np.testing.assert_allclose(
          outputs[0], other_output, atol=1e-5, rtol=1.5e-5)

  def test_decoder_shapes_per_layer(self):
    decoder = h_transformer_test_utils.config_decoder_only(
        embed_size=self.embed_size, vocab_size=self.vocab_size)
    output1, variables = decoder.init_with_output(
        random.PRNGKey(0),
        self.inputs,
        enable_dropout=False,
    )

    reformatted = decoder.apply({},
                                variables['params'],
                                method=decoder.to_save_format)
    with self.subTest(name='check_params_and_output_shape'):
      check_params(reformatted, 'decoder_only_shapes_per_layer.json')
      self.assertEqual(output1.shape, self.expected_output_shape)

    # Convert back to Flax module structure format and test again.
    params2 = decoder.apply({}, reformatted, method=decoder.from_save_format)
    output2 = decoder.apply(
        {'params': params2},
        self.inputs,
        enable_dropout=False,
    )
    with self.subTest(name='check_flax_module_outputs'):
      np.testing.assert_allclose(output1, output2, rtol=1e-8)


class DecoderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.inputs = np.array([
        [101, 183, 20, 75],
        [101, 392, 19, 7],
    ],
                           dtype=np.int32)
    self.embed_size = 13
    self.vocab_size = 2000
    (batch, seq_len) = self.inputs.shape
    self.expected_output_shape = (batch, seq_len, self.vocab_size)

  @parameterized.named_parameters(
      dict(testcase_name='scan_no_parallel', scan_layers=True, parallel=False),
      dict(
          testcase_name='no_scan_no_parallel',
          scan_layers=False,
          parallel=False),
      dict(testcase_name='scan_parallel', scan_layers=True, parallel=True),
      dict(testcase_name='no_scan_parallel', scan_layers=False, parallel=True),
  )
  def test_decoder_run(self, scan_layers, parallel):
    decoder = h_transformer_test_utils.config_decoder(
        embed_size=self.embed_size,
        vocab_size=self.vocab_size,
        scan_layers=scan_layers,
        parallel=parallel)
    output, _ = decoder.init_with_output(
        random.PRNGKey(0), self.inputs, enable_dropout=False)
    self.assertEqual(output.shape, self.expected_output_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name='scan',
          scan_layers=True,
          layer_remat_options=[
              utils.LayerRematOptions.MINIMAL, utils.LayerRematOptions.FULL
          ]),
      dict(
          testcase_name='no_scan',
          scan_layers=False,
          layer_remat_options=[
              utils.LayerRematOptions.NONE, utils.LayerRematOptions.MINIMAL,
              utils.LayerRematOptions.FULL
          ]),
  )
  def test_scan_and_remat(self, scan_layers, layer_remat_options):
    """Tests if decoder returns the same output for different scan/remat."""
    outputs = []
    for layer_remat in layer_remat_options:
      decoder = h_transformer_test_utils.config_decoder(
          embed_size=self.embed_size,
          vocab_size=self.vocab_size,
          scan_layers=scan_layers,
          layer_remat=layer_remat)
      output, _ = decoder.init_with_output(
          random.PRNGKey(0), self.inputs, enable_dropout=False)
      outputs.append(output)

    for other_output in outputs[1:]:
      np.testing.assert_allclose(
          outputs[0], other_output, atol=1e-5, rtol=1.5e-5)

  def test_decoder_shapes_per_layer(self):
    decoder = h_transformer_test_utils.config_decoder(
        embed_size=self.embed_size, vocab_size=self.vocab_size)
    output1, variables = decoder.init_with_output(
        random.PRNGKey(0),
        self.inputs,
        enable_dropout=False,
    )

    reformatted = decoder.apply({},
                                variables['params'],
                                method=decoder.to_save_format)
    with self.subTest(name='check_params_and_output_shape'):
      check_params(reformatted, 'decoder_shapes_per_layer.json')
      self.assertEqual(output1.shape, self.expected_output_shape)

    # Convert back to Flax module structure format and test again.
    params2 = decoder.apply({}, reformatted, method=decoder.from_save_format)
    output2 = decoder.apply(
        {'params': params2},
        self.inputs,
        enable_dropout=False,
    )
    with self.subTest(name='check_flax_module_outputs'):
      np.testing.assert_allclose(output1, output2, rtol=1e-8)


class EncoderDecoderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    inputs = np.array([
        [101, 183, 20, 75],
        [101, 392, 19, 7],
    ], dtype=np.int32)
    self.encoder_input_tokens = inputs
    self.decoder_input_tokens = inputs
    self.decoder_target_tokens = inputs
    self.embed_size = 13
    self.vocab_size = 2000
    (batch, seq_len) = inputs.shape
    self.expected_output_shape = (batch, seq_len, self.vocab_size)

  @parameterized.named_parameters(
      dict(testcase_name='scan', scan_layers=True),
      dict(testcase_name='no_scan', scan_layers=False),
  )
  def test_encoder_decoder_run(self, scan_layers):
    encoder_decoder = h_transformer_test_utils.config_encoder_decoder(
        embed_size=self.embed_size,
        vocab_size=self.vocab_size,
        scan_layers=scan_layers)
    output, _ = encoder_decoder.init_with_output(
        random.PRNGKey(0),
        encoder_input_tokens=self.encoder_input_tokens,
        decoder_input_tokens=self.decoder_input_tokens,
        decoder_target_tokens=self.decoder_target_tokens,
        enable_dropout=False,
    )
    self.assertEqual(output.shape, self.expected_output_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name='scan',
          scan_layers=True,
          layer_remat_options=[
              utils.LayerRematOptions.MINIMAL, utils.LayerRematOptions.FULL
          ]),
      dict(
          testcase_name='no_scan',
          scan_layers=False,
          layer_remat_options=[
              utils.LayerRematOptions.NONE, utils.LayerRematOptions.MINIMAL,
              utils.LayerRematOptions.FULL
          ]),
  )
  def test_scan_and_remat(self, scan_layers, layer_remat_options):
    """Tests if encoder_decoder returns the same output for different scan/remat."""
    outputs = []
    for layer_remat in layer_remat_options:
      encoder_decoder = h_transformer_test_utils.config_encoder_decoder(
          embed_size=self.embed_size,
          vocab_size=self.vocab_size,
          scan_layers=scan_layers,
          layer_remat=layer_remat)
      output, _ = encoder_decoder.init_with_output(
          random.PRNGKey(0),
          encoder_input_tokens=self.encoder_input_tokens,
          decoder_input_tokens=self.decoder_input_tokens,
          decoder_target_tokens=self.decoder_target_tokens,
          enable_dropout=False,
      )
      outputs.append(output)

    for other_output in outputs[1:]:
      np.testing.assert_allclose(
          outputs[0], other_output, atol=1e-5, rtol=1.5e-5)

  def test_encoder_decoder_shapes_per_layer(self):
    encoder_decoder = h_transformer_test_utils.config_encoder_decoder(
        embed_size=self.embed_size, vocab_size=self.vocab_size)
    output1, variables = encoder_decoder.init_with_output(
        random.PRNGKey(0),
        encoder_input_tokens=self.encoder_input_tokens,
        decoder_input_tokens=self.decoder_input_tokens,
        decoder_target_tokens=self.decoder_target_tokens,
        enable_dropout=False,
    )

    reformatted = encoder_decoder.apply({},
                                        variables['params'],
                                        method=encoder_decoder.to_save_format)
    with self.subTest(name='check_params_and_output_shape'):
      check_params(reformatted, 'encoder_decoder_shapes_per_layer.json')
      self.assertEqual(output1.shape, self.expected_output_shape)

    # Convert back to Flax module structure format and test again.
    params2 = encoder_decoder.apply({},
                                    reformatted,
                                    method=encoder_decoder.from_save_format)
    output2 = encoder_decoder.apply(
        {'params': params2},
        encoder_input_tokens=self.encoder_input_tokens,
        decoder_input_tokens=self.decoder_input_tokens,
        decoder_target_tokens=self.decoder_target_tokens,
        enable_dropout=False,
    )
    with self.subTest(name='check_flax_module_outputs'):
      np.testing.assert_allclose(output1, output2, rtol=1e-8)


if __name__ == '__main__':
  absltest.main()
