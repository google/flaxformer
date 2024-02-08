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

"""Tests for perceiver_ar_architecture."""

from absl.testing import absltest
from jax import random
import numpy as np

from flaxformer import testing_utils
from flaxformer.architectures.perceiver_ar import perceiver_ar_architecture_test_utils as perceiver_ar_test_utils

expected_files = testing_utils.ExpectedJsonFiles(
    'flaxformer/architectures/perceiver_ar/'
    'testdata')
check_params = expected_files.check_params_shapes_only


class DecoderOnlyTest(absltest.TestCase):

  def test_decoder_shapes_per_layer(self):
    """Tests if the decoder parameter have the expected shapes."""
    decoder = perceiver_ar_test_utils.test_make_decoder_only1(
        num_latents=2, parallel=False)
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
    check_params(reformatted, 'decoder_shapes_per_layer.json')
    self.assertEqual(output.shape, (2, 2, 4))

    # Convert back to Flax module structure format and test again.
    params2 = decoder.apply({}, reformatted, method=decoder.from_save_format)
    output2 = decoder.apply(
        {'params': params2},
        decoder_input_tokens=inputs,
        decoder_target_tokens=inputs,  # used for mask generation
        enable_dropout=False,
    )
    np.testing.assert_allclose(output, output2, rtol=1e-8)

  def test_parallel_decoder_shapes_per_layer(self):
    """Tests if the decoder parameter have the expected shapes."""
    decoder = perceiver_ar_test_utils.test_make_decoder_only1(
        num_latents=2, parallel=True)
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
    check_params(reformatted, 'parallel_decoder_shapes_per_layer.json')
    self.assertEqual(output.shape, (2, 2, 4))

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
    decoder = perceiver_ar_test_utils.make_parallel_fused_transformer_config(
        num_latents=2)
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
    check_params(reformatted, 'decoder_shapes_fused_parallel.json')
    self.assertEqual(output.shape, (2, 2, 4))

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


if __name__ == '__main__':
  absltest.main()
