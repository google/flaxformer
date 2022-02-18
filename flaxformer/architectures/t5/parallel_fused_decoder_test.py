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

"""Tests for t5_architecture."""

from absl.testing import absltest
from jax import random
import numpy as np

from flaxformer import testing_utils
from flaxformer.architectures.t5 import t5_architecture_test_utils as t5_test_utils

expected_files = testing_utils.ExpectedJsonFiles(
    'flaxformer/architectures/t5/testdata')
check_params = expected_files.check_params_shapes_only


class ParallelFusedDecoderOnlyTest(absltest.TestCase):

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


if __name__ == '__main__':
  absltest.main()
