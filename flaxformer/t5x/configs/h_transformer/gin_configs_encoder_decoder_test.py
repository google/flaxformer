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

"""Tests for encoder_decoder gin configs in this directory."""

# "Unused" imports below are needed by gin configs.
# pylint: disable=unused-import

import os

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
import gin
from jax import numpy as jnp
from jax import random
import numpy as np
from t5x import models as t5x_models


class GinConfigsTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(GinConfigsTest, cls).setUpClass()
    cls.root = os.path.join(
        absltest.get_default_test_srcdir(),
        'flaxformer/t5x/configs/h_transformer')
    gin.add_config_file_search_path(cls.root)

  def setUp(self):
    super().setUp()
    gin.clear_config()

  @parameterized.named_parameters(
      dict(
          testcase_name='1d_encoder_decoder_base',
          filename='h_transformer_1d_encoder_decoder_base.gin'),
      dict(
          testcase_name='1d_encoder_decoder_small',
          filename='h_transformer_1d_encoder_decoder_small.gin'),
      dict(
          testcase_name='1d_encoder_decoder_large',
          filename='h_transformer_1d_encoder_decoder_large.gin'),
  )
  def test_model_gin_config(self, filename):
    path = os.path.join(self.root, 'models', filename)
    gin.parse_config_file(path)
    gin.finalize()  # Check for required values, etc.

    model_config_ref: gin.ConfigurableReference = gin.query_parameter('%MODEL')

    # Instantiate T5X model (e.g. `t5x.models.EncoderDecoderModel`).
    model: t5x_models.BaseModel = model_config_ref.scoped_configurable_fn()

    encoder_input_tokens = jnp.array([[1, 2, 1, 0], [1, 3, 0, 0]])
    decoder_input_tokens = jnp.array([[1, 2, 0, 0], [4, 5, 0, 0]])
    decoder_target_tokens = jnp.array([[1, 2, 0, 0], [4, 5, 0, 0]])
    decoder_loss_weights = jnp.array([[1, 1, 0, 0], [1, 1, 0, 0]])

    variables = model.module.init(
        random.PRNGKey(0),
        encoder_input_tokens=encoder_input_tokens,
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        enable_dropout=False)

    output = model.module.apply({'params': variables['params']},
                                encoder_input_tokens=encoder_input_tokens,
                                decoder_input_tokens=decoder_input_tokens,
                                decoder_target_tokens=decoder_target_tokens,
                                enable_dropout=False)
    del output  # Unused.

    batch = {
        'encoder_input_tokens': encoder_input_tokens,
        'decoder_input_tokens': decoder_input_tokens,
        'decoder_target_tokens': decoder_target_tokens,
        'decoder_loss_weights': decoder_loss_weights
    }
    res = model.score_batch(variables['params'], batch)
    del res  # Unused.

  def test_architecture_gin_config(self):
    filename = 'h_transformer_1d_encoder_decoder.gin'
    path = os.path.join(self.root, 'architectures', filename)
    gin.parse_config_file(path)
    gin.parse_config("""
        NUM_HEADS = 2
        NUM_DECODER_LAYERS = 2
        NUM_ENCODER_LAYERS = 2
        EMBED_DIM = 8
        MLP_DIM = 8
        NUM_EMBEDDINGS = 128
        """)
    gin.finalize()  # Check for required values, etc.

    arch_config_ref: gin.ConfigurableReference = gin.query_parameter(
        '%ARCHITECTURE')

    # Instantiate architecture.
    arch: nn.Module = arch_config_ref.scoped_configurable_fn()

    shape = [4, 8]
    encoder_input_tokens = np.ones(shape, dtype=np.int32)
    decoder_input_tokens = np.ones(shape, dtype=np.int32)
    decoder_target_tokens = np.ones(shape, dtype=np.int32)

    output, variables = arch.init_with_output(
        random.PRNGKey(0),
        encoder_input_tokens=encoder_input_tokens,
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        enable_dropout=False)
    del output  # Unused.

    # Call with expected arrays (e.g. Call `__call__` with concrete sequences).
    _ = arch.apply(
        variables,
        encoder_input_tokens=encoder_input_tokens,
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        enable_dropout=False)


if __name__ == '__main__':
  absltest.main()
