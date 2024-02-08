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

"""Tests for Mixture of Experts gin configs."""

import os

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
import gin
from jax import numpy as jnp
from jax import random
import numpy as np
from t5x import models as t5x_models
# "Unused" imports below are needed by gin configs.
# pylint: disable=unused-import
from t5x import utils
from t5x.contrib.moe import adafactor_utils
from t5x.contrib.moe import models
# pylint: enable=unused-import


class GinConfigsTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(GinConfigsTest, cls).setUpClass()
    cls.root = os.path.join(
        absltest.get_default_test_srcdir(),
        'flaxformer/t5x/configs/moe')
    gin.add_config_file_search_path(cls.root)

  def setUp(self):
    super().setUp()
    gin.clear_config()

  @parameterized.parameters(
      'experts_choose_small.gin',
      'experts_choose_tiny.gin',
      'tokens_choose_small.gin',
      'tokens_choose_tiny.gin',
  )
  def test_encoder_decoder_model_gin_config(self, filename):
    path = os.path.join(self.root, 'models', filename)
    gin.parse_config_file(path)
    gin.parse_config("""
      NUM_EXPERTS = 2
      NUM_MODEL_PARTITIONS = 1
      """)
    gin.finalize()  # Check for required values, etc.

    model_config_ref: gin.ConfigurableReference = gin.query_parameter('%MODEL')

    # Instantiate T5X model (`t5x.contrib.moe.models.MoeEncoderDecoderModel`).
    model: t5x_models.BaseModel = model_config_ref.scoped_configurable_fn()

    encoder_input_tokens = jnp.ones((2, 4))
    # For this test, decoder input and target tokens are fake values.
    decoder_input_tokens = jnp.array([[1, 2, 1, 0], [0, 1, 0, 2]])
    decoder_target_tokens = jnp.array([[1, 2, 1, 0], [0, 1, 0, 2]])
    decoder_loss_weights = jnp.array([[1, 1, 1, 0], [0, 1, 0, 1]])

    encoder_kwargs = {'encoder_input_tokens': encoder_input_tokens}

    variables = model.module.init(
        random.PRNGKey(0),
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        enable_dropout=False,
        **encoder_kwargs)

    _ = model.module.apply({'params': variables['params']},
                           decoder_input_tokens=decoder_input_tokens,
                           decoder_target_tokens=decoder_target_tokens,
                           enable_dropout=False,
                           **encoder_kwargs)

    batch = {
        'encoder_input_tokens': encoder_input_tokens,
        'decoder_input_tokens': decoder_input_tokens,
        'decoder_target_tokens': decoder_target_tokens,
        'decoder_loss_weights': decoder_loss_weights
    }
    _ = model.score_batch(variables['params'], batch)


  def test_architecture_gin_config(self):
    path = os.path.join(self.root, 'architectures', 'moe.gin')
    gin.parse_config_file(path)
    gin.parse_config("""
        NUM_HEADS = 2
        NUM_ENCODER_LAYERS = 2
        NUM_DECODER_LAYERS = 2
        NUM_ENCODER_SPARSE_LAYERS = 1
        NUM_DECODER_SPARSE_LAYERS = 1
        HEAD_DIM = 4
        EMBED_DIM = 8
        MLP_DIM = 8
        NUM_EMBEDDINGS = 128
        NUM_EXPERTS = 2
        NUM_MODEL_PARTITIONS = 1
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

    encoder_kwargs = {'encoder_input_tokens': encoder_input_tokens}

    _, variables = arch.init_with_output(
        random.PRNGKey(0),
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        enable_dropout=False,
        decode=False,
        max_decode_length=None,
        **encoder_kwargs)

    # Call with expected arrays (e.g. Call `__call__` with concrete sequences).
    _ = arch.apply(
        variables,
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        **encoder_kwargs)


if __name__ == '__main__':
  absltest.main()
