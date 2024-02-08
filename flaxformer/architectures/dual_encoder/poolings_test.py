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

"""Tests for pooling layers."""
from typing import Optional

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen
from jax import numpy as jnp
from jax import random
import numpy as np
from flaxformer.architectures.dual_encoder import poolings
from flaxformer.architectures.t5 import t5_architecture
from flaxformer.components import dense
from flaxformer.components import layer_norm
from flaxformer.components.attention import dense_attention


def _get_layer_factory():
  def _get_layer(shared_relative_position_bias: Optional[linen.Module]):
    attention = dense_attention.MultiHeadDotProductAttention(
        num_heads=2, head_dim=2, use_bias=False, dtype=jnp.float32
    )
    mlp = dense.MlpBlock(
        use_bias=False, intermediate_dim=2, activations=('relu',)
    )
    dropout_factory = lambda: linen.Dropout(0.0)
    layer = t5_architecture.EncoderLayer(
        attention=attention,
        mlp=mlp,
        dropout_factory=dropout_factory,
        layer_norm_factory=layer_norm.T5LayerNorm,
        shared_relative_position_bias=shared_relative_position_bias,
    )
    return layer

  return _get_layer


class PoolingsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.batch_size = 2
    self.seq_len = 3
    self.hidden_size = 4
    self.input_masks = np.ones((self.batch_size, self.seq_len))
    self.encoder_mask = np.ones((self.batch_size, 1, 1, self.seq_len))

  @parameterized.named_parameters(
      ('max_pooling', poolings.MaxPooling()),
      ('mean_pooling', poolings.MeanPooling()),
      ('attention_pooling', poolings.AttentionPooling()),
      (
          'multihead_attention_pooling',
          poolings.MultiHeadAttentionPooling(
              num_heads=2, head_dim=2, layer_norm_factory=layer_norm.T5LayerNorm
          ),
      ),
      (
          'multi_layer_pooling',
          poolings.MultiLayerPooling(
              layer_factory=_get_layer_factory(),
              num_layers=2,
              layer_norm_factory=layer_norm.T5LayerNorm,
          ),
      ),
      (
          'multi_layer_pooling_with_mean_pooling',
          poolings.MultiLayerPooling(
              layer_factory=_get_layer_factory(),
              num_layers=2,
              layer_norm_factory=layer_norm.T5LayerNorm,
              pooler_factory=poolings.MeanPooling,
          ),
      ),
      ('last_token_pooling', poolings.LastTokenPooling()),
  )
  def test_poolings(self, pooler):
    """Test if the pooling layers have correct shapes and types."""
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng, 2)
    encoded_inputs = random.normal(
        key1, (self.batch_size, self.seq_len, self.hidden_size)
    )

    rngs = {'params': rng, 'dropout': key2}
    encodings, _ = pooler.init_with_output(
        rngs, encoded_inputs, self.input_masks
    )
    self.assertEqual(encodings.shape, (self.batch_size, self.hidden_size))
    self.assertEqual(encodings.dtype, jnp.float32)

  def test_last_token_poolings(self):
    rngs = {'params': random.PRNGKey(0)}
    encoded_inputs = jnp.array(
        [
            [[0.2, 0.4], [0.22, 0.42], [0.23, 0.43], [0.24, 0.44]],
            [[0.3, 0.6], [-0.32, 0.62], [0.33, -0.63], [-0.34, -0.64]],
            [[-0.4, 0.8], [0.42, -0.82], [-0.43, 0.83], [0.44, -0.84]],
        ],
        dtype=jnp.float32,
    )
    input_masks = jnp.array(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=jnp.int32,
    )
    encodings, _ = poolings.LastTokenPooling().init_with_output(
        rngs, encoded_inputs, input_masks
    )
    np.testing.assert_array_equal(
        encodings,
        jnp.array(
            [
                [0.23, 0.43],
                [-0.34, -0.64],
                [0.42, -0.82],
            ],
            dtype=jnp.float32,
        ),
    )


if __name__ == '__main__':
  absltest.main()
