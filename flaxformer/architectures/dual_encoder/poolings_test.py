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

"""Tests for pooling layers."""
from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
from jax import random
import numpy as np
from flaxformer.architectures.dual_encoder import poolings
from flaxformer.components import layer_norm


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
      ('multihead_attention_pooling',
       poolings.MultiHeadAttentionPooling(
           num_heads=2, head_dim=2, layer_norm_factory=layer_norm.T5LayerNorm)))
  def test_poolings(self, pooler):
    """Test if the pooling layers have correct shapes and types."""
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng, 2)
    encoded_inputs = random.normal(
        key1, (self.batch_size, self.seq_len, self.hidden_size))

    rngs = {'params': rng, 'dropout': key2}
    encodings, _ = pooler.init_with_output(rngs, encoded_inputs,
                                           self.input_masks)
    self.assertEqual(encodings.shape, (self.batch_size, self.hidden_size))
    self.assertEqual(encodings.dtype, jnp.float32)


if __name__ == '__main__':
  absltest.main()
