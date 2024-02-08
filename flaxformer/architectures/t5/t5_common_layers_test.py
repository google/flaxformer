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

"""Tests for t5_layers."""

from absl.testing import absltest
from flax import linen as nn
from jax import numpy as jnp
from jax import random
import numpy as np

from flaxformer.architectures.t5 import t5_common_layers
from flaxformer.components import embedding

EMBEDDING_DIM = 7
MLP_DIM = 32
NUM_HEADS = 2
NUM_LAYERS = 3
ACTIVATIONS = ('gelu',)
DROPOUT_RATE = 0.14
HEAD_DIM = 4


class T5BaseTest(absltest.TestCase):

  def test_encoder_layer(self):
    layer = t5_common_layers.encoder_layer(
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        mlp_dim=MLP_DIM,
        activations=ACTIVATIONS,
        dropout_rate=DROPOUT_RATE)
    inputs = np.array(
        [
            # Batch 1.
            [[101, 183, 20, 75, 10]],
            # Batch 2.
            [[101, 392, 19, 7, 20]],
        ],
        dtype=np.int32)
    _, variables = layer.init_with_output(
        random.PRNGKey(0),
        inputs,
        enable_dropout=False,
    )

    input_inner_dim = 5

    # Validate that the QKV dims are being set appropriately.
    attention_params = variables['params']['attention']
    expected_qkv_shape = [input_inner_dim, HEAD_DIM * NUM_HEADS]
    np.testing.assert_equal(expected_qkv_shape,
                            np.shape(attention_params['query']['kernel']))
    np.testing.assert_equal(expected_qkv_shape,
                            np.shape(attention_params['key']['kernel']))
    np.testing.assert_equal(expected_qkv_shape,
                            np.shape(attention_params['value']['kernel']))

    # Validate that the MLP dims are being set appropriately.
    mlp_params = variables['params']['mlp']
    np.testing.assert_equal([input_inner_dim, MLP_DIM],
                            np.shape(mlp_params['wi']['kernel']))
    np.testing.assert_equal([MLP_DIM, input_inner_dim],
                            np.shape(mlp_params['wo']['kernel']))

    # Validate that the activations are being set.
    self.assertEqual(ACTIVATIONS, layer.mlp.activations)

    # Validate the dropout rate is being respected.
    self.assertEqual(DROPOUT_RATE, layer.attention.dropout_rate)
    self.assertEqual(DROPOUT_RATE, layer.mlp.intermediate_dropout_rate)
    self.assertEqual(0.0, layer.mlp.final_dropout_rate)

  def test_decoder_layer(self):
    layer = t5_common_layers.decoder_layer(
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        mlp_dim=MLP_DIM,
        activations=ACTIVATIONS,
        dropout_rate=DROPOUT_RATE)
    targets = np.array(
        # Batch 1.
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
            # Batch 2.
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
        ],
        dtype=np.float32)
    encoded = np.array(
        # Batch 1.
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            # Batch 2.
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        ],
        dtype=np.float32)

    _, variables = layer.init_with_output(
        random.PRNGKey(0),
        targets,
        enable_dropout=False,
        encoded=encoded,
    )
    input_inner_dim = 2
    # Validate that the QKV dims are being set appropriately.
    expected_qkv_shape = [input_inner_dim, HEAD_DIM * NUM_HEADS]
    attention_params = variables['params']['self_attention']
    np.testing.assert_equal(expected_qkv_shape,
                            np.shape(attention_params['query']['kernel']))
    np.testing.assert_equal(expected_qkv_shape,
                            np.shape(attention_params['key']['kernel']))
    np.testing.assert_equal(expected_qkv_shape,
                            np.shape(attention_params['value']['kernel']))

    attention_params = variables['params']['encoder_decoder_attention']
    np.testing.assert_equal(expected_qkv_shape,
                            np.shape(attention_params['query']['kernel']))
    np.testing.assert_equal(expected_qkv_shape,
                            np.shape(attention_params['key']['kernel']))
    np.testing.assert_equal(expected_qkv_shape,
                            np.shape(attention_params['value']['kernel']))

    # Validate that the MLP dims are being set appropriately.
    mlp_params = variables['params']['mlp']
    np.testing.assert_equal([input_inner_dim, MLP_DIM],
                            np.shape(mlp_params['wi']['kernel']))
    np.testing.assert_equal([MLP_DIM, input_inner_dim],
                            np.shape(mlp_params['wo']['kernel']))

    # Validate that the activations are being set.
    self.assertEqual(ACTIVATIONS, layer.mlp.activations)

    # Validate the dropout rate is being respected.
    self.assertEqual(DROPOUT_RATE, layer.self_attention.dropout_rate)
    self.assertEqual(DROPOUT_RATE, layer.mlp.intermediate_dropout_rate)
    self.assertEqual(0.0, layer.mlp.final_dropout_rate)

  def test_encoder(self):
    shared_embedder = embedding.Embed(
        num_embeddings=5,
        features=EMBEDDING_DIM,
        cast_input_dtype=jnp.int32,
        dtype=jnp.float32,
        attend_dtype=jnp.float32,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        name='token_embedder')
    layer = t5_common_layers.encoder(
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        mlp_dim=MLP_DIM,
        num_layers=NUM_LAYERS,
        shared_token_embedder=shared_embedder,
        activations=ACTIVATIONS,
        dropout_rate=DROPOUT_RATE)
    inputs = np.array(
        [
            # Batch 1.
            [101, 183, 20, 75],
            # Batch 2.
            [101, 392, 19, 7],
        ],
        dtype=np.int32)

    _, variables = layer.init_with_output(
        random.PRNGKey(0),
        inputs,
        enable_dropout=False,
    )

    # Validate that there are 3 encoder layers.
    self.assertContainsSubset(['layers_0', 'layers_1', 'layers_2'],
                              list(variables['params'].keys()))

    # Validate that the QKV dims are being passed appropriately.
    attention_params = variables['params']['layers_2']['attention']
    expected_qkv_shape = [EMBEDDING_DIM, HEAD_DIM * NUM_HEADS]
    np.testing.assert_equal(expected_qkv_shape,
                            np.shape(attention_params['query']['kernel']))
    np.testing.assert_equal(expected_qkv_shape,
                            np.shape(attention_params['key']['kernel']))
    np.testing.assert_equal(expected_qkv_shape,
                            np.shape(attention_params['value']['kernel']))

    # Validate that the MLP dims are being passed appropriately.
    mlp_params = variables['params']['layers_2']['mlp']
    np.testing.assert_equal([EMBEDDING_DIM, MLP_DIM],
                            np.shape(mlp_params['wi']['kernel']))
    np.testing.assert_equal([MLP_DIM, EMBEDDING_DIM],
                            np.shape(mlp_params['wo']['kernel']))

  def test_decoder(self):
    shared_embedder = embedding.Embed(
        num_embeddings=5,
        features=EMBEDDING_DIM,
        cast_input_dtype=jnp.int32,
        dtype=jnp.float32,
        attend_dtype=jnp.float32,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        name='token_embedder')
    layer = t5_common_layers.decoder(
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        mlp_dim=MLP_DIM,
        num_layers=NUM_LAYERS,
        shared_token_embedder=shared_embedder,
        activations=('relu',),
        dropout_rate=0.1)
    decoder_input_tokens = np.array(
        [
            # Batch 1.
            [101, 183, 20, 75, 10],
            # Batch 2.
            [101, 392, 19, 7, 20],
        ],
        dtype=np.int32)
    encoder_outputs = np.array(
        # Batch 1.
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
            # Batch 2.
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
        ],
        dtype=np.float32)
    _, variables = layer.init_with_output(
        random.PRNGKey(0),
        encoder_outputs,
        decoder_input_tokens,
        enable_dropout=False,
    )

    # Validate that there are 3 encoder layers.
    self.assertContainsSubset(['layers_0', 'layers_1', 'layers_2'],
                              list(variables['params'].keys()))

    # Validate that the QKV dims are being passed appropriately.
    expected_qkv_shape = [EMBEDDING_DIM, HEAD_DIM * NUM_HEADS]
    attention_params = variables['params']['layers_2']['self_attention']
    np.testing.assert_equal(expected_qkv_shape,
                            np.shape(attention_params['query']['kernel']))
    np.testing.assert_equal(expected_qkv_shape,
                            np.shape(attention_params['key']['kernel']))
    np.testing.assert_equal(expected_qkv_shape,
                            np.shape(attention_params['value']['kernel']))

    encoder_inner_dim = 2
    expected_encoder_kv_shape = [encoder_inner_dim, HEAD_DIM * NUM_HEADS]
    attention_params = variables['params']['layers_2'][
        'encoder_decoder_attention']
    np.testing.assert_equal(expected_qkv_shape,
                            np.shape(attention_params['query']['kernel']))
    np.testing.assert_equal(expected_encoder_kv_shape,
                            np.shape(attention_params['key']['kernel']))
    np.testing.assert_equal(expected_encoder_kv_shape,
                            np.shape(attention_params['value']['kernel']))

    # Validate that the MLP dims are being passed appropriately.
    mlp_params = variables['params']['layers_2']['mlp']
    np.testing.assert_equal([EMBEDDING_DIM, MLP_DIM],
                            np.shape(mlp_params['wi']['kernel']))
    np.testing.assert_equal([MLP_DIM, EMBEDDING_DIM],
                            np.shape(mlp_params['wo']['kernel']))


if __name__ == '__main__':
  absltest.main()
