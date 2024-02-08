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

"""Tests for the dual encoder architecture."""

import dataclasses
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from jax import numpy as jnp
from jax import random
import numpy as np

from flaxformer import testing_utils
from flaxformer.architectures.dual_encoder import components
from flaxformer.architectures.dual_encoder import dual_encoder_architecture
from flaxformer.architectures.dual_encoder import l2_norm
from flaxformer.architectures.dual_encoder import poolings
from flaxformer.architectures.dual_encoder import similarity_functions
from flaxformer.architectures.dual_encoder import single_tower_logit_functions
from flaxformer.architectures.t5 import t5_architecture as flaxformer_t5_architecture
from flaxformer.components import dense
from flaxformer.components import embedding
from flaxformer.components import layer_norm
from flaxformer.components import relative_position_biases
from flaxformer.components.attention import dense_attention

expected_files = testing_utils.ExpectedJsonFiles(
    'flaxformer/architectures/dual_encoder/testdata/'
)
check_dual_encoder_params = expected_files.check_params_shapes_only

PROJECTION_DIM = 768
OUTPUT_DIM = 3
EMBEDDING_INIT = nn.initializers.normal(stddev=1.0)
RELPOS_BIAS_INIT = nn.initializers.variance_scaling(1.0, 'fan_avg', 'uniform')
ATTENTION_KERNEL_INIT = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal')
MLP_KERNEL_INIT = nn.initializers.variance_scaling(1.0, 'fan_in',
                                                   'truncated_normal')
FINAL_KERNEL_INIT = nn.initializers.variance_scaling(1.0, 'fan_in',
                                                     'truncated_normal')
BIAS_INIT = nn.initializers.normal(stddev=1e-6)



def make_token_emb1(vocab_size, dtype, name='token_embedder', num_features=13):
  """First test configuration for token embeddings."""
  return embedding.Embed(
      num_embeddings=vocab_size,
      features=num_features,
      cast_input_dtype=jnp.int32,
      dtype=dtype,
      attend_dtype=jnp.float32,  # for logit training stability
      embedding_init=EMBEDDING_INIT,
      name=name,
  )


def make_attention1(num_attn_heads, dtype):
  """First test configuration for attention."""
  return dense_attention.MultiHeadDotProductAttention(
      num_heads=num_attn_heads,
      dtype=dtype,
      qkv_features=512,
      head_dim=None,
      kernel_init=ATTENTION_KERNEL_INIT,
      bias_init=BIAS_INIT,
      use_bias=False,
      broadcast_dropout=True,
      dropout_rate=0.1)


def make_mlp1(dtype):
  """First test configuration for the MLP."""
  return dense.MlpBlock(
      use_bias=False,
      intermediate_dim=2048,
      activations=('relu',),
      kernel_init=MLP_KERNEL_INIT,
      bias_init=BIAS_INIT,
      intermediate_dropout_rate=0.1,
      final_dropout_rate=0.1,
      dtype=dtype)


def _make_relpos_bias(
    num_attn_heads: int,
    dtype: Any) -> relative_position_biases.RelativePositionBiases:
  return relative_position_biases.RelativePositionBiases(
      num_buckets=32,
      max_distance=128,
      num_heads=num_attn_heads,
      dtype=dtype,
      embedding_init=RELPOS_BIAS_INIT)


def make_test_dual_encoder(
    similarity_fn: str,
    pool_method: str,
) -> dual_encoder_architecture.DualEncoder:
  dtype = jnp.float32
  num_attn_heads = 8
  make_dropout = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,))
  make_layer_norm = layer_norm.T5LayerNorm

  def _make_encoder_layer(shared_relative_position_bias):
    assert shared_relative_position_bias is not None
    return flaxformer_t5_architecture.EncoderLayer(
        attention=make_attention1(num_attn_heads, dtype),
        mlp=make_mlp1(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        shared_relative_position_bias=shared_relative_position_bias)

  def _make_projection_layer():
    return dense.DenseGeneral(
        PROJECTION_DIM,
        use_bias=False,
        dtype=dtype,
        kernel_init=FINAL_KERNEL_INIT,
        bias_init=BIAS_INIT)

  def _make_encoder(*, shared_token_embedder=None):
    assert shared_token_embedder is None
    return flaxformer_t5_architecture.Encoder(
        num_layers=3,
        token_embedder_factory=lambda: make_token_emb1(4, dtype),
        layer_factory=_make_encoder_layer,
        input_dropout_factory=make_dropout,
        output_dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        shared_relative_position_bias_factory=lambda: _make_relpos_bias(  # pylint: disable=g-long-lambda
            num_attn_heads, dtype),
    )

  def _make_pooler(pool_method):
    """These make utilities are only for tests.

    In practice, the pooler functors are set in gin files.

    Arguments:
      pool_method: pooling method to obtain the encodings

    Returns:
      pooler functor.
    """
    if pool_method == 'mean':
      return poolings.MeanPooling()
    elif pool_method == 'max':
      return poolings.MaxPooling()
    elif pool_method == 'attention':
      return poolings.AttentionPooling()
    else:
      raise ValueError(f'Do not support pooling method: {pool_method}.')

  def _make_similarity_layer():
    if similarity_fn == 'batch_dot_product':
      return similarity_functions.BatchDotProduct(name=similarity_fn)
    if similarity_fn == 'pointwise_ffnn':
      make_dropout = lambda: nn.Dropout(rate=0.1)
      return similarity_functions.PointwiseFFNN(
          OUTPUT_DIM, dropout_factory=make_dropout, name=similarity_fn
      )
    if similarity_fn == 'single_tower_pointwise_ffnn':
      make_dropout = lambda: nn.Dropout(rate=0.1)
      return single_tower_logit_functions.SingleTowerPointwiseFFNN(
          OUTPUT_DIM, dropout_factory=make_dropout, name=similarity_fn
      )

  def _make_l2_norm():
    return l2_norm.L2Norm()

  return dual_encoder_architecture.DualEncoder(
      shared_token_embedder_factory=lambda: None,
      encoder_factory=_make_encoder,
      pooler_factory=None
      if pool_method == 'first'
      else lambda: _make_pooler(pool_method),
      l2_norm_factory=_make_l2_norm,
      projection_layer_factory=_make_projection_layer,
      similarity_layer_factory=_make_similarity_layer,
      dtype=dtype,  # pytype: disable=wrong-keyword-args
  )




class DualEncoderTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('max_pooling', 'max'), ('mean_pooling', 'mean'),
      ('attention_pooling', 'attention'), ('first_token', 'first'))
  def test_dual_encoder_with_batch_dot_product_shapes(self, pool_method):
    """Tests if the dual encoder with batch dot product has correct output shapes."""

    left_inputs = np.array(
        [
            # Batch 1.
            [101, 183, 20, 75],
            # Batch 2.
            [101, 392, 19, 7],
        ],
        dtype=np.int32)
    right_inputs = np.array(
        [
            # Batch 1.
            [101, 283, 85],
            # Batch 2.
            [101, 492, 17],
        ],
        dtype=np.int32)

    model = make_test_dual_encoder(
        similarity_fn='batch_dot_product', pool_method=pool_method)
    results, variables = model.init_with_output(
        random.PRNGKey(0), left_inputs, right_inputs, enable_dropout=False
    )
    left_encoded = results.left_encoded
    right_encoded = results.right_encoded
    logits = results.logits

    reformatted = model.apply({},
                              variables['params'],
                              method=model.to_save_format)
    if pool_method == 'attention':
      check_dual_encoder_params(
          reformatted,
          'dual_encoder_shapes_batch_dot_product_attention_pooling.json')
    else:
      check_dual_encoder_params(reformatted,
                                'dual_encoder_shapes_batch_dot_product.json')

    self.assertEqual(left_encoded.shape, (2, PROJECTION_DIM))
    self.assertEqual(right_encoded.shape, (2, PROJECTION_DIM))
    self.assertEqual(logits.shape, (2, 2))

  def test_dual_encoder_with_batch_dot_product_negative_shapes(self):
    """Tests if the dual encoder with batch dot product and negative inpus has correct output shapes."""

    left_inputs = np.array(
        [
            # Batch 1.
            [101, 183, 20, 75],
            # Batch 2.
            [101, 392, 19, 7],
        ],
        dtype=np.int32)
    right_inputs = np.array(
        [
            # Batch 1.
            [101, 283, 85],
            # Batch 2.
            [101, 492, 17],
        ],
        dtype=np.int32)

    right_negative_inputs = np.array(
        [
            # Batch 1.
            [101, 135, 27],
            # Batch 2.
            [101, 129, 76],
        ],
        dtype=np.int32)

    model = make_test_dual_encoder(
        similarity_fn='batch_dot_product', pool_method='mean')
    results, variables = model.init_with_output(
        random.PRNGKey(0),
        left_inputs,
        right_inputs,
        right_negative_inputs,
        enable_dropout=False,
    )
    left_encoded = results.left_encoded
    right_encoded = results.right_encoded
    logits = results.logits

    reformatted = model.apply({},
                              variables['params'],
                              method=model.to_save_format)
    check_dual_encoder_params(reformatted,
                              'dual_encoder_shapes_batch_dot_product.json')

    left_batch_size = left_inputs.shape[0]
    right_batch_size = right_inputs.shape[0]
    negative_batch_size = right_negative_inputs.shape[0]
    self.assertEqual(left_encoded.shape, (left_batch_size, PROJECTION_DIM))
    self.assertEqual(right_encoded.shape, (right_batch_size, PROJECTION_DIM))
    self.assertEqual(logits.shape,
                     (left_batch_size, right_batch_size + negative_batch_size))

  def test_dual_encoder_with_pointwise_ffnn_shapes(self):
    """Tests if the dual encoder with pointwise ffnn has correct output shapes."""

    left_inputs = np.array(
        [
            # Batch 1.
            [101, 183, 20, 75],
            # Batch 2.
            [101, 392, 19, 7],
        ],
        dtype=np.int32)
    right_inputs = np.array(
        [
            # Batch 1.
            [101, 283, 85],
            # Batch 2.
            [101, 492, 17],
        ],
        dtype=np.int32)

    model = make_test_dual_encoder(
        similarity_fn='pointwise_ffnn', pool_method='first')
    results, variables = model.init_with_output(
        random.PRNGKey(0), left_inputs, right_inputs, enable_dropout=False
    )
    left_encoded = results.left_encoded
    right_encoded = results.right_encoded
    logits = results.logits

    reformatted = model.apply({},
                              variables['params'],
                              method=model.to_save_format)
    check_dual_encoder_params(reformatted,
                              'dual_encoder_shapes_pointwise_ffnn.json')

    self.assertEqual(left_encoded.shape, (2, PROJECTION_DIM))
    self.assertEqual(right_encoded.shape, (2, PROJECTION_DIM))
    self.assertEqual(logits.shape, (2, OUTPUT_DIM))

  def test_dual_encoder_with_single_tower_pointwise_ffnn_shapes(self):
    """Tests if DE with single tower pointwise ffnn has correct output shapes.
    """

    left_inputs = np.array(
        [
            # Batch 1.
            [101, 183, 20, 75],
            # Batch 2.
            [101, 392, 19, 7],
        ],
        dtype=np.int32)
    right_inputs = np.array(
        [
            # Batch 1.
            [101, 283, 85],
            # Batch 2.
            [101, 492, 17],
        ],
        dtype=np.int32)

    model = make_test_dual_encoder(
        similarity_fn='single_tower_pointwise_ffnn', pool_method='first')
    results, variables = model.init_with_output(
        random.PRNGKey(0), left_inputs, right_inputs, enable_dropout=False
    )
    left_encoded = results.left_encoded
    right_encoded = results.right_encoded
    logits = results.logits

    reformatted = model.apply({},
                              variables['params'],
                              method=model.to_save_format)

    check_dual_encoder_params(
        reformatted, 'dual_encoder_shapes_single_tower_pointwise_ffnn.json')

    self.assertEqual(left_encoded.shape, (2, PROJECTION_DIM))
    self.assertEqual(right_encoded.shape, (2, PROJECTION_DIM))
    self.assertEqual(logits.shape, (2, OUTPUT_DIM))



if __name__ == '__main__':
  absltest.main()
