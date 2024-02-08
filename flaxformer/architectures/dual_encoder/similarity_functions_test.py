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

"""Tests for similarity functions."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from jax import random
import jax.numpy as jnp
import tensorflow as tf

from flaxformer.architectures.dual_encoder import poolings
from flaxformer.architectures.dual_encoder import similarity_functions
from flaxformer.components import dense
from flaxformer.components import layer_norm
from flaxformer.components.attention import dense_attention


BATCH_SIZE = 2
DTYPE = jnp.float32
NUM_ATTN_HEADS = 13
OUTPUT_DIM = 17

ATTENTION_KERNEL_INIT = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal'
)
BIAS_INIT = nn.initializers.normal(stddev=1e-6)
MLP_KERNEL_INIT = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'truncated_normal'
)


def make_attention():
  """Test configuration for attention."""
  return dense_attention.MultiHeadDotProductAttention(
      num_heads=NUM_ATTN_HEADS,
      dtype=DTYPE,
      qkv_features=512,
      head_dim=None,
      kernel_init=ATTENTION_KERNEL_INIT,
      bias_init=BIAS_INIT,
      use_bias=False,
      broadcast_dropout=True,
      dropout_rate=0.1,
  )


def make_dropout():
  """Test configuration for the dropout layer."""
  return nn.Dropout(rate=0.1)


def make_mlp():
  """Test configuration for the MLP."""
  return dense.MlpBlock(
      use_bias=False,
      intermediate_dim=2048,
      out_dim=1,
      activations=('relu',),
      kernel_init=MLP_KERNEL_INIT,
      bias_init=BIAS_INIT,
      intermediate_dropout_rate=0.1,
      dtype=DTYPE,
  )


def make_batch_attention_similarity_model():
  """Test configuration for BatchAttentionSimilarity module."""
  return similarity_functions.BatchAttentionSimilarity(
      attention=make_attention(),
      mlp_layer=make_mlp(),
      layer_norm_factory=layer_norm.T5LayerNorm,
      activation_fn='linear',
      dropout_factory=make_dropout,
  )




class SimilarityFunctionsTest(absltest.TestCase):

  def test_pointwise_ffnn(self):
    """Test if the PointwiseFFNN similarity function has correct shapes."""
    model = similarity_functions.PointwiseFFNN(
        OUTPUT_DIM, dropout_factory=make_dropout
    )

    rng = random.PRNGKey(0)
    key1, key2, key3 = random.split(rng, 3)
    x = random.normal(key1, (BATCH_SIZE, 8))
    y = random.normal(key2, (BATCH_SIZE, 8))
    z, _ = model.init_with_output(key3, x, y, enable_dropout=False)
    self.assertEqual(z.shape, (BATCH_SIZE, OUTPUT_DIM))

  def test_pointwise_ffnn_without_dropout(self):
    """Test if the PointwiseFFNN similarity function has correct shapes."""
    model = similarity_functions.PointwiseFFNN(OUTPUT_DIM, dropout_factory=None)

    rng = random.PRNGKey(0)
    key1, key2, key3 = random.split(rng, 3)
    x = random.normal(key1, (BATCH_SIZE, 8))
    y = random.normal(key2, (BATCH_SIZE, 8))
    z, _ = model.init_with_output(key3, x, y, enable_dropout=False)
    self.assertEqual(z.shape, (BATCH_SIZE, OUTPUT_DIM))

  def test_batch_dot_product(self):
    """Test if the BatchDotProduct similarity function has correct shapes."""
    rng = random.PRNGKey(0)
    key1, key2, key3 = random.split(rng, 3)
    x = random.normal(key1, (BATCH_SIZE, 8))
    y = random.normal(key2, (BATCH_SIZE, 8))
    model = similarity_functions.BatchDotProduct()
    z, _ = model.init_with_output(key3, x, y)
    self.assertEqual(z.shape, (BATCH_SIZE, BATCH_SIZE))

  def test_batch_dot_product_with_negative(self):
    """Test if the BatchDotProduct similarity function has correct shapes."""
    rng = random.PRNGKey(0)
    key1, key2, key3, key4 = random.split(rng, 4)
    left_encodings = random.normal(key1, (BATCH_SIZE, 8))
    right_encodings = random.normal(key2, (BATCH_SIZE, 8))
    right_negative_encodings = random.normal(key3, (BATCH_SIZE, 8))
    model = similarity_functions.BatchDotProduct()
    logits, _ = model.init_with_output(
        key4, left_encodings, right_encodings, right_negative_encodings
    )
    # The shape of logits equals [num_positive, num_positive + num_negative]
    # where both num_positive and num_negative equals BATCH_SIZE.
    self.assertEqual(logits.shape, (BATCH_SIZE, BATCH_SIZE + BATCH_SIZE))

  def test_batch_dot_product_with_negative_use_only_explicit_hard_negatives(
      self,
  ):
    """Test if the BatchDotProduct similarity function has correct shapes."""
    rng = random.PRNGKey(0)
    key1, key2, key3, key4 = random.split(rng, 4)
    left_encodings = random.normal(key1, (BATCH_SIZE, 8))
    right_encodings = random.normal(key2, (BATCH_SIZE, 8))
    right_negative_encodings = random.normal(key3, (BATCH_SIZE, 8))
    model = similarity_functions.BatchDotProduct(
        use_only_explicit_hard_negatives=True
    )
    logits, _ = model.init_with_output(
        key4, left_encodings, right_encodings, right_negative_encodings
    )
    # The shape of logits equals [num_positive, num_positive + 1].
    self.assertEqual(logits.shape, (BATCH_SIZE, BATCH_SIZE + 1))

  def test_pointwise_ffnn_with_multiple_layers(self):
    """Test the Multi-layer PointwiseFFNN has correct shapes."""
    model = similarity_functions.PointwiseFFNN(
        OUTPUT_DIM, dropout_factory=None, intermediate_features=[1024, 512, 55])

    rng = random.PRNGKey(0)
    key1, key2, key3 = random.split(rng, 3)
    x = random.normal(key1, (BATCH_SIZE, 8))
    y = random.normal(key2, (BATCH_SIZE, 8))
    z, _ = model.init_with_output(key3, x, y, enable_dropout=False)
    self.assertEqual(z.shape, (BATCH_SIZE, OUTPUT_DIM))


class SimilarityFunctionsParameterizedTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.named_parameters(('pointwise', True), ('elementwise', False))
  def test_batch_attention_similarity(self, pointwise_similarity):
    """Test the BatchAttentionSimilarity."""
    model = make_batch_attention_similarity_model()

    if pointwise_similarity:
      batch_size_1 = BATCH_SIZE
      batch_size_2 = BATCH_SIZE
    else:
      batch_size_1 = 2 * BATCH_SIZE
      batch_size_2 = 3 * BATCH_SIZE

    rng = random.PRNGKey(0)
    keys = random.split(rng, 3)
    left_encodings = random.normal(keys[0], (batch_size_1, 11, 23))
    right_encodings = random.normal(keys[1], (batch_size_2, 3, 23))
    left_mask = jnp.ones((batch_size_1, 11))
    right_mask = jnp.ones((batch_size_2, 3))
    output, _ = model.init_with_output(
        keys[2],
        left_encodings,
        right_encodings,
        left_mask,
        right_mask,
        enable_dropout=False,
        pointwise_similarity=pointwise_similarity,
    )
    if pointwise_similarity:
      self.assertEqual(output.shape, (batch_size_1,))
    else:
      self.assertEqual(output.shape, (batch_size_1, batch_size_2))

      # The following is to catch the ordering of the output, to ensure the
      # [i, j]-th element is the similarity calculated from
      # [left_encoding[i], right_encoding[j]].
      pointwise_output, _ = model.init_with_output(
          keys[2],
          left_encodings,
          right_encodings[:batch_size_1, :, :],
          left_mask,
          right_mask[:batch_size_1, :],
          enable_dropout=False,
          pointwise_similarity=True,
      )

      self.assertAllClose(jnp.diagonal(output), pointwise_output)


if __name__ == '__main__':
  absltest.main()
