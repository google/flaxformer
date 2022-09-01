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

"""Tests for similarity functions."""
from absl.testing import absltest
from flax import linen as nn
from jax import random

from flaxformer.architectures.dual_encoder import similarity_functions

OUTPUT_DIM = 3
BATCH_SIZE = 2


class SimilarityFunctionsTest(absltest.TestCase):

  def test_pointwise_ffnn(self):
    """Test if the PointwiseFFNN similarity function has correct shapes."""
    make_dropout = lambda: nn.Dropout(rate=0.1)
    model = similarity_functions.PointwiseFFNN(
        OUTPUT_DIM, dropout_factory=make_dropout)

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
    logits, _ = model.init_with_output(key4, left_encodings, right_encodings,
                                       right_negative_encodings)
    # The shape of logits equals [num_positive, num_positive + num_negative]
    # where both num_positive and num_negative equals BATCH_SIZE.
    self.assertEqual(logits.shape, (BATCH_SIZE, BATCH_SIZE + BATCH_SIZE))

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


if __name__ == "__main__":
  absltest.main()
