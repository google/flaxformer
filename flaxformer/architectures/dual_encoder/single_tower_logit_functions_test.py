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
from absl.testing import absltest
from flax import linen as nn
from jax import random

from flaxformer.architectures.dual_encoder import single_tower_logit_functions

OUTPUT_DIM = 3
BATCH_SIZE = 2


class SimilarityFunctionsTest(absltest.TestCase):

  def test_single_tower_pointwise_ffnn(self):
    """Test if the PointwiseFFNN similarity function has correct shapes."""
    make_dropout = lambda: nn.Dropout(rate=0.1)
    model = single_tower_logit_functions.SingleTowerPointwiseFFNN(
        OUTPUT_DIM, dropout_factory=make_dropout)
    rng = random.PRNGKey(0)
    key1, key2, key3 = random.split(rng, 3)
    x = random.normal(key1, (BATCH_SIZE, 8))
    y = random.normal(key2, (BATCH_SIZE, 8))
    z, _ = model.init_with_output(key3, x, y, enable_dropout=False)
    self.assertEqual(z.shape, (BATCH_SIZE, OUTPUT_DIM))

if __name__ == "__main__":
  absltest.main()
