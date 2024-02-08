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

"""Tests for flaxformer.architectures.bert.heads."""
import json
import pathlib

from typing import Optional
from absl.testing import absltest
from absl.testing import parameterized
import flax
from flax import linen as nn
import jax
import numpy as np

from flaxformer import testing_utils
from flaxformer.architectures.bert import bert
from flaxformer.architectures.bert import heads
from flaxformer.types import Array

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


def _testdata_dir() -> pathlib.Path:
  return (
      pathlib.Path(absltest.get_default_test_srcdir()) /
      'flaxformer/architectures/bert/testdata')


class BertHeadsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.token_ids = np.array([[5, 6, 7], [4, 2, 9]], dtype=np.int32)
    self.position_ids = np.array([[0, 1, 2]], dtype=np.int32)
    self.segment_ids = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.int32)
    self.mask = np.ones((2, 3))
    self.vocab_size = 10
    self.hidden_size = 4
    self.model = bert.BertEncoder(
        hidden_size=self.hidden_size,
        intermediate_dim=5,
        num_hidden_layers=2,
        num_attention_heads=2,
        vocab_size=self.vocab_size,
        max_length=11,
        num_segments=2)

  def test_bert_pooler(self):
    """Test whether Bert Pooler returns correct shape."""
    bert_output, _ = self.model.init_with_output(
        jax.random.PRNGKey(0),
        self.token_ids,
        self.position_ids,
        self.segment_ids,
        self.mask,
        enable_dropout=False)
    pooler = heads.BertPooler()
    output, variables = pooler.init_with_output(
        jax.random.PRNGKey(0), bert_output)
    self.assertEqual((2, self.hidden_size), output.shape)

    params = variables['params']
    self.assertDictEqual(
        testing_utils.param_dtypes_shapes_axes(params,
                                               variables['params_axes']),
        {
            'dense': {
                'bias': ['float32', 'mlp=4'],
                'kernel': ['float32', 'embed=4', 'mlp=4']
            }
        })

  def test_bert_classifier(self):
    """Tests whether Classifier returns correct shape."""
    num_classes = 10
    bert_output, _ = self.model.init_with_output(
        jax.random.PRNGKey(0),
        self.token_ids,
        self.position_ids,
        self.segment_ids,
        self.mask,
        enable_dropout=False)
    pooler = heads.BertPooler()
    classifier_head = heads.ClassifierHead(
        pooler, num_classes, enable_dropout=False)
    output, variables = classifier_head.init_with_output(
        jax.random.PRNGKey(0), bert_output)
    self.assertEqual((2, num_classes), output.shape)

    params = variables['params']
    self.assertDictEqual(
        testing_utils.param_dtypes_shapes_axes(params,
                                               variables['params_axes']),
        {
            'dense': {
                'bias': ['float32', 'mlp=10'],
                'kernel': ['float32', 'embed=4', 'mlp=10']
            },
            'pooler': {
                'dense': {
                    'bias': ['float32', 'mlp=4'],
                    'kernel': ['float32', 'embed=4', 'mlp=4']
                }
            }
        })

  def test_bert_nsp(self):
    """Tests whether NSP returns correct shape."""
    bert_output, _ = self.model.init_with_output(
        jax.random.PRNGKey(0),
        self.token_ids,
        self.position_ids,
        self.segment_ids,
        self.mask,
        enable_dropout=False)
    pooler = heads.BertPooler()
    nsp_head = heads.NSPHead(pooler)
    output, variables = nsp_head.init_with_output(
        jax.random.PRNGKey(0), bert_output)
    self.assertEqual((2, 2), output.shape)

    params = variables['params']
    self.assertDictEqual(
        testing_utils.param_dtypes_shapes_axes(params,
                                               variables['params_axes']),
        {
            'dense': {
                'bias': ['float32', 'mlp=2'],
                'kernel': ['float32', 'embed=4', 'mlp=2']
            },
            'pooler': {
                'dense': {
                    'bias': ['float32', 'mlp=4'],
                    'kernel': ['float32', 'embed=4', 'mlp=4']
                }
            }
        })

  def test_gather_indices(self):
    """Tests that gather indices selects the right items in a batch."""
    inputs = np.array([[[0], [1], [2]], [[3], [4], [5]], [[6], [7], [8]]],
                      dtype=np.float32)
    indices = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int32)
    expected = np.array([[[0], [1]], [[4], [5]], [[6], [8]]], dtype=np.float32)
    result = heads.gather_indices(inputs, indices)  # pytype: disable=wrong-arg-types  # jax-ndarray
    np.testing.assert_array_equal(result, expected)

  @parameterized.named_parameters(
      ('without_masked_positions', None),
      ('with_masked_positions', np.array([[1, 2], [0, 1]], dtype=np.int32)))
  def test_bert_mlm(self, masked_positions):
    """Tests whether MLM returns correct shape."""

    class BertMlm(nn.Module):
      encoder: bert.BertEncoder
      mlm_head: heads.MLMHead

      def __call__(self,
                   token_ids: Array,
                   *,
                   position_ids: Array,
                   segment_ids: Array,
                   input_mask: Array,
                   masked_positions: Optional[Array] = None,
                   enable_dropout: bool = True) -> Array:
        bert_output = self.encoder(
            token_ids,
            position_ids=position_ids,
            segment_ids=segment_ids,
            input_mask=input_mask,
            enable_dropout=enable_dropout)
        return self.mlm_head(bert_output, masked_positions=masked_positions)

    mlm_head = heads.MLMHead(self.model, self.hidden_size, self.vocab_size)
    encoder_with_mlm = BertMlm(self.model, mlm_head)
    output, variables = encoder_with_mlm.init_with_output(
        jax.random.PRNGKey(0),
        self.token_ids,
        position_ids=self.position_ids,
        segment_ids=self.segment_ids,
        input_mask=self.mask,
        masked_positions=masked_positions,
        enable_dropout=False)
    params = variables['params']
    param_shapes = testing_utils.param_shapes(params)
    expected_encoder_param_shapes = json.load(
        open(_testdata_dir() / 'model_param_shapes.json'))
    expected_param_shapes = {
        'encoder': expected_encoder_param_shapes,
        'mlm_head': {
            'bias': [10],
            'dense': {
                'bias': [4],
                'kernel': [4, 4]
            },
            'layer_norm': {
                'bias': [4],
                'scale': [4]
            }
        }
    }
    batch_size, seq_length = self.token_ids.shape
    if masked_positions is not None:
      seq_length = masked_positions.shape[1]

    self.assertSameStructure(param_shapes, expected_param_shapes)
    self.assertEqual((batch_size, seq_length, self.vocab_size), output.shape)

    params = flax.core.unfreeze(variables['params'])['mlm_head']
    del params['layer_norm']
    del params['bias']
    self.assertDictEqual(
        testing_utils.param_dtypes_shapes_axes(
            params, variables['params_axes']['mlm_head']), {
                'dense': {
                    'bias': ['float32', 'mlp=4'],
                    'kernel': ['float32', 'embed=4', 'mlp=4']
                }
            })

  def test_MLP(self):
    """Tests whether Token Classifier returns correct shape."""
    num_classes = 10
    bert_output, _ = self.model.init_with_output(
        jax.random.PRNGKey(0),
        self.token_ids,
        self.position_ids,
        self.segment_ids,
        self.mask,
        enable_dropout=False)
    mlp = heads.MLP(features=[self.hidden_size, num_classes])
    output, variables = mlp.init_with_output(
        jax.random.PRNGKey(0), bert_output, enable_dropout=False)
    # We have batch_size=2 and seq_length=3
    self.assertEqual((2, 3, num_classes), output.shape)

    params = variables['params']
    self.assertDictEqual(
        testing_utils.param_dtypes_shapes_axes(params,
                                               variables['params_axes']),
        {
            'dense_0': {
                'bias': ['float32', 'mlp=4'],
                'kernel': ['float32', 'embed=4', 'mlp=4']
            },
            'dense_1': {
                'bias': ['float32', 'mlp=10'],
                'kernel': ['float32', 'embed=4', 'mlp=10']
            }
        })

  def test_bert_token_classifier(self):
    """Tests whether Token Classifier returns correct shape."""
    num_classes = 10
    bert_output, _ = self.model.init_with_output(
        jax.random.PRNGKey(0),
        self.token_ids,
        self.position_ids,
        self.segment_ids,
        self.mask,
        enable_dropout=False)
    token_classifier_head = heads.TokenClassifierHead(
        features=[self.hidden_size, num_classes])
    output, variables = token_classifier_head.init_with_output(
        jax.random.PRNGKey(0), bert_output, enable_dropout=False)
    # We have batch_size=2 and seq_length=3
    self.assertEqual((2, 3, num_classes), output.shape)

    params = variables['params']
    self.assertDictEqual(
        testing_utils.param_dtypes_shapes_axes(params,
                                               variables['params_axes']),
        {
            'mlp': {
                'dense_0': {
                    'bias': ['float32', 'mlp=4'],
                    'kernel': ['float32', 'embed=4', 'mlp=4']
                },
                'dense_1': {
                    'bias': ['float32', 'mlp=10'],
                    'kernel': ['float32', 'embed=4', 'mlp=10']
                }
            }
        })

if __name__ == '__main__':
  absltest.main()
