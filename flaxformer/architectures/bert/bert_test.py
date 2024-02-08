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

"""Tests for flaxformer.architecture.bert.bert."""
import dataclasses
import json
import pathlib

from absl.testing import absltest
from flax.core import unfreeze
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from flaxformer import testing_utils
from flaxformer.architectures.bert import bert
from flaxformer.architectures.bert import configs as bert_configs
from flaxformer.architectures.bert import heads
from flaxformer.components.attention import dense_attention

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


def _testdata_dir() -> pathlib.Path:
  return (
      pathlib.Path(absltest.get_default_test_srcdir()) /
      'flaxformer/architectures/bert/testdata')


def make_bert_inputs():
  """Returns inputs that can be fed to a BERT encoder."""
  token_ids = np.array([[5, 6, 7], [4, 2, 9]], dtype=np.int32)
  position_ids = np.array([[0, 1, 2]], dtype=np.int32)
  segment_ids = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.int32)
  mask = np.ones((2, 3))
  return token_ids, position_ids, segment_ids, mask


class BertEncoderTest(absltest.TestCase):

  def test_output_shape(self):
    """Tests that BertEncoder outputs are of the correct shape."""
    token_ids = np.array([[5, 6, 7], [4, 2, 9]], dtype=np.int32)
    position_ids = np.array([[0, 1, 2]], dtype=np.int32)
    segment_ids = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.int32)
    mask = np.ones((2, 3))
    hidden_size = 4
    model = bert.BertEncoder(
        hidden_size=hidden_size,
        intermediate_dim=5,
        vocab_size=10,
        max_length=11,
        num_segments=2,
        num_hidden_layers=3,
        num_attention_heads=2)
    output, _ = model.init_with_output(
        jax.random.PRNGKey(0),
        token_ids,
        position_ids,
        segment_ids,
        mask,
        enable_dropout=False)
    self.assertEqual(token_ids.shape + (hidden_size,), output.shape)

  def test_param_shapes(self):
    token_ids = np.array([[5, 6, 7], [4, 2, 9]], dtype=np.int32)
    position_ids = np.array([[0, 1, 2]], dtype=np.int32)
    segment_ids = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.int32)
    mask = np.ones((2, 3))
    hidden_size = 4
    model = bert.BertEncoder(
        hidden_size=hidden_size,
        intermediate_dim=5,
        vocab_size=10,
        max_length=11,
        num_segments=2,
        num_hidden_layers=2,
        num_attention_heads=2)
    params = model.init(
        random.PRNGKey(0),
        token_ids,
        position_ids,
        segment_ids,
        mask,
        enable_dropout=False)['params']
    self.assertSameStructure(
        testing_utils.param_shapes(params),
        json.load(open(_testdata_dir() / 'model_param_shapes.json')),
        'Full params = ' +
        testing_utils.format_params_shapes(testing_utils.param_shapes(params)))

  def test_bert_from_bert_base_config(self):
    """Tests that BertEncoder can be constructed from a config object."""
    token_ids = np.array([[5, 6, 7], [4, 2, 9]], dtype=np.int32)
    position_ids = np.array([[0, 1, 2]], dtype=np.int32)
    segment_ids = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.int32)
    mask = np.ones((2, 3))

    # A BERT Base config but smaller.
    config = bert_configs.BertBaseConfig(
        hidden_size=4,
        intermediate_dim=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        vocab_size=11,
        max_length=13,
        num_segments=3)
    model = bert.BertEncoder(**dataclasses.asdict(config))
    output, _ = model.init_with_output(
        jax.random.PRNGKey(0),
        token_ids,
        position_ids,
        segment_ids,
        mask,
        enable_dropout=False)
    self.assertEqual(token_ids.shape + (config.hidden_size,), output.shape)


class BertMlmNspTest(absltest.TestCase):

  def test_output_shapes(self):
    hidden_size = 4
    vocab_size = 10
    encoder = bert.BertEncoder(
        hidden_size=hidden_size,
        intermediate_dim=5,
        vocab_size=vocab_size,
        max_length=11,
        num_segments=2,
        num_hidden_layers=2,
        num_attention_heads=2)
    pooler = heads.BertPooler()
    mlm_head = heads.MLMHead(
        encoder=encoder, hidden_size=hidden_size, vocab_size=vocab_size)
    nsp_head = heads.NSPHead(pooler=pooler)
    model = bert.BertMlmNsp(
        encoder=encoder, pooler=pooler, mlm_head=mlm_head, nsp_head=nsp_head)
    token_ids, position_ids, segment_ids, mask = make_bert_inputs()
    output, _ = model.init_with_output(
        jax.random.PRNGKey(0),
        token_ids,
        position_ids=position_ids,
        segment_ids=segment_ids,
        input_mask=mask,
        enable_dropout=False)
    mlm_output, nsp_output = output
    self.assertEqual(token_ids.shape + (vocab_size,), mlm_output.shape)
    self.assertEqual(token_ids.shape[:1] + (2,), nsp_output.shape)

  def test_params(self):
    hidden_size = 4
    vocab_size = 10
    encoder = bert.BertEncoder(
        hidden_size=hidden_size,
        intermediate_dim=5,
        vocab_size=vocab_size,
        max_length=11,
        num_segments=2,
        num_hidden_layers=2,
        num_attention_heads=2)
    pooler = heads.BertPooler()
    mlm_head = heads.MLMHead(
        encoder=encoder, hidden_size=hidden_size, vocab_size=vocab_size)
    nsp_head = heads.NSPHead(pooler=pooler)
    model = bert.BertMlmNsp(
        encoder=encoder, pooler=pooler, mlm_head=mlm_head, nsp_head=nsp_head)
    token_ids, position_ids, segment_ids, mask = make_bert_inputs()
    params = model.init(
        jax.random.PRNGKey(0),
        token_ids,
        position_ids=position_ids,
        segment_ids=segment_ids,
        input_mask=mask,
        enable_dropout=False)['params']
    param_shapes = testing_utils.param_shapes(params)
    expected_encoder_param_shapes = json.load(
        open(_testdata_dir() / 'model_param_shapes.json'))
    expected_param_shapes = {
        'encoder': expected_encoder_param_shapes,
        'mlm_head': {
            'bias': (10,),
            'dense': {
                'bias': (4,),
                'kernel': (4, 4)
            },
            'layer_norm': {
                'bias': (4,),
                'scale': (4,)
            }
        },
        'nsp_head': {
            'dense': {
                'bias': (2,),
                'kernel': (4, 2)
            }
        },
        'pooler': {
            'dense': {
                'bias': (4,),
                'kernel': (4, 4)
            }
        }
    }
    self.assertSameStructure(param_shapes, expected_param_shapes)


class BertClassifierTest(absltest.TestCase):

  def test_output_shapes(self):
    hidden_size = 4
    num_classes = 3
    encoder = bert.BertEncoder(
        hidden_size=hidden_size,
        intermediate_dim=5,
        vocab_size=10,
        max_length=11,
        num_segments=2,
        num_hidden_layers=2,
        num_attention_heads=2)
    pooler = heads.BertPooler()
    classifier_head = heads.ClassifierHead(
        pooler=pooler, num_classes=num_classes)
    model = bert.BertClassifier(
        encoder=encoder, pooler=pooler, classifier_head=classifier_head)
    token_ids, position_ids, segment_ids, mask = make_bert_inputs()
    output, _ = model.init_with_output(
        jax.random.PRNGKey(0),
        token_ids,
        position_ids=position_ids,
        segment_ids=segment_ids,
        input_mask=mask,
        enable_dropout=False)
    self.assertEqual((token_ids.shape[0], num_classes), output.shape)

  def test_params(self):
    hidden_size = 4
    num_classes = 3
    encoder = bert.BertEncoder(
        hidden_size=hidden_size,
        intermediate_dim=5,
        vocab_size=10,
        max_length=11,
        num_segments=2,
        num_hidden_layers=2,
        num_attention_heads=2)
    pooler = heads.BertPooler()
    classifier_head = heads.ClassifierHead(
        pooler=pooler, num_classes=num_classes, use_bias=True)
    model = bert.BertClassifier(
        encoder=encoder, pooler=pooler, classifier_head=classifier_head)
    token_ids, position_ids, segment_ids, mask = make_bert_inputs()
    params = model.init(
        jax.random.PRNGKey(0),
        token_ids,
        position_ids=position_ids,
        segment_ids=segment_ids,
        input_mask=mask,
        enable_dropout=False)['params']
    param_shapes = testing_utils.param_shapes(params)
    expected_encoder_param_shapes = json.load(
        open(_testdata_dir() / 'model_param_shapes.json'))
    expected_param_shapes = {
        'encoder': expected_encoder_param_shapes,
        'classifier_head': {
            'dense': {
                'bias': (3,),
                'kernel': (4, 3)
            },
        },
        'pooler': {
            'dense': {
                'bias': (4,),
                'kernel': (4, 4)
            }
        }
    }
    self.assertSameStructure(param_shapes, expected_param_shapes)


class EncoderLayerTest(absltest.TestCase):

  def test_wrong_inputs_dimension(self):
    """Tests if the encoder exception is raised for wrong `inputs` dimension."""
    params_key, dropout_key = random.split(random.PRNGKey(0), 2)
    input_shape = (4,)
    inputs = random.uniform(random.PRNGKey(0), input_shape, dtype=jnp.float32)
    encoder_layer = bert.make_encoder_layer(
        bert.make_attention_layer(num_heads=2),
        hidden_size=4,
        intermediate_dim=14)
    with self.assertRaisesRegex(AssertionError, r'.+ shape \(4,\).*'):
      encoder_layer.init_with_output(
          {
              'params': params_key,
              'dropout': dropout_key
          },
          inputs=inputs,
          attention_targets=inputs)


class AttentionBlockTest(absltest.TestCase):

  def test_output_shape(self):
    """Tests that BERT attention block's output is of correct shape."""
    params_key, dropout_key = random.split(random.PRNGKey(0), 2)
    input_shape = (2, 3, 4)
    inputs = random.uniform(random.PRNGKey(0), input_shape, dtype=jnp.float32)
    layer = bert.make_attention_block(
        bert.make_attention_layer(num_heads=2), hidden_size=4)
    result, variables = layer.init_with_output(
        {
            'params': params_key,
            'dropout': dropout_key
        },
        inputs=inputs,
        attention_targets=inputs,
        enable_dropout=False)
    self.assertEqual(input_shape, result.shape)

    # Note: The layernorm has no axes annotations.
    params = unfreeze(variables['params'])
    del params['layer_norm']
    self.assertDictEqual(
        testing_utils.param_dtypes_shapes_axes(params,
                                               variables['params_axes']),
        {
            'attention_layer': {
                'key': {
                    'kernel': ['float32', 'embed=4', 'joined_kv=4'],
                    'bias': ['float32', 'joined_kv=4']
                },
                'query': {
                    'kernel': ['float32', 'embed=4', 'joined_kv=4'],
                    'bias': ['float32', 'joined_kv=4']
                },
                'value': {
                    'kernel': ['float32', 'embed=4', 'joined_kv=4'],
                    'bias': ['float32', 'joined_kv=4']
                },
            },
            'dense_layer': {
                'kernel': ['float32', 'joined_kv=4', 'embed=4'],
                'bias': ['float32', 'embed=4']
            },
        })

  def test_output_shape_with_mask(self):
    """Tests that attention block's output is of correct shape with a mask."""
    params_key, dropout_key = random.split(random.PRNGKey(0), 2)
    batch_size, max_seq_len, emb_dims = (2, 3, 4)
    input_shape = (batch_size, max_seq_len, emb_dims)
    inputs_mask = jnp.array([[1, 0, 0], [1, 1, 0]])
    inputs = random.uniform(random.PRNGKey(0), input_shape, dtype=jnp.float32)
    mask = dense_attention.make_attention_mask(inputs_mask, inputs_mask)

    layer = bert.make_attention_block(
        bert.make_attention_layer(num_heads=2), hidden_size=4)
    result, _ = layer.init_with_output(
        {
            'params': params_key,
            'dropout': dropout_key
        },
        inputs=inputs,
        attention_targets=inputs,
        mask=mask,
        enable_dropout=False)
    self.assertEqual(input_shape, result.shape)

  def test_wrong_head_count(self):
    """Tests that exception is raised for wrong head count."""
    params_key, dropout_key = random.split(random.PRNGKey(0), 2)
    input_shape = (2, 3, 4)
    inputs = random.uniform(random.PRNGKey(0), input_shape, dtype=jnp.float32)
    layer = bert.make_attention_block(
        bert.make_attention_layer(num_heads=5), hidden_size=4)

    with self.assertRaisesRegex(AssertionError, '.* 4 is not divisible by 5'):
      layer.init_with_output({
          'params': params_key,
          'dropout': dropout_key
      },
                             inputs=inputs,
                             attention_targets=inputs,
                             enable_dropout=False)


class MlpBlockTest(absltest.TestCase):

  def test_output_shape(self):
    """Tests that BERT mlp block's output is of correct shape."""
    params_key, dropout_key = random.split(random.PRNGKey(0), 2)
    input_shape = (2, 3, 4)
    inputs = random.uniform(random.PRNGKey(0), input_shape, dtype=jnp.float32)
    layer = bert.make_mlp_block(hidden_size=4, intermediate_dim=14)
    result, variables = layer.init_with_output(
        {
            'params': params_key,
            'dropout': dropout_key
        },
        inputs,
        enable_dropout=False)
    self.assertEqual(input_shape, result.shape)

    # Note: The layernorm has no axes annotations.
    params = unfreeze(variables['params'])
    del params['layer_norm']
    self.assertDictEqual(
        testing_utils.param_dtypes_shapes_axes(params,
                                               variables['params_axes']),
        {
            'mlp': {
                'dense_layer': {
                    'kernel': ['float32', 'embed=4', 'mlp=14'],
                    'bias': ['float32', 'mlp=14']
                },
            },
            'dense_layer': {
                'kernel': ['float32', 'mlp=14', 'embed=4'],
                'bias': ['float32', 'embed=4']
            },
        })


if __name__ == '__main__':
  absltest.main()
