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

"""Convert BERT TF checkpoints to Flaxformer's Encoder format."""

from typing import Any, Dict, Tuple

import flax
from flax.core import frozen_dict
import tensorflow as tf

from flaxformer import param_conversion_util


def load_params_from_tf_checkpoint(
    checkpoint_path: str) -> Tuple[flax.core.FrozenDict, flax.core.FrozenDict]:
  """Load Flax BERT Encoder params from a TF BERT checkpoint.

  The method loads the provided TF checkpoint, stores it in the dictionary
  with tf parameter names as keys and the weight tensors as values. Unlike Flax
  TF doesn't return the nested structure of the model weights, i.e. keys are
  recorded in the following format:
  "outer_layer_name/..middle_layer_name/...inner_tensor". Since we know the
  mapping between the TF keys and the Flax keys we recursively assemble
  the Flax params dictionary starting from the most outer layer.

  Args:
    checkpoint_path: Path to the TF checkpoint.

  Returns:
    A tuple of frozen dicts that are the parameters for the bert.BertEncoder and
    bert.heads.BertPooler.
  """
  ckpt_reader = tf.train.load_checkpoint(checkpoint_path)
  tf_params = {
      tf_name: ckpt_reader.get_tensor(tf_name)
      for tf_name in ckpt_reader.get_variable_to_dtype_map()
  }
  encoder_params = convert_bert_encoder_params(tf_params, 'bert/')
  pooler_params = param_conversion_util.convert_tf_params(
      tf_params, {
          ('dense', 'bias'): 'bias',
          ('dense', 'kernel'): 'kernel'
      }, 'bert/pooler/dense/')

  # TODO: Update this when BertEncoder is deleted.
  return frozen_dict.freeze(encoder_params), frozen_dict.freeze(pooler_params)


# TODO: Delete this when BertEncoder is deleted.
def convert_bert_encoder_params(tf_params: Dict[str, tf.Tensor],
                                prefix: str) -> Dict[str, Any]:
  """Loads all Flax BertEncoder parameters from the TF params."""
  return {
      'embedder':
          param_conversion_util.convert_tf_params(
              tf_params, {
                  ('embedders_position_ids', 'embedding'):
                      'position_embeddings',
                  ('embedders_segment_ids', 'embedding'):
                      'token_type_embeddings',
                  ('embedders_token_ids', 'embedding'):
                      'word_embeddings',
              }, f'{prefix}embeddings/'),
      'layer_norm':
          convert_layer_norm_params(tf_params,
                                    f'{prefix}embeddings/LayerNorm/'),
      'encoder_block':
          convert_encoder_block_params(tf_params, f'{prefix}encoder/'),
  }


def convert_full_encoder_params(tf_params: Dict[str, tf.Tensor],
                                prefix: str) -> Dict[str, Any]:
  """Loads all Flax Encoder parameters from the TF params."""
  return {
      'embedder_block':
          convert_embedder_block_params(tf_params, f'{prefix}embeddings/'),
      'encoder_block':
          convert_encoder_block_params(tf_params, f'{prefix}encoder/'),
  }


def convert_embedder_block_params(tf_params: Dict[str, tf.Tensor],
                                  prefix: str) -> Dict[str, Any]:
  """Loads all Flax EmbedderBlock parameters from the TF params."""
  return {
      'embedder': convert_embedder_params(tf_params, prefix),
      'layer_norm': convert_layer_norm_params(tf_params, f'{prefix}LayerNorm/'),
  }


def convert_embedder_params(tf_params: Dict[str, tf.Tensor],
                            prefix: str) -> Dict[str, Any]:
  """Loads all Flax Embedder parameters from the TF params."""
  return param_conversion_util.convert_tf_params(
      tf_params, {
          ('embedders_input_ids', 'embedding'): 'word_embeddings',
          ('embedders_position_ids', 'embedding'): 'position_embeddings',
          ('embedders_segment_ids', 'embedding'): 'token_type_embeddings',
      }, prefix)


def convert_encoder_block_params(tf_params: Dict[str, tf.Tensor],
                                 prefix: str) -> Dict[str, Any]:
  """Loads all Flax EncoderBlock parameters from the TF params."""
  layer_indices = param_conversion_util.get_int_regex_matches(
      f'{prefix}layer_(\\d+)/', tf_params)
  if not layer_indices:
    raise ValueError(f'No layers found with prefix {prefix!r}')

  layers = {}
  for i in layer_indices:
    layers[f'layers_{i}'] = (
        convert_encoder_layer_params(tf_params, f'{prefix}layer_{i}/'))
  return {'layer_sequence': layers}


def convert_encoder_layer_params(tf_params: Dict[str, tf.Tensor],
                                 prefix: str) -> Dict[str, Any]:
  """Loads all Flax EncoderLayer parameters from the TF params."""
  return {
      'attention_block':
          convert_self_attention_block_params(tf_params, f'{prefix}attention/'),
      'mlp_block':
          convert_mlp_block_params(tf_params, prefix),
  }


def convert_self_attention_block_params(tf_params: Dict[str, tf.Tensor],
                                        prefix: str) -> Dict[str, Any]:
  """Loads all Flax AttentionBlock parameters from the TF params."""
  return {
      'attention_layer':
          convert_attention_layer_params(tf_params, f'{prefix}self/'),
      'dense_layer':
          convert_dense_layer_params(tf_params, f'{prefix}output/dense/'),
      'layer_norm':
          convert_layer_norm_params(tf_params, f'{prefix}output/LayerNorm/'),
  }


def convert_mlp_block_params(tf_params: Dict[str, tf.Tensor],
                             prefix: str) -> Dict[str, Any]:
  """Loads all Flax MlpBlock parameters from the TF params."""
  return {
      'mlp': {
          'dense_layer':
              convert_dense_layer_params(tf_params,
                                         f'{prefix}intermediate/dense/'),
      },
      'dense_layer':
          convert_dense_layer_params(tf_params, f'{prefix}output/dense/'),
      'layer_norm':
          convert_layer_norm_params(tf_params, f'{prefix}output/LayerNorm/'),
  }


def convert_attention_layer_params(tf_params: Dict[str, tf.Tensor],
                                   prefix: str) -> Dict[str, Any]:
  """Loads all Flax MultiHeadDotProductAttention parameters from the TF params."""
  return {
      'query': convert_dense_layer_params(tf_params, f'{prefix}query/'),
      'key': convert_dense_layer_params(tf_params, f'{prefix}key/'),
      'value': convert_dense_layer_params(tf_params, f'{prefix}value/'),
  }


def convert_dense_layer_params(tf_params: Dict[str, tf.Tensor],
                               prefix: str) -> Dict[str, Any]:
  """Loads all Flax DenseGeneral parameters from the TF params."""
  return param_conversion_util.convert_tf_params(tf_params, {
      ('kernel',): 'kernel',
      ('bias',): 'bias',
  }, prefix)


def convert_layer_norm_params(tf_params: Dict[str, tf.Tensor],
                              prefix: str) -> Dict[str, Any]:
  """Loads all Flax LayerNorm parameters from the TF params."""
  return param_conversion_util.convert_tf_params(tf_params, {
      ('scale',): 'gamma',
      ('bias',): 'beta',
  }, prefix)
