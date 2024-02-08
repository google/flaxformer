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

"""BERT Base configuration."""
import abc
import dataclasses
from flax import linen as nn
from jax import numpy as jnp
from flaxformer.components import initializers
from flaxformer.types import Initializer  # pylint: disable=g-multiple-import


@dataclasses.dataclass
class BertConfig(abc.ABC):
  """BERT configuration base dataclass."""
  # The size of embeddings/hidden layers, and the size of MLP intermediates.
  hidden_size: int
  intermediate_dim: int

  # The total number of layers and the number of attention heads in each layer.
  num_hidden_layers: int
  num_attention_heads: int

  # The size of the input/output vocabulary, the maximum supported length, and
  # the number of segments (type
  vocab_size: int
  max_length: int  # `max_position_embeddings` in legacy BertConfig.
  num_segments: int  # `type_vocab_size` in legacy BertConfig.

  # Initializers, activations and dtypes for all the layers.
  # Legacy BertConfig has `initializer_range` which can be matched using
  # initializers.truncated_normal(stddev=initializer_range).
  bias_init: Initializer
  kernel_init: Initializer
  layer_norm_epsilon: float
  dtype: jnp.dtype
  # TODO: Support a `hidden_activation` config for the MLP.
  # `hidden_act` in legacy BertConfig.

  dropout_rate: float
  # TODO: Support a `attention_probs_dropout_rate` config.


@dataclasses.dataclass
class BertBaseConfig(BertConfig):
  """BERT Base configuration."""

  hidden_size: int = 768
  intermediate_dim: int = 3072

  num_hidden_layers: int = 12
  num_attention_heads: int = 12

  vocab_size: int = 30522
  max_length: int = 512
  num_segments: int = 2

  bias_init: Initializer = nn.initializers.zeros
  kernel_init: Initializer = initializers.truncated_normal(stddev=0.02)

  layer_norm_epsilon: float = 1e-12
  dtype: jnp.dtype = jnp.float32
  # TODO: Set `hidden_activation` to jax.nn.gelu.

  dropout_rate: float = 0.1
  # TODO: Set `attention_probs_dropout_rate` to 0.1.


@dataclasses.dataclass
class BertLargeConfig(BertConfig):
  """BERT Large configuration."""

  hidden_size: int = 1024
  intermediate_dim: int = 4096

  num_hidden_layers: int = 24
  num_attention_heads: int = 16

  vocab_size: int = 30522
  max_length: int = 512
  num_segments: int = 2

  bias_init: Initializer = nn.initializers.zeros
  kernel_init: Initializer = initializers.truncated_normal(stddev=0.02)

  layer_norm_epsilon: float = 1e-12
  dtype: jnp.dtype = jnp.float32
  # TODO: Set `hidden_activation` to jax.nn.gelu.

  dropout_rate: float = 0.1
  # TODO: Set `attention_probs_dropout_rate` to 0.1.
