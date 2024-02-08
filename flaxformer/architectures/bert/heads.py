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

"""This module contains various BERT heads."""

from typing import Optional
from typing import Sequence

from flax import linen as nn
from flax.linen.initializers import zeros
import jax.numpy as jnp

from flaxformer.components import dense
from flaxformer.components import initializers
from flaxformer.types import Activation
from flaxformer.types import Array
from flaxformer.types import DType
from flaxformer.types import Initializer

_DEFAULT_LAYER_NORM = 1e-12
_DEFAULT_INIT_RANGE = 0.02


class BertPooler(nn.Module):
  """Pools the CLS embedding and passes it though the `Dense` layer.

  Attributes:
    kernel_init: Initializer for the dense layer kernel.
    dtype: The dtype of the computation (default: float32).
  """
  kernel_init: Initializer = initializers.truncated_normal(
      stddev=_DEFAULT_INIT_RANGE)
  dtype: DType = jnp.float32

  def _extract_cls_embedding(self, encoded_inputs: Array) -> Array:
    """Slice all tokens embeddings to get CLs embedding."""
    # We need to slice the dimension 1 (counting from zero) of the inputs and
    # extract the embedding at the position 0, which corresponds to the CLS
    # token emebedding. This operations returns the tensor slice of size
    # [batch_size, hidden_size].
    return encoded_inputs[:, 0]

  @nn.compact
  def __call__(self, encoded_inputs: Array, **unused_kwargs):
    """Pools the CLS embedding and applies MLP to it.

    Args:
      encoded_inputs: The inputs (e.g., token's embeddings) that come from the
        final layer of the BERT encoder. <float32>[batch_size, seq_length,
        hidden_size].
      **unused_kwargs: unused.

    Returns:
      An array of logits <float32>[batch_size, hidden_size].
    """
    cls_embedding = self._extract_cls_embedding(encoded_inputs)
    cls_embedding = dense.DenseGeneral(
        features=cls_embedding.shape[-1],
        use_bias=True,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        kernel_axis_names=('embed', 'mlp'),
        name='dense')(
            cls_embedding)
    return nn.tanh(cls_embedding)


class ClassifierHead(nn.Module):
  """Classification head.

  Attributes:
    pooler: An instance of the BertPooler class.
    num_classes: The output layer size, which is the number of classes.
    kernel_init: Initializer for the classifier dense layer kernel.
    dropout_rate: Dropout probability used across all the model layers.
    dtype: The dtype of the computation (default: float32).
    enable_dropout: Enables dropout when set to True.
    use_bias: Use bias or not in the dense layer.
  """
  pooler: BertPooler
  num_classes: int
  kernel_init: Initializer = initializers.truncated_normal(
      stddev=_DEFAULT_INIT_RANGE)
  dropout_rate: float = 0.
  dtype: DType = jnp.float32
  enable_dropout: Optional[bool] = None
  use_bias: bool = True

  def setup(self):
    if self.enable_dropout is not None:
      deterministic = not self.enable_dropout
    else:
      deterministic = None
    self.dropout_layer = nn.Dropout(
        rate=self.dropout_rate, deterministic=deterministic)
    self.dense = dense.DenseGeneral(
        features=self.num_classes,
        use_bias=self.use_bias,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        kernel_axis_names=('embed', 'mlp'),
        name='dense')

  def __call__(self,
               encoded_inputs: Array,
               *,
               enable_dropout: Optional[bool] = None,
               **unused_kwargs) -> Array:
    """Pools the CLS emebdding and projects into the logits.

    Args:
      encoded_inputs: The inputs (e.g., token's embeddings) that come from the
        final layer of the BERT encoder. <float32>[batch_size, seq_length,
        hidden_size].
      enable_dropout: Enables dropout when set to True.
      **unused_kwargs: unused.

    Returns:
      An array of logits <float32>[batch_size, num_classes].
    """
    if enable_dropout is not None:
      deterministic = not enable_dropout
    else:
      deterministic = None
    cls_embedding = self.pooler(encoded_inputs)
    cls_embedding = self.dropout_layer(cls_embedding, deterministic)
    logits = self.dense(cls_embedding)
    return logits


class NSPHead(nn.Module):
  """Next sentence prediction head.

  Attributes:
    pooler: An instance of the BertPooler class.
    kernel_init: Initializer for the classifier dense layer kernel.
    dtype: The dtype of the computation (default: float32).
  """
  pooler: BertPooler
  kernel_init: Initializer = initializers.truncated_normal(
      stddev=_DEFAULT_INIT_RANGE)
  dtype: DType = jnp.float32

  def setup(self):
    self.mlp = dense.DenseGeneral(
        features=2,
        use_bias=True,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        kernel_axis_names=('embed', 'mlp'),
        name='dense')

  def __call__(self, encoded_inputs: Array, **unused_kwargs) -> Array:
    """Pools the CLS embedding and projects it into 2 logits.

    Args:
      encoded_inputs: The inputs (e.g., token's embeddings) that come from the
        final layer of the BERT encoder. <float32>[batch_size, seq_length,
        hidden_size].
      **unused_kwargs: unused.

    Returns:
      An array of logits <float32>[batch_size, 2].
    """
    cls_embedding = self.pooler(encoded_inputs)
    return self.mlp(cls_embedding)


def gather_indices(inputs: Array, indices: Array) -> Array:
  """Gathers the vectors at the specific indices over a minibatch.

  Example:

    inputs = [[[0], [1], [2]],
              [[3], [4], [5]],
              [[6], [7], [8]]]

    indices = [[0, 1],
               [1, 2],
               [0, 2]]

    gather_indices(inputs, indices) = [[[0], [1]],
                                       [[4], [5]],
                                       [[6], [8]]]

  Args:
    inputs: A 3-D input array shaped [batch_size, seq_length, features].
    indices: A 2-D indices array with the positions that need to be selected,
      with shape <int>[batch_size, indices_seq_length].

  Returns:
    The inputs, but only those positions (on axis 1) that are given by the
    indices.
  """
  # We can't index a 3D array with a 2D array, so we have to flatten the inputs.
  # This way, we are indexing a 2D input array with a 1D indices array, and then
  # we reshape it back.
  batch_size, seq_length, features = inputs.shape
  flat_offsets = (jnp.arange(batch_size) * seq_length).reshape([-1, 1])
  flat_indices = (indices + flat_offsets).reshape([-1])
  flat_inputs = inputs.reshape([batch_size * seq_length, features])
  gathered_inputs = jnp.take(flat_inputs, flat_indices, axis=0, mode='clip')
  # Reshape back into [batch_size, indices_seq_length, features].
  return gathered_inputs.reshape([batch_size, -1, features])


class MLMHead(nn.Module):
  """Masked Language Model head.

  Attributes:
    embed: `Embed` module of the BertEncoder for token_ids. We need it to
      extract the word embeddings. See tests on how to access this submodule.
    hidden_size: The output layer size, which is the number of classes.
    vocab_size: The vocabulary size.
    kernel_init: Initializer for the classifier dense layer kernel.
    dropout_rate: Dropout probability used across all the model layers.
    activation: Activation function.
    dtype: The dtype of the computation (default: float32).
  """
  encoder: nn.Module
  hidden_size: int
  vocab_size: int
  kernel_init: Initializer = initializers.truncated_normal(
      stddev=_DEFAULT_INIT_RANGE)
  dropout_rate: float = 0.
  activation: Activation = nn.gelu
  dtype: DType = jnp.float32

  def setup(self):
    self.mlm_hidden_layer = dense.DenseGeneral(
        features=self.hidden_size,
        kernel_init=self.kernel_init,
        dtype=self.dtype,
        use_bias=True,
        kernel_axis_names=('embed', 'mlp'),
        name='dense')
    self.layer_norm = nn.LayerNorm(
        epsilon=_DEFAULT_LAYER_NORM, name='layer_norm')
    self.bias = self.param('bias', zeros, (self.vocab_size,))

  def __call__(self, encoded_inputs: Array, *,
               masked_positions: Optional[Array]) -> Array:
    """Transforms the encodings and computes logits for each token.

    Args:
      encoded_inputs: The inputs (e.g., token representations) that come from
        the final layer of the BERT encoder. <float32>[batch_size, seq_length,
        hidden_size].
      masked_positions: The positions on which to apply the MLM head. Typically
        only 15% of the positions are masked out, and we don't want to predict
        other positions to save computation. This array may contain padding
        values, and could be shorter than `encoded_inputs`.

    Returns:
      Predicted logits across vocab for each token in the sentence
      <float32>[batch_size, seq_length, vocab_size].
    """
    if masked_positions is not None:
      # Only predict for the provided masked positions.
      masked_out_inputs = gather_indices(encoded_inputs, masked_positions)
    else:
      masked_out_inputs = encoded_inputs  # Predict for all positions.
    mlm_hidden = self.mlm_hidden_layer(masked_out_inputs)
    mlm_hidden_activated = self.activation(mlm_hidden)
    mlm_hidden_normalized = self.layer_norm(mlm_hidden_activated)
    embedder = self.encoder.embedder.embedders['token_ids']
    mlm_decoded = embedder.attend(mlm_hidden_normalized)
    mlm_decoded += self.bias
    return mlm_decoded


class MLP(nn.Module):
  """Multi-layer perceptron.

  An MLP with a variable amount of hidden layers and configurable activations.

  Attributes:
    features: Sequence of model layer sizes.
    kernel_init: Initializer for the classifier dense layer kernel.
    dropout_rate: Dropout probability used across all the model layers.
    activations: Activation functions to be used after the hidden layers.
      Activations are applied to each intermediate hidden layer, but not to the
      last one, hence the length of this argument should be the length of
      `features` minus one. If set to None, will apply a gelu to intermediate
      layers.
    enable_dropout: Enables dropout when set to True.
    dtype: The dtype of the computation (default: float32).
    use_bias: Use bias or not in the dense layer.
  """
  features: Sequence[int]
  kernel_init: Initializer = initializers.truncated_normal(
      stddev=_DEFAULT_INIT_RANGE)
  dropout_rate: float = 0.
  activations: Optional[Sequence[Activation]] = None
  enable_dropout: Optional[bool] = None
  dtype: DType = jnp.float32
  use_bias: bool = True

  @nn.compact
  def __call__(self,
               inputs: Array,
               *,
               enable_dropout: Optional[bool] = None,
               **unused_kwargs) -> Array:
    """Applies the MLP to the inputs.

    Args:
      inputs: The model inputs. <float32>[batch_size, seq_length, hidden_size].
      enable_dropout: Enables dropout when set to True.
      **unused_kwargs: unused.

    Returns:
      The output of the model, of size <float32>[batch_size, seq_length,
      num_classes].
    """
    if enable_dropout is not None:
      deterministic = not enable_dropout
    elif self.enable_dropout is not None:
      deterministic = not self.enable_dropout
    else:
      deterministic = True

    activations = self.activations
    if activations is None:
      activations = [nn.gelu] * (len(self.features) - 1)
    elif len(self.activations) != len(self.features) - 1:
      raise ValueError('`activations` must be of length `len(features) - 1`. '
                       f'Got {len(self.activations)}, expected '
                       f'{len(self.features) - 1}.')

    x = inputs
    for i, feat in enumerate(self.features):
      x = dense.DenseGeneral(
          features=feat,
          kernel_init=self.kernel_init,
          dtype=self.dtype,
          use_bias=self.use_bias,
          kernel_axis_names=('embed', 'mlp'),
          name=f'dense_{i}')(
              x)
      if i != len(self.features) - 1:
        x = activations[i](x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)

    return x


class TokenClassifierHead(nn.Module):
  """Token classification head.

  A classification head that can be used for per-token classification (i.e.
  sequence classification) tasks, such as POS tagging, NER, and BIO tagging.

  Attributes:
    features: Sequence of MLP layer sizes.
    kernel_init: Initializer for the classifier dense layer kernel.
    dropout_rate: Dropout probability used across all the model layers.
    activations: Activation functions to be used after the MLP hidden layers,
      except for the final layer.
    enable_dropout: Enables dropout when set to True.
    dtype: The dtype of the computation (default: float32).
    use_bias: Use bias or not in the dense layer.
  """
  features: Sequence[int]
  kernel_init: Initializer = initializers.truncated_normal(
      stddev=_DEFAULT_INIT_RANGE)
  dropout_rate: float = 0.
  activations: Optional[Sequence[Activation]] = None
  enable_dropout: Optional[bool] = None
  dtype: DType = jnp.float32
  use_bias: bool = True

  @nn.compact
  def __call__(self,
               encoded_inputs: Array,
               *,
               enable_dropout: Optional[bool] = None,
               **unused_kwargs) -> Array:
    """Transforms the encodings and computes logits for each token.

    Args:
      encoded_inputs: The inputs (e.g., token representations) that come from
        the final layer of the BERT encoder. <float32>[batch_size, seq_length,
        hidden_size].
      enable_dropout: Enables dropout when set to True.
      **unused_kwargs: unused.

    Returns:
      Predicted logits across classes for each token in the sentence
      <float32>[batch_size, seq_length, num_classes].
    """
    return MLP(
        self.features,
        self.kernel_init,
        self.dropout_rate,
        self.activations,
        self.enable_dropout,
        self.dtype,
        self.use_bias,
        name='mlp')(
            inputs=encoded_inputs, enable_dropout=enable_dropout)
