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

"""Similarity functions for dual encoder models.

We define a variety of similarity functions for dual encoder models.

Batch similarity functions are computed on all pairs of the left encodings and
right encodings so that the returned similarity matrix satisfies

  S[i,j] = similarity(encodings1[i], encodings2[j]).

Pointwise similarity functions are computed just on the original pairs, and thus
return a vector of the same length as the batch size:

  S[i] = similarity(encodings1[i], encodings2[i]).
"""

from typing import Any, Callable, Iterable, Optional

from flax import linen as nn
from flax.linen.linear import default_kernel_init
from jax import lax
import jax.numpy as jnp

from flaxformer.components import dense
from flaxformer.components.attention import dense_attention
from flaxformer.types import Array
from flaxformer.types import DType
from flaxformer.types import Initializer


# ============================ Pointwise Similarity ============================
class PointwiseFFNN(nn.Module):
  """Pointwise feed-forward NN similarity functions.

  The two encodings are concatenated and then fed into a fully-connected layers
  to produce the similarity.

  Optionally, other features of the two encodings can be computed: element-wise
  difference and element-wise product (see the InferSent paper:
  https://arxiv.org/abs/1705.02364).

  Attributes:
    features: tuple with numbers of output features.
    use_bias: whether to add a bias to the output (default: False).
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    use_concat_feature: Whether add the two encodings.
    use_difference_feature: Whether add the difference of two encodings.
    use_product_feature: Whether add the product of two encodings.
    dropout_factory: A callable that returns a new dropout instance. This is
      applied after the feature concatenation.
    intermediate_features: An iterable containing dimensions for intermediate
      layers. These are the hidden layers before the last hidden layer.
    intermediate_act_fn: An activation function for the hidden layers.
  """
  features: Iterable[int] | int
  use_bias: bool = False
  dtype: DType = jnp.float32
  kernel_init: Initializer = default_kernel_init  # pytype: disable=annotation-type-mismatch  # jax-types
  bias_init: Initializer = nn.initializers.zeros
  precision: Any = None
  act_fn: str = 'relu'
  use_concat_feature: bool = True
  use_difference_feature: bool = True
  use_product_feature: bool = True
  dropout_factory: Optional[Callable[[], nn.Module]] = None
  intermediate_features: Optional[Iterable[int] | int] = None
  intermediate_act_fn: str = 'relu'

  def _build_layer(self, f):
    return dense.DenseGeneral(
        axis=-1,
        features=f,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        kernel_axis_names=['embed', 'affinity'],
        precision=self.precision)

  def setup(self):
    layer_features = self.intermediate_features or []
    ffnn_layers = []

    # Build the layers
    for f in layer_features:
      ffnn_layers.append(self._build_layer(f))
    ffnn_layers.append(self._build_layer(self.features))
    self.ffnn_layers = ffnn_layers

    # dropout
    if self.dropout_factory:
      self.dropout = self.dropout_factory()  # pylint: disable=not-callable
    else:
      self.dropout = None

    # intermediate activations
    if self.intermediate_act_fn != 'linear':
      self.intermediate_activation = getattr(nn, self.intermediate_act_fn)
    else:
      self.intermediate_activation = None

    # final activation
    if self.act_fn != 'linear':
      self.final_activation = getattr(nn, self.act_fn)
    else:
      self.final_activation = None

  def __call__(self,
               encodings1: Array,
               encodings2: Optional[Array] = None,
               *,
               enable_dropout: bool = True) -> Array:
    """Compute the pointiwse feed-forward NN similarity from 1 or 2 encodings.

    Args:
      encodings1: A 2-D tensor of (left) encodings with shape [batch size,
        encoding dim].
      encodings2: An optional 2-D tensor of (right) encodings with shape [batch
        size, encoding dim].
      enable_dropout: Whether to enable dropout layers.

    Returns:
      A 1-D tensor of similarities with shape [batch size].
    """
    inputs = []
    encodings1_dim = encodings1.shape[-1]
    if encodings2 is not None:
      encodings2_dim = encodings2.shape[-1]

      # Optionally add the two encodings as features.
      if self.use_concat_feature:
        inputs += [encodings1, encodings2]

      # If using element-wise features, enforce that the encodings have the same
      # dimension.
      if self.use_difference_feature or self.use_product_feature:
        if encodings1_dim != encodings2_dim:
          raise ValueError(
              'If using element-wise features, enforce that the encodings have '
              'the same dimension. The dimensions are: encodings1_dim: %d, '
              'encodings2_dim: %d' % (encodings1_dim, encodings2_dim))

      # Optionally add the element-wise difference as a feature.
      if self.use_difference_feature:
        inputs += [jnp.abs(encodings1 - encodings2)]

      # Optionally add the element-wise product as a feature.
      if self.use_product_feature:
        inputs += [lax.mul(encodings1, encodings2)]
    else:
      inputs = [encodings1]

    inputs = jnp.concatenate(inputs, axis=-1)
    if self.dropout_factory:
      inputs = self.dropout(inputs, deterministic=not enable_dropout)

    # Pass through the hidden layers
    for layer in self.ffnn_layers[:-1]:
      inputs = layer(inputs)
      if self.intermediate_activation:
        inputs = self.intermediate_activation(inputs)
      if self.dropout:
        inputs = self.dropout(inputs, deterministic=not enable_dropout)

    # Pass through the final layer
    logits = self.ffnn_layers[-1](inputs)
    if self.final_activation:
      logits = self.final_activation(logits)

    return logits


class DotProduct(nn.Module):
  """Vanilla row-wise version of dot product similarity function."""

  def __call__(
      self, left_encodings: Array, right_encodings: Array, **params
  ) -> tuple[Array, ...]:
    """Computes the point-wise product similarity from two encodings.

    Args:
      left_encodings: A 2-D tensor of (left) encodings with shape [batch size,
        encoding dim].
      right_encodings: A 2-D tensor of (right) encodings with shape [batch size,
        encoding dim].
      **params: Hyperparameters dict.

    Returns:
      A 2-D tensor of dot product similarities with shape [batch size, 1].
    """
    # Implement the dot product as module to be consistent to other similarity
    # functions.
    del self
    return jnp.sum(left_encodings * right_encodings, axis=-1, keepdims=True)  # pytype: disable=bad-return-type  # jnp-type


# ============================ Batch Similarity ================================
class BatchDotProduct(nn.Module):
  """Batch version of dot product similarity function."""
  use_only_explicit_hard_negatives: bool = False

  @nn.compact
  def __call__(
      self,
      left_encodings: Array,
      right_encodings: Array,
      right_additional_encodings: Optional[Array] = None,
      **params,
  ) -> Array:
    """Compute the batch dot product similarity from two encodings.

    Args:
      left_encodings: A 2-D tensor of (left) encodings with shape [batch size,
        encoding dim].
      right_encodings: A 2-D tensor of (right) encodings with shape [batch size,
        encoding dim].
      right_additional_encodings: An optional 2-D tensor of (right) additional
        encodings with shape [batch_size * num_hard_negatives, encoding_dim].
      **params: Hyperparameters dict.

    Returns:
      logits: A 2-D tensor of dot product similarities. If
        right_additional_encodings are provided, then the output shape is
        [batch_size, batch_size + num_hard_negatives] if
        use_only_explicit_hard_negatives is True, and
        [batch_size, batch_size * (1 + num_hard_negatives)] if
        use_only_explicit_hard_negatives is False. If right_additional_encodings
        are not provided, then the output shape is [batch_size, batch_size].
    """
    if self.use_only_explicit_hard_negatives:
      # Compute in-batch logits of shape [batch_size, batch_size].
      logits = jnp.dot(left_encodings, right_encodings.transpose())
      if right_additional_encodings is not None:
        batch_size, encoding_dim = left_encodings.shape
        right_additional_encodings = right_additional_encodings.reshape(
            [batch_size, -1, encoding_dim]
        )
        # Logits for explicitly provided hard negatives. The shape
        # is [batch_size, num_hard_negatives].
        additional_logits = jnp.sum(
            left_encodings[:, jnp.newaxis, :] * right_additional_encodings,
            axis=-1,
        )
        # Final logits of shape [batch_size, batch_size + num_hard_negatives].
        logits = jnp.concatenate([logits, additional_logits], axis=-1)
    else:
      if right_additional_encodings is not None:
        right_encodings = jnp.concatenate(
            [right_encodings, right_additional_encodings], axis=0
        )
      # Final logits. Each examples uses all other hard negatives in the batch,
      # so shape is [batch_size, batch_size * (1 + num_hard_negatives)].
      logits = jnp.dot(left_encodings, right_encodings.transpose())

    return logits  # pytype: disable=bad-return-type  # jnp-type


class DoNothing(nn.Module):
  """A do-nothing similarity function.

  This is useful if we want to just take the embeddings and compute the losses
  outside the forward module.
  """

  @nn.compact
  def __call__(
      self,
      left_encodings: Array,
      right_encodings: Array,
      right_additional_encodings: Optional[Array] = None,
      **params,
  ) -> tuple[Array, ...]:
    """Compute the batch dot product similarity from two encodings.

    Args:
      left_encodings: Unused. A 2-D tensor of (left) encodings with shape [batch
        size, encoding dim].
      right_encodings: Unused. A 2-D tensor of (right) encodings with shape
        [batch size, encoding dim].
      right_additional_encodings: Unused. An optional 2-D tensor of (right)
        additional encodings with shape [batch size, encoding dim].
      **params: Unused. Hyperparameters dict.

    Returns:
      A single 0.
    """
    del left_encodings
    del right_encodings
    del right_additional_encodings
    del params
    return jnp.zeros((), dtype=jnp.int32)  # pytype: disable=bad-return-type  # jnp-type


class BatchAttentionSimilarity(nn.Module):
  """Batched attention-based similiarity score.

  Attributes:
    attention: the attention module.
    mlp_layer: the MLP module, applied after attention.
    layer_norm_factory:  A callable that returns a new layer norm. This is
      applied before the attention module and before the MLP.
    activation_fn: An activation function for the hidden layers. Default to
      'linear', where no activation is requested.
    dropout_factory:  An optional callable that returns a new dropout instance.
      If it exists, it is applied after the attention module.
  """

  attention: nn.Module
  mlp_layer: nn.Module
  layer_norm_factory: Callable[[], nn.Module]
  activation_fn: str = 'linear'
  dropout_factory: Optional[Callable[[], nn.Module]] = None

  def setup(self):
    assert self.mlp_layer.out_dim == 1, (
        'Requires mlp layer to return a Tensor with output dimension 1. '
        f'Currently, mlp_layer.out_dim={self.mlp_layer.out_dim}'
    )
    self.pre_attention_layer_norm = self.layer_norm_factory()
    self.pre_mlp_layer_norm = self.layer_norm_factory()

    # Optional dropout module.
    if self.dropout_factory:
      self.dropout = self.dropout_factory()  # pylint: disable=not-callable
    else:
      self.dropout = None

    if self.activation_fn == 'linear':
      self.activation = None
    else:
      self.activation = getattr(nn, self.activation_fn)

  def __call__(
      self,
      encoded_input_1: Array,
      encoded_input_2: Array,
      encoded_input_mask_1: Array,
      encoded_input_mask_2: Array,
      *,
      pointwise_similarity: bool = True,
      enable_dropout: bool = True,
      **_,
  ) -> Array:
    """Computes the attention based similarity from two encodings.

    encoded_input_1 and encoded_input_2 are the encoded inputs from
    the two encoder towers. They share the same batch size and encoding
    dimension. If the left and right towers are asymmetric, or generate
    embeddings of different dimensionality, please add projection layers
    to map them into the same dimsion.

    Args:
      encoded_input_1: A 3-D tensor of the shape [batch_size, sequence_length_1,
        encoding_dim].
      encoded_input_2: A 3-D tensor of the shape [batch_size, sequence_length_2,
        encoding_dim].
      encoded_input_mask_1: A 2-D tensor of the shape [batch_size,
        sequence_length_1], as a binary mask for the non-padded tokens of
        encoded_input_1.
      encoded_input_mask_2: A 2-D tensor of the shape [batch_size,
        sequence_length_2], as a binary mask for the non-padded tokens of
        encoded_input_2.
      pointwise_similarity: a bool indicaing whether to return pointwise
        similarity only. Default to True. If False, it allows encoded_input_1
        and encoded_input_2 have different batch sizes, and returns a score
        matrix of the shape [batch_size_1, batch_size_2], with elements being
        Similarity(encoded_input_1[i], encoded_input_2[j]), where i in [0,
        batch_size_1) and j in [0, batch_size_2). If True, it requires the batch
        sizes of encoded_input_1 and encoded_input_2 to be the same, and returns
        a vector of Similarity(encoded_input_1[i], encoded_input_2[i]), where i
        in [0, batch_size).
      enable_dropout: a bool indicating whether to use dropout.

    Returns:
      similarity_tensor: a float tensor with similarity score.
        If pointwise_similarity = True, the tensor is of the shape [batch_size],
        otherwise [batch_size_1, batch_size_2].
    """
    batch_size_1 = encoded_input_1.shape[0]
    batch_size_2 = encoded_input_2.shape[0]
    if pointwise_similarity:
      assert batch_size_1 == batch_size_2, (
          'pointwise similarity requires both inputs have the same batch. '
          f'Current shape are {encoded_input_1.shape} and '
          f'{encoded_input_2.shape}.'
      )
    else:
      input_ndim = jnp.ndim(encoded_input_1)
      # To calculate the similarity between the i-th element in encoded_input_1
      # and the j-th element in encoded_input_2, we need to tile / repeat the
      # encoded_input_1/2.
      # The encoded_input_1 is tiled along the batch dimension by batch_size_2.
      # e.g. encoded_input_1 has 3 elements [[1-st], [2-nd], [3-rd]], and
      # batch_size_2 = 2. The tiled results will be
      # [[1-st], [2-nd], [3-rd], [1-st], [2-nd], [3-rd]].
      encoded_input_1 = jnp.tile(
          encoded_input_1, (batch_size_2,) + (1,) * (input_ndim - 1)
      )
      encoded_input_mask_1 = jnp.tile(
          encoded_input_mask_1,
          (batch_size_2,) + (1,) * (jnp.ndim(encoded_input_mask_1) - 1),
      )
      # The encoded_input_2 is repeated along the batch dimension by
      # batch_size_1. e.g. encoded_input_2 has 2 elements [[1-st], [2-nd]], and
      # batch_size_1 = 3. The tiled results will be
      # [[1-st], [2-nd],[1-st], [2-nd], [1-st], [2-nd]].
      encoded_input_2 = jnp.repeat(encoded_input_2, batch_size_1, axis=0)
      encoded_input_mask_2 = jnp.repeat(
          encoded_input_mask_2, batch_size_1, axis=0
      )

    dtype = encoded_input_1.dtype

    encoded_input_1 = self.pre_attention_layer_norm(encoded_input_1)
    encoded_input_2 = self.pre_attention_layer_norm(encoded_input_2)

    # Mask to remove padding tokens.
    encoded_mask = dense_attention.make_attention_mask(
        encoded_input_mask_1, encoded_input_mask_2, dtype=dtype
    )

    # attention map returns a tensor of the shape [batch_size,
    # sequence_length_1, embedding_dim]
    mlp_input = self.attention(
        encoded_input_1,
        encoded_input_2,
        encoded_mask,
        enable_dropout=enable_dropout,
    )

    if self.dropout is not None and enable_dropout:
      mlp_input = self.dropout(mlp_input, deterministic=not enable_dropout)

    mlp_input = self.pre_mlp_layer_norm(mlp_input)

    # mlp_layer returns [batch_size, sequence_length_1, 1]
    logits = self.mlp_layer(mlp_input, enable_dropout=enable_dropout)

    logits = jnp.reshape(logits, encoded_input_mask_1.shape)

    avg_logits = jnp.sum(logits * encoded_input_mask_1, axis=1) / jnp.sum(
        encoded_input_mask_1, axis=1
    )
    if self.activation:
      avg_logits = self.activation(avg_logits)

    if not pointwise_similarity:
      # In case of element-wise similarity, the returned tensor will be a metrix
      # with [i, j]-th element being similarity(encoded_input_1[i],
      # encoded_input_2[j]).
      avg_logits = jnp.reshape(
          avg_logits, [batch_size_1, batch_size_2], order='F'
      )

    return avg_logits


