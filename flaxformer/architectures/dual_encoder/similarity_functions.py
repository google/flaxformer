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

"""Similarity functions for dual encoder models.

We define a variety of similarity functions for dual encoder models.

Batch similarity functions are computed on all pairs of the left encodings and
right encodings so that the returned similarity matrix satisfies

  S[i,j] = similarity(encodings1[i], encodings2[j]).

Pointwise similarity functions are computed just on the original pairs, and thus
return a vector of the same length as the batch size:

  S[i] = similarity(encodings1[i], encodings2[i]).
"""

from typing import Any, Callable, Iterable, Optional, Tuple, Union

from flax import linen as nn
from flax.linen.linear import default_kernel_init
from jax import lax
import jax.numpy as jnp

from flaxformer.components import dense
from flaxformer.types import Array
from flaxformer.types import DType
from flaxformer.types import Initializer


#============================ Pointwise Similarity =============================
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
  """
  features: Union[Iterable[int], int]
  use_bias: bool = False
  dtype: DType = jnp.float32
  kernel_init: Initializer = default_kernel_init
  bias_init: Initializer = nn.initializers.zeros
  precision: Any = None
  act_fn: str = 'relu'
  use_concat_feature: bool = True
  use_difference_feature: bool = True
  use_product_feature: bool = True
  dropout_factory: Optional[Callable[[], nn.Module]] = None

  def setup(self):
    # TODO: Support multple layers of feed-forward networks.
    self.ffnn_layers = [
        dense.DenseGeneral(
            axis=-1,
            features=self.features,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_axis_names=['embed', 'affinity'],
            precision=self.precision)
    ]
    self.activation = getattr(nn,
                              self.act_fn) if self.act_fn != 'linear' else None
    if self.dropout_factory:
      self.dropout = self.dropout_factory()  # pylint: disable=not-callable

  def __call__(self,
               encodings1: Array,
               encodings2: Array,
               *,
               enable_dropout: bool = True) -> Array:
    """Compute the pointiwse feed-forward NN similarity from two encodings.

    Args:
      encodings1: A 2-D tensor of (left) encodings with shape [batch size,
        encoding dim].
      encodings2: A 2-D tensor of (right) encodings with shape [batch size,
        encoding dim].
      enable_dropout: Whether to enable dropout layers.

    Returns:
      A 1-D tensor of similarities with shape [batch size].
    """
    # TODO: Extend to multiple fully-connected layers.
    inputs = []
    encodings1_dim = encodings1.shape[-1]
    encodings2_dim = encodings2.shape[-1]

    # Optionally add the two encodings as features.
    if self.use_concat_feature:
      inputs += [encodings1, encodings2]

    # If using element-wise features, enforce that the encodings have the same
    # dimension.
    if self.use_difference_feature or self.use_product_feature:
      if encodings1_dim != encodings2_dim:
        raise ValueError(
            'If using element-wise features, enforce that the encodings have the same dimension.'
        )

    # Optionally add the element-wise difference as a feature.
    if self.use_difference_feature:
      inputs += [jnp.abs(encodings1 - encodings2)]

    # Optionally add the element-wise product as a feature.
    if self.use_product_feature:
      inputs += [lax.mul(encodings1, encodings2)]

    inputs = jnp.concatenate(inputs, axis=-1)
    if self.dropout_factory:
      inputs = self.dropout(inputs, deterministic=not enable_dropout)

    logits = self.ffnn_layers[-1](inputs)
    if self.activation:
      logits = self.activation(logits)
    return logits


class DotProduct(nn.Module):
  """Vanilla row-wise version of dot product similarity function."""

  def __call__(self, left_encodings: Array, right_encodings: Array,
               **params) -> Tuple[Array, ...]:
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
    return jnp.sum(left_encodings * right_encodings, axis=-1, keepdims=True)


#============================ Batch Similarity =================================
class BatchDotProduct(nn.Module):
  """Batch version of dot product similarity function."""

  @nn.compact
  def __call__(self,
               left_encodings: Array,
               right_encodings: Array,
               right_additional_encodings: Optional[Array] = None,
               **params) -> Tuple[Array, ...]:
    """Compute the batch dot product similarity from two encodings.

    Args:
      left_encodings: A 2-D tensor of (left) encodings with shape [batch size,
        encoding dim].
      right_encodings: A 2-D tensor of (right) encodings with shape [batch size,
        encoding dim].
      right_additional_encodings: An optional 2-D tensor of (right) additional
        encodings with shape [batch size, encoding dim].
      **params: Hyperparameters dict.

    Returns:
      A 2-D tensor of dot product similarities with shape
      [batch size, batch size].
    """
    if right_additional_encodings is not None:
      right_encodings = jnp.concatenate(
          [right_encodings, right_additional_encodings], axis=0)
    logits = jnp.dot(left_encodings, right_encodings.transpose())

    return logits
