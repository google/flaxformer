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

"""Architecture classes for dual encoder models.

These are combinators which assemble components (L2Norm, MLP, etc.) into
networks.
"""

import inspect
from typing import Callable, Mapping, Optional, Union

from flax import linen as nn
import flax.struct
import jax.numpy as jnp
from typing_extensions import Protocol

from flaxformer.architectures.common import param_remapping
from flaxformer.architectures.t5 import t5_architecture as flaxformer_t5_architecture
from flaxformer.components import embedding
from flaxformer.components.attention import dense_attention
from flaxformer.types import Array
from flaxformer.types import DType


def check_use_negative_inputs(logit_creation_layer: nn.Module) -> bool:
  call_args = inspect.signature(logit_creation_layer).parameters
  return 'right_additional_encodings' in call_args


@flax.struct.dataclass
class DualEncoderOutput:
  """Outputs of T5X retrieval DualEncoder architecture."""

  left_encoded: Array | Mapping[str, Array]
  right_encoded: Array | Mapping[str, Array]
  logits: Array | Mapping[str, Array]


class MakeEncoderFn(Protocol):
  """Signature for functions that will make a low-level Encoder."""

  def __call__(
      self,
      *,
      shared_token_embedder: Optional[embedding.Embed] = None,
  ) -> flaxformer_t5_architecture.Encoder:
    """Makes a low-level Encoder instance.

    Args:
      shared_token_embedder: Shared token embedder instance, which should be
        passed to the returned module. If this is non-None, you should use it
        instead of providing your own token embedder.

    Returns:
      Encoder instance.
    """




class DualEncoder(nn.Module, param_remapping.ParameterRemappable):
  """Dual encoder model.

  The left tower and the right tower share parameters.

  Attributes:
    encoder_factory: Factory which will make the lower-level Encoder object. If
      shared_token_embedder_factory is non-None, then the result of it will be
      passed as the `shared_token_embedder` argument to `encoder_factory`.
    shared_token_embedder_factory: A callable that returns an embedder that can
      be shared between the encoder and decoder.
    pooler_factory: Optional specialization of encoder output pooling layer.
    l2_norm_factory: Optional specialization of encoder output normalization.
    projection_layer_factory: Optional specialization of encoder output
      projection layer.
    similarity_layer_factory: Optional specialization of encoder output
      similarity layer.
    dtype: DType for dual encoder to cast embedded inputs, and for attention
      mask generation.
  """
  # Core components: shared token embedder and low-level encoder.
  encoder_factory: MakeEncoderFn
  shared_token_embedder_factory: Optional[Callable[[], embedding.Embed]] = None
  pooler_factory: Optional[Callable[[], nn.Module]] = None
  l2_norm_factory: Optional[Callable[[], nn.Module]] = None
  projection_layer_factory: Optional[Callable[[], nn.Module]] = None
  similarity_layer_factory: Optional[Callable[[], nn.Module]] = None

  # Configures behavior when the model is called. Many of these might eventually
  # be better as call parameters.
  dtype: DType = jnp.float32

  def setup(self):
    self.token_embedder = (
        self.shared_token_embedder_factory()  # pylint: disable=not-callable
        if self.shared_token_embedder_factory else None)
    self.encoder = self.encoder_factory(
        shared_token_embedder=self.token_embedder)

    if self.pooler_factory:
      self.pooler = self.pooler_factory()  # pylint: disable=not-callable

    if self.l2_norm_factory:
      self.l2_norm = self.l2_norm_factory()  # pylint: disable=not-callable

    if self.projection_layer_factory:
      self.projection_layer = self.projection_layer_factory()  # pylint: disable=not-callable

    if self.similarity_layer_factory:
      self.similarity_layer = self.similarity_layer_factory()  # pylint: disable=not-callable

  def encode(self,
             encoder_input_tokens: jnp.ndarray,
             encoder_segment_ids=None,
             encoder_positions=None,
             *,
             enable_dropout: bool = True) -> Array:
    """Applies Transformer encoder-branch on the inputs.

    Args:
      encoder_input_tokens: input data to the encoder.
      encoder_segment_ids: encoder input segmentation info for packed examples.
      encoder_positions: encoder input subsequence positions for packed
        examples.
      enable_dropout: Enables dropout if set to True.

    Returns:
      encoded feature array from the transformer encoder.
    """
    # Make padding attention mask.
    encoder_mask = dense_attention.make_attention_mask(
        encoder_input_tokens > 0, encoder_input_tokens > 0, dtype=self.dtype)
    # Add segmentation block-diagonal attention mask if using segmented data.
    if encoder_segment_ids is not None:
      encoder_mask = dense_attention.combine_masks(
          encoder_mask,
          dense_attention.make_attention_mask(
              encoder_segment_ids,
              encoder_segment_ids,
              jnp.equal,
              dtype=self.dtype))

    encoded = self.encoder(  # pytype: disable=attribute-error
        encoder_input_tokens,
        inputs_positions=encoder_positions,
        encoder_mask=encoder_mask,
        enable_dropout=enable_dropout)

    if self.pooler_factory:
      input_masks = jnp.array(encoder_input_tokens > 0, jnp.float32)
      encodings = self.pooler(
          encoded, input_masks, deterministic=not enable_dropout)
    else:
      # Fallback to use first token.
      encodings = encoded[:, 0, :]

    if self.projection_layer_factory:
      projection_output = self.projection_layer(encodings)
    else:
      projection_output = encodings

    if self.l2_norm_factory:
      encoded = self.l2_norm(projection_output)
    else:
      encoded = projection_output

    return encoded

  @property
  def encoder_embedder(self) -> embedding.MultiEmbed:
    return self.encoder.embedder

  def compute_similarity(self,
                         left_encoded: Array,
                         right_encoded: Array,
                         right_negative_encoded: Optional[Array] = None,
                         enable_dropout: bool = True) -> Array:
    if right_negative_encoded is not None:
      if not check_use_negative_inputs(self.similarity_layer):
        raise ValueError(
            'Received negative inputs but similarity layer'
            f' "{self.similarity_layer.name}" does not support negative inputs.'
        )
      return self.similarity_layer(
          left_encoded,
          right_encoded,
          right_negative_encoded,
          enable_dropout=enable_dropout,
      )
    return self.similarity_layer(
        left_encoded, right_encoded, enable_dropout=enable_dropout
    )

  def __call__(self,
               left_encoder_input_tokens,
               right_encoder_input_tokens,
               right_negative_encoder_input_tokens=None,
               left_encoder_segment_ids=None,
               right_encoder_segment_ids=None,
               right_negative_encoder_segment_ids=None,
               left_encoder_positions=None,
               right_encoder_positions=None,
               right_negative_encoder_positions=None,
               *,
               enable_dropout: bool = True) -> DualEncoderOutput:
    """Applies Dual Encoder model on the inputs.

    Args:
      left_encoder_input_tokens: input data to the left encoder.
      right_encoder_input_tokens: input data to the right encoder.
      right_negative_encoder_input_tokens: input negative data to the right
        encoder.
      left_encoder_segment_ids: left encoder segmentation info for packed
        examples.
      right_encoder_segment_ids: right encoder segmentation info for packed
        examples.
      right_negative_encoder_segment_ids: right encoder segmentation info for
        packed negative examples.
      left_encoder_positions: left encoder subsequence positions for packed
        examples.
      right_encoder_positions: right encoder subsequence positions for packed
        examples.
      right_negative_encoder_positions: right encoder subsequence positions for
        packed negative examples.
      enable_dropout: Enables dropout if set to True.

    Returns:
      encodings and similarity scores from the dual encoder.
    """
    if self.similarity_layer_factory is None:
      raise ValueError(
          'DualEncoder instances without a similarity layer may only be used'
          ' for encoding inputs, not comparing them.'
      )

    left_encoded = self.encode(
        left_encoder_input_tokens,
        encoder_segment_ids=left_encoder_segment_ids,
        encoder_positions=left_encoder_positions,
        enable_dropout=enable_dropout)

    right_encoded = self.encode(
        right_encoder_input_tokens,
        encoder_segment_ids=right_encoder_segment_ids,
        encoder_positions=right_encoder_positions,
        enable_dropout=enable_dropout)

    right_negative_encoded = None
    if right_negative_encoder_input_tokens is not None:
      right_negative_encoded = self.encode(
          right_negative_encoder_input_tokens,
          encoder_segment_ids=right_negative_encoder_segment_ids,
          encoder_positions=right_negative_encoder_positions,
          enable_dropout=enable_dropout,
      )

    logits = self.compute_similarity(
        left_encoded,
        right_encoded,
        right_negative_encoded,
        enable_dropout=enable_dropout,
    )

    return DualEncoderOutput(left_encoded, right_encoded, logits)


