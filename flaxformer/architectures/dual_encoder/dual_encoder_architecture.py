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

"""This file contains "architecture" classes for dual encoder models.

These are combinators which assemble components (L2Norm, MLP, etc.) into
networks.
"""

import inspect
from typing import Callable, Dict, Iterable, Mapping, NamedTuple, Optional, Sequence, Union
from flax import linen as nn
from jax import lax
from jax import random
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


class NonRepeatingDropout(nn.Module):
  """Add non-repeating dropout layer.

  Different from linen's nn.Dropout, this dropout module keeps a "use counter"
  so that it can automatically changing dropout mask based on the counter. This
  requires setting the new dropout variable collection mutable, e.g. in
  model.apply(..., mutable='dropout').
  """
  rate: float
  broadcast_dims: Iterable[int] = ()
  deterministic: Optional[bool] = None

  @nn.compact
  def __call__(self, inputs, deterministic: Optional[bool] = None, rng=None):
    deterministic = nn.merge_param('deterministic', self.deterministic,
                                   deterministic)
    if self.rate == 0.:
      return inputs
    keep_prob = 1. - self.rate
    if deterministic:
      return inputs
    else:
      if rng is None:
        rng = self.make_rng('dropout')
        cntr = self.variable(
            'dropout', 'counter',
            lambda: jnp.array(0))  # <--- autoincrements dropout
        rng = random.fold_in(
            rng, cntr.value)  # requires that we set mutable='dropout' in apply
        cntr.value += 1  #
      broadcast_shape = list(inputs.shape)
      for dim in self.broadcast_dims:
        broadcast_shape[dim] = 1
      mask = random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
      mask = jnp.broadcast_to(mask, inputs.shape)
      return lax.select(mask, inputs / keep_prob, jnp.zeros_like(inputs))


class DualEncoderOutput(NamedTuple):
  left_encoded: Array
  right_encoded: Array
  logits: Union[Array, Dict[str, Array]]


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




def _call_optional(
    fn: Optional[Callable[[], nn.Module]]) -> Optional[nn.Module]:
  return fn() if fn else None


class PassThroughMultimodalEncoder(nn.Module,
                                   param_remapping.ParameterRemappable):
  """An multimodal encoder that passes through the inputs."""
  shared_token_embedder: Optional[embedding.Embed] = None

  def __call__(self,
               batch: Mapping[str, Array],
               enable_dropout: bool = True) ->...:
    """Passes through the inputs.

    The features are embedded separately and concatenated to form the input
    sequence to pass to the Transformer.

    Zero-valued inputs are considered padding when populating the
    self-attention mask.

    Args:
      batch: feature name to values
      enable_dropout: whether dropout is disabled

    Returns:
      triple of (encoded values, encoder mask, encoder segment ids)
    """
    # The passthrough multimodal encoder is supposed to only take in embedding
    # features and ignores the text token inputs.
    batch_encoder = {
        k: v for (k, v) in batch.items() if not k.startswith('targets') and
        not k.endswith('_loss_weights') and not k.endswith('text_tokens')
    }

    # Concatenate embedding features without linearization.
    # TODO: Support optional linearization for input embeddings.
    batch_encoder_embeddings = list(batch_encoder.values())
    # Each encoder_embedding has the shape (num_embeddings, embedding_size).
    batch_encoder_embeddings_lengths = set([
        encoder_embedding.shape[-1]
        for encoder_embedding in batch_encoder_embeddings
    ])
    if len(batch_encoder_embeddings_lengths) != 1:
      raise ValueError(
          'Input embeddings for PassThroughMultimodalEncoder should have the same embedding dimensions!'
      )
    # After concatenation, it becomes (all_num_embeddings, embedding_size).
    encoder_outputs = jnp.concatenate(batch_encoder_embeddings, axis=0)
    encoder_mask = multimodal_feature.attention_mask_for_zeros(
        list(batch_encoder.values()))

    return (encoder_outputs, encoder_mask, None)


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
    multi_logit_layer_factories: This is similar to similarity layer,
    because both are used to create logits. Use similarity layer to create only
    1 logits, where as multi_logit_layer_factories to create more than
    one logits in the form of a dictionary. The different logits can be used to
    compute different losses. similarity_layer_factory takes a single factory,
    where as multi_logit_layer_factories takes a list of factories. The
    same factory functions which are passed to similarity layer can be passed
    here. Please do not set both, if you want to use a similarity layer, just
    append it to this list, and keep the similarity_layer empty while using
    this.
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
  multi_logit_layer_factories: Optional[Sequence[Callable[[],
                                                          nn.Module]]] = None

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

    if self.similarity_layer_factory and self.multi_logit_layer_factories:
      raise ValueError(
          'Both similarity_layer_factory and multi_logit_layer_factories create'
          'layer(s) that compute logits. Do not set values for both as they are'
          'redundant. Similarity layers is used to produce one set of logits as'
          'outputs, where as logit creation layers is used to produce a '
          'dictionary of logits. The simplest fix would be to take your '
          'similarity_layer_factory and add it to the list of'
          'logit_creation_layer_factories in your architecture gin file. (It is'
          'recommended to use only multi_logit_layer_factories)')
    if self.similarity_layer_factory:
      self.similarity_layer = self.similarity_layer_factory()  # pylint: disable=not-callable

    if self.multi_logit_layer_factories:
      self.multi_logit_layers = [
          logit_layer_factory()
          for logit_layer_factory in self.multi_logit_layer_factories  # pylint: disable=not-an-iterable
      ]

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
    # For backward compatibility of teams using self.compute_similarity without
    # passing the logit_creation layer.
    return self.compute_logits(left_encoded, right_encoded,
                               self.similarity_layer, right_negative_encoded,
                               enable_dropout)

  def compute_logits(self,
                     left_encoded: Array,
                     right_encoded: Array,
                     logit_creation_layer: nn.module.Module,
                     right_negative_encoded: Optional[Array] = None,
                     enable_dropout: bool = True) -> Array:

    if check_use_negative_inputs(logit_creation_layer):
      return logit_creation_layer(
          left_encoded,
          right_encoded,
          right_negative_encoded,
          enable_dropout=enable_dropout)

    return logit_creation_layer(
        left_encoded, right_encoded, enable_dropout=enable_dropout)

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
          enable_dropout=enable_dropout)

    if self.similarity_layer_factory is None and self.multi_logit_layer_factories is None:
      raise ValueError(
          'DualEncoder instances without a similarity layer or logit creation layer may only be used for encoding inputs, not comparing them.'
      )

    if right_negative_encoded is not None:
      if self.similarity_layer_factory:
        all_layers = [self.similarity_layer]
      else:
        all_layers = self.multi_logit_layers

      # when right_negative_encoded is being passed, check if any of the layers
      # are making use of them
      negative_inputs_being_used = any(
          check_use_negative_inputs(layer) for layer in all_layers)

      if not negative_inputs_being_used:
        raise ValueError(
            'Negative inputs were provided but none of the'
            'logit_creation_layers or similarity_layers are using them')

    if self.similarity_layer_factory:
      logits = self.compute_logits(
          left_encoded,
          right_encoded,
          self.similarity_layer,
          right_negative_encoded,
          enable_dropout=enable_dropout)
    else:
      # create a dictionary of logits using all logit_creation_layer_factories
      logits = {}
      for logit_creation_layer in self.multi_logit_layers:
        current_logits = self.compute_logits(
            left_encoded,
            right_encoded,
            logit_creation_layer,
            right_negative_encoded,
            enable_dropout=enable_dropout)
        logits[logit_creation_layer.name] = current_logits

    return DualEncoderOutput(left_encoded, right_encoded, logits)


