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

"""This module contains a BERT model implementation including its layers."""

from __future__ import annotations

from typing import Optional, Tuple
import chex
from flax import linen as nn
import jax.numpy as jnp

from flaxformer import transformer_common as common
from flaxformer.architectures.bert import heads
from flaxformer.components import dense
from flaxformer.components import embedding
from flaxformer.components import initializers
from flaxformer.components.attention import dense_attention
from flaxformer.types import Activation
from flaxformer.types import Array
from flaxformer.types import DType
from flaxformer.types import Initializer

_DEFAULT_LAYER_NORM = 1e-12
_DEFAULT_INIT_RANGE = 0.02


class FullEncoder(nn.Module):
  """An encoder that embeds inputs and then encodes those representations.

  Note that the submodules are responsible for their own dropout and layer norm.
  """
  embedder_block: EmbedderBlock
  encoder_block: EncoderBlock

  def __call__(self,
               input_ids: Array,
               *,
               input_mask: Array,
               position_ids: Optional[Array] = None,
               segment_ids: Optional[Array] = None,
               enable_dropout: bool = True) -> Array:
    """Embeds the inputs and then encodes those representations.

    Args:
      input_ids: <int>[batch..., seq_len]. The first position in the sequence
        should correspond to a "beginning of input" symbol, generally `[CLS]`.
      input_mask: <int>[batch..., seq_len]. Indicates which positions in
        `input_ids` are non-padding (0 for padding, 1 otherwise).
      position_ids: <int>[batch..., seq_len]. The position of each input ID
        within its sequence. This is typically just `range(seq_len)` for each
        sequence, but custom behavior may be desired if, for instance, multiple
        examples are packed into each sequence.
      segment_ids: <int>[batch..., seq_len]. Indicates the "type" of each input
        position. For a traditional BERT-style model with two segments, valid
        values would be {0, 1}.
      enable_dropout: Enables dropout when set to True.

    Returns:
      <float>[batch..., seq_len, hidden_size].
    """
    # Validate inputs and create variables for dimension sizes.
    chex.assert_shape(input_ids, (..., None))
    chex.assert_type(input_ids, int)
    *batch_sizes, seq_len = input_ids.shape
    chex.assert_shape(input_mask, (*batch_sizes, seq_len))
    if position_ids is not None:
      chex.assert_shape(position_ids, (*batch_sizes, seq_len))
      chex.assert_type(position_ids, int)
    if segment_ids is not None:
      chex.assert_shape(segment_ids, (*batch_sizes, seq_len))
      chex.assert_type(segment_ids, int)

    embeddings = self.embedder_block(
        input_ids=input_ids,
        position_ids=position_ids,
        segment_ids=segment_ids,
        enable_dropout=enable_dropout)
    chex.assert_shape(embeddings, (*batch_sizes, seq_len, None))

    attention_mask = dense_attention.make_attention_mask(input_mask, input_mask)
    chex.assert_shape(attention_mask, (*batch_sizes, 1, seq_len, seq_len))

    result = self.encoder_block(
        embeddings,
        attention_mask=attention_mask,
        enable_dropout=enable_dropout)
    chex.assert_shape(result, (*batch_sizes, seq_len, None))

    return result


class EmbedderBlock(nn.Module):
  """Embeds the inputs, then applies dropout and layer norm."""

  embedder: embedding.MultiEmbed
  dropout: nn.Dropout
  layer_norm: nn.LayerNorm

  def __call__(self,
               input_ids: Array,
               *,
               position_ids: Optional[Array] = None,
               segment_ids: Optional[Array] = None,
               enable_dropout: bool = True) -> Array:
    """Embeds the inputs, then applies dropout and layer norm.

    Args:
      input_ids: <int>[batch..., seq_len]. The first position in the sequence
        should correspond to a "beginning of input" symbol, generally `[CLS]`.
      position_ids: <int>[batch..., seq_len]. The position of each input ID
        within its sequence. This is typically just `range(seq_len)` for each
        sequence, but custom behavior may be desired if, for instance, multiple
        examples are packed into each sequence.
      segment_ids: <int>[batch..., seq_len]. Indicates the "type" of each input
        position. For a traditional BERT-style model with two segments, valid
        values would be {0, 1}.
      enable_dropout: Enables dropout when set to True.

    Returns:
      <float>[batch..., seq_len, hidden_size].
    """
    # Validate inputs and create variables for dimension sizes.
    chex.assert_shape(input_ids, (..., None))
    chex.assert_type(input_ids, int)
    *batch_sizes, seq_len = input_ids.shape
    if position_ids is not None:
      chex.assert_shape(position_ids, (*batch_sizes, seq_len))
      chex.assert_type(position_ids, int)
    if segment_ids is not None:
      chex.assert_shape(segment_ids, (*batch_sizes, seq_len))
      chex.assert_type(segment_ids, int)

    embeddings = self.embedder(  # pytype: disable=wrong-arg-types  # jax-ndarray
        input_ids=input_ids,
        position_ids=position_ids,
        segment_ids=segment_ids,
    )
    chex.assert_shape(embeddings, (*batch_sizes, seq_len, None))

    embeddings = self.layer_norm(embeddings)
    embeddings = self.dropout(embeddings, deterministic=not enable_dropout)
    chex.assert_shape(embeddings, (*batch_sizes, seq_len, None))

    return embeddings


class EncoderBlock(nn.Module):
  """A BERT encoder block: a sequence of transformer layers.

  Passes inputs through the full stack of encoder layers. Note that dropout and
  layer norm are performed within and at the end of each encoder layer.
  """
  layer_sequence: common.LayerSequence

  def __call__(self,
               inputs: Array,
               *,
               attention_mask: Array,
               enable_dropout: bool = True) -> Array:
    """Applies all the layers starting with the inputs.

    Args:
      inputs: The inputs, <float>[..., seq_len, hidden_size].
      attention_mask: The mask over input positions, <bool>[..., num_heads,
        seq_len, seq_len].
      enable_dropout: Enables dropout when set to True.

    Returns:
      The encoded inputs, <float>[..., seq_len, hidden_size].
    """
    return self.layer_sequence(
        inputs, attention_mask=attention_mask, enable_dropout=enable_dropout)


class EncoderLayer(nn.Module):
  """Transformer-based encoder layer used in BERT.

  Performs the following:
  1. An attention block, which ends with dropout and layer norm.
  2. An MLP block, which ends with dropout and layer norm.
  """
  attention_block: AttentionBlock
  mlp_block: MlpBlock

  def __call__(self,
               inputs: Array,
               *,
               attention_targets: Optional[Array] = None,
               attention_mask: Optional[Array] = None,
               attention_bias: Optional[Array] = None,
               enable_dropout: bool = True) -> Array:
    """Applies one attention block and one MLP block.

    Args:
      inputs: <float>[batch..., seq_len, features]. Sequences of inputs. These
        may attend to values in `attention_targets`.
      attention_targets: <float>[batch..., target_seq_len, target_features]:
        Sequence of values that the `inputs` positions may attend to. If `None`,
        `inputs` will be used.
      attention_mask: <bool>[batch..., num_heads, seq_len, target_seq_len].
        Attention mask where True indicates that the position in `inputs` may
        attend to the position in `attention_targets`.
      attention_bias: <float>[batch..., num_heads, seq_len, target_seq_len].
        Bias for the attention weights. This can be used for incorporating
        causal masks, padding masks, proximity bias, etc.
      enable_dropout: Enables dropout if set to True.

    Returns:
      [batch..., seq_len, features]
    """
    result = self.attention_block(
        inputs=inputs,
        attention_targets=(attention_targets
                           if attention_targets is not None else inputs),
        mask=attention_mask,
        bias=attention_bias,
        enable_dropout=enable_dropout)
    chex.assert_equal_shape((inputs, result))
    result = self.mlp_block(result, enable_dropout=enable_dropout)
    chex.assert_equal_shape((inputs, result))
    return result


class AttentionBlock(nn.Module):
  """A single transformer attention block.

  Performs the following:
  1. Attention.
  2. Dense projection back to the input shape.
  3. Dropout.
  4. Residual connection.
  5. LayerNorm.

  Attributes:
    attention_layer: The attention layer.
    dense_layer: The dense layer for projecting the attention layer's output
      back to the shape of `inputs`.
    dropout: Performs dropout.
    layer_norm: Performs layer normalization.
  """
  attention_layer: dense_attention.DenseAttention
  dense_layer: dense.DenseGeneral
  dropout: nn.Dropout
  layer_norm: nn.LayerNorm

  def __call__(self,
               inputs: Array,
               attention_targets: Array,
               *,
               mask: Optional[Array] = None,
               bias: Optional[Array] = None,
               enable_dropout: bool = True) -> Array:
    """Applies a single transformer attention block.

    Args:
      inputs: <float>[batch..., seq_len, features]. Sequences of inputs. These
        may attend to values in `attention_targets`.
      attention_targets: <float>[batch..., target_seq_len, target_features]:
        Sequence of values that the `inputs` positions may attend to.
      mask: <bool>[batch..., num_heads, seq_len, target_seq_len]. Attention mask
        where True indicates that the position in `inputs` may attend to the
        position in `attention_targets`.
      bias: <float>[batch..., num_heads, seq_len, target_seq_len]. Bias for the
        attention weights. This can be used for incorporating causal masks,
        padding masks, proximity bias, etc.
      enable_dropout: Enables dropout if set to True.

    Returns:
      <float>[batch..., seq_len, features]
    """
    # Validate inputs and create variables for dimension sizes.
    chex.assert_shape(inputs, (..., None, self.dense_layer.features))
    *batch_sizes, seq_len, features = inputs.shape
    chex.assert_shape(attention_targets, (*batch_sizes, None, None))
    target_seq_len = attention_targets.shape[-2]
    if mask is not None:
      chex.assert_shape(mask, (*batch_sizes, None, seq_len, target_seq_len))
    if bias is not None:
      chex.assert_shape(bias, (*batch_sizes, None, seq_len, target_seq_len))

    # TODO: Do we want the dropout that's in the attention layer?
    # [batch..., seq_len, num_heads, head_dim]
    attention_output = self.attention_layer(
        inputs,
        attention_targets,
        mask=mask,
        bias=bias,
        enable_dropout=enable_dropout)

    chex.assert_shape(attention_output, (*batch_sizes, seq_len, None, None))
    num_heads = attention_output.shape[-2]
    chex.assert_is_divisible(features, num_heads)
    if mask is not None:
      chex.assert_shape(mask,
                        (*batch_sizes, {num_heads, 1}, seq_len, target_seq_len))
    if bias is not None:
      chex.assert_shape(bias,
                        (*batch_sizes, {num_heads, 1}, seq_len, target_seq_len))

    # Project back to input shape.
    # [batch..., seq_len, features]
    result = self.dense_layer(attention_output)
    chex.assert_equal_shape((inputs, result))

    result = self.dropout(result, deterministic=not enable_dropout)
    result = result + inputs
    result = self.layer_norm(result)
    return result


class MlpBlock(nn.Module):
  """A single transformer MLP block.

  Performs the following:
  1. MLP.
  2. Dense projection back to the input shape.
  3. Dropout.
  4. Residual connection.
  5. LayerNorm.

  Attributes:
    mlp: The MLP layer.
    dense_layer: The dense layer for projecting the MLP layer's output back to
      the shape of `inputs`.
    dropout: Performs dropout.
    layer_norm: Performs layer normalization.
  """
  mlp: Mlp
  dense_layer: dense.DenseGeneral
  dropout: nn.Dropout
  layer_norm: nn.LayerNorm

  def __call__(self, inputs: Array, *, enable_dropout: bool = True) -> Array:
    """Applies a single transformer MLP block.

    Args:
      inputs: <float>[batch..., seq_len, hidden_size]. Sequences of inputs.
      enable_dropout: Enables dropout if set to True.

    Returns:
      [batch..., seq_len, hidden_size]
    """
    chex.assert_shape(inputs, (..., None, self.dense_layer.features))

    # [batch..., seq_len, intermediate_dim]
    mlp_output = self.mlp(inputs)

    # Project back to input shape.
    # [batch..., seq_len, hidden_size]
    result = self.dense_layer(mlp_output)
    chex.assert_equal_shape((inputs, result))

    result = self.dropout(result, deterministic=not enable_dropout)
    result = result + inputs
    result = self.layer_norm(result)
    return result


class Mlp(nn.Module):
  dense_layer: dense.DenseGeneral
  activation: Activation

  def __call__(self, inputs: Array) -> Array:
    result = self.dense_layer(inputs)
    return self.activation(result)


def make_full_encoder(
    hidden_size: int,
    intermediate_dim: int,
    vocab_size: int,
    max_length: int,
    num_segments: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    dropout_rate: float = 0.0,
    dtype: jnp.dtype = jnp.float32,
    kernel_init: Initializer = initializers.truncated_normal(
        stddev=_DEFAULT_INIT_RANGE),
    bias_init: Initializer = nn.initializers.zeros,
    layer_norm_epsilon: float = _DEFAULT_LAYER_NORM,
) -> FullEncoder:
  """Returns a newly constructed Encoder.

  Args:
    hidden_size: The size of the embeddings and the BERT layers.
    intermediate_dim: Size of the feed-forward layer in the TransformerLayer's
      MLP block. Corresponds to `dff` in the transformer paper.
    vocab_size: The vocabulary size.
    max_length: The number of supported token positions.
    num_segments: The number of segments (token types).
    num_hidden_layers: Total number of hidden BertTransformer layers. 12 in the
      BERT-base model, 24 in the BERT-large model.
    num_attention_heads: Total number of self-attention heads. 12 in the
      BERT-base model, 16 in the BERT-large model.
    dropout_rate: Dropout probability used across all the model layers.
    dtype: The dtype of the computation (float16/float32/float64).
    kernel_init: Initializer method for attention and mlp layers kernels.
    bias_init: Initializer method for attention and mlp layers biases.
    layer_norm_epsilon: The layer norm epsilon parameter.
  """
  return FullEncoder(
      embedder_block=EmbedderBlock(
          embedder=embedding.MultiEmbed({
              'input_ids':
                  embedding.Embed(
                      num_embeddings=vocab_size,
                      features=hidden_size,
                      embedding_init=kernel_init),
              'position_ids':
                  embedding.Embed(
                      num_embeddings=max_length,
                      features=hidden_size,
                      embedding_init=kernel_init),
              'segment_ids':
                  embedding.Embed(
                      num_embeddings=num_segments,
                      features=hidden_size,
                      embedding_init=kernel_init)
          }),
          layer_norm=nn.LayerNorm(epsilon=layer_norm_epsilon),
          dropout=nn.Dropout(rate=dropout_rate)),
      encoder_block=make_encoder_block(
          hidden_size=hidden_size,
          intermediate_dim=intermediate_dim,
          num_layers=num_hidden_layers,
          num_attention_heads=num_attention_heads,
          dropout_rate=dropout_rate,
          dtype=dtype,
          kernel_init=kernel_init,
          bias_init=bias_init,
          layer_norm_epsilon=layer_norm_epsilon))


def make_encoder_block(
    *,
    hidden_size: int,
    intermediate_dim: int,
    num_layers: int,
    num_attention_heads: int,
    dropout_rate: float = 0.0,
    dtype: jnp.dtype = jnp.float32,
    kernel_init: Initializer = initializers.truncated_normal(
        stddev=_DEFAULT_INIT_RANGE
    ),
    bias_init: Initializer = nn.initializers.zeros,
    layer_norm_epsilon: float = _DEFAULT_LAYER_NORM,
    sow_attention_intermediates: bool = False,
) -> EncoderBlock:
  """Returns a newly constructed EncoderBlock.

  Args:
    hidden_size: The size of the embeddings and the BERT layers.
    intermediate_dim: Size of the feed-forward layer in the BertLayer MlpBlock.
      Corresponds to `dff` in the transformer paper.
    num_layers: Total number of hidden BertTransformer layers. 12 in the
      BERT-base model, 24 in the BERT-large model.
    num_attention_heads: Total number of self-attention heads. 12 in the
      BERT-base model, 16 in the BERT-large model.
    dropout_rate: Dropout probability used across all the model layers.
    dtype: The dtype of the computation (float16/float32/float64).
    kernel_init: Initializer method for attention and mlp layers kernels.
    bias_init: Initializer method for attention and mlp layers biases.
    layer_norm_epsilon: The layer norm epsilon parameter.
    sow_attention_intermediates: Whether to track attention intermediates using
      Module.sow.
  """

  def make_layer() -> EncoderLayer:
    return make_encoder_layer(
        make_attention_layer(
            num_heads=num_attention_heads,
            dropout_rate=dropout_rate,
            kernel_init=kernel_init,
            bias_init=bias_init,
            dtype=dtype,
            sow_intermediates=sow_attention_intermediates,
        ),
        hidden_size=hidden_size,
        intermediate_dim=intermediate_dim,
        dtype=dtype,
        dropout_rate=dropout_rate,
        kernel_init=kernel_init,
        bias_init=bias_init,
        layer_norm_epsilon=layer_norm_epsilon,
    )

  return EncoderBlock(
      common.LayerSequence(num_layers=num_layers, make_layer=make_layer),  # pytype: disable=wrong-keyword-args
  )


class BertEncoder(nn.Module):
  """A BERT encoder model that embeds inputs and encodes them with BERT layers.

  Note that dropout and layer norm are performed within and at the end of each
  encoder layer.

  Attributes:
    hidden_size: The size of the embeddings and the BERT layers.
    intermediate_dim: Size of the feed-forward layer in the BertLayer MlpBlock.
      Corresponds to `dff` in the transformer paper.
    vocab_size: The vocabulary size.
    max_length: The number of supported token positions.
    num_segments: The number of segments (token types).
    num_hidden_layers: Total number of hidden BertTransformer layers. 12 in the
      BERT-base model, 24 in the BERT-large model.
    num_attention_heads: Total number of self-attention heads. 12 in the
      BERT-base model, 16 in the BERT-large model.
    dropout_rate: Dropout probability used across all the model layers.
    dtype: The dtype of the computation (float16/float32/float64).
    kernel_init: Initializer method for attention and mlp layers kernels.
    bias_init: Initializer method for attention and mlp layers biases.
    layer_norm_epsilon: The layer norm epsilon parameter.
    enable_dropout: Enables dropout when set to True.
  """
  hidden_size: int
  intermediate_dim: int
  vocab_size: int
  max_length: int
  num_segments: int
  num_hidden_layers: int
  num_attention_heads: int
  dropout_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32
  kernel_init: Initializer = initializers.truncated_normal(
      stddev=_DEFAULT_INIT_RANGE)
  bias_init: Initializer = nn.initializers.zeros
  layer_norm_epsilon: float = _DEFAULT_LAYER_NORM
  enable_dropout: Optional[bool] = None

  def setup(self):
    self.embedder = embedding.MultiEmbed({
        'token_ids':
            embedding.Embed(
                num_embeddings=self.vocab_size,
                features=self.hidden_size,
                embedding_init=self.kernel_init),
        'position_ids':
            embedding.Embed(
                num_embeddings=self.max_length,
                features=self.hidden_size,
                embedding_init=self.kernel_init),
        'segment_ids':
            embedding.Embed(
                num_embeddings=self.num_segments,
                features=self.hidden_size,
                embedding_init=self.kernel_init)
    })

    self.layer_norm = nn.LayerNorm(epsilon=self.layer_norm_epsilon)
    self.embeddings_dropout = nn.Dropout(rate=self.dropout_rate)

    self.encoder_block = make_encoder_block(
        hidden_size=self.hidden_size,
        intermediate_dim=self.intermediate_dim,
        num_layers=self.num_hidden_layers,
        num_attention_heads=self.num_attention_heads,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        layer_norm_epsilon=self.layer_norm_epsilon)

  def __call__(self,
               token_ids: Array,
               position_ids: Array,
               segment_ids: Array,
               input_mask: Array,
               enable_dropout: Optional[bool] = None) -> Array:
    """Embeds the inputs and encodes them with BERT layers.

    Args:
      token_ids: The token IDs, <int>[..., seq_len].
      position_ids: The position IDs, <int>[..., seq_len]. Should broadcast over
        token_ids, or have the same shape.
      segment_ids:  The segment (token type) IDs, <int>[..., seq_len].
      input_mask: The mask over token IDs, <bool>[..., seq_len].
      enable_dropout: Enables dropout when set to True.

    Returns:
      The encoded inputs.
    """
    embedded_inputs = (
        self.embed_and_combine_inputs(token_ids, position_ids, segment_ids))
    return self.encode_from_embedded(
        embedded_inputs, input_mask, enable_dropout=enable_dropout)

  def embed_and_combine_inputs(self, token_ids: Array, position_ids: Array,
                               segment_ids: Array) -> Array:
    """Embeds the inputs and combines them for further processing."""
    embedded_inputs = self.embedder(  # pytype: disable=wrong-arg-types  # jax-ndarray
        token_ids=token_ids, position_ids=position_ids, segment_ids=segment_ids
    )
    return embedded_inputs

  def finalize_embeddings(self,
                          embedded_inputs: Array,
                          *,
                          enable_dropout: Optional[bool] = None) -> Array:
    """Finalize embedded inputs to be sent to the first transformer layer."""
    enable_dropout = nn.module.merge_param('enable_dropout',
                                           self.enable_dropout, enable_dropout)

    embedded_inputs = self.layer_norm(embedded_inputs)
    if enable_dropout is not None:
      deterministic = not enable_dropout
    else:
      deterministic = None
    embedded_inputs = self.embeddings_dropout(embedded_inputs, deterministic)
    return embedded_inputs

  def encode_from_embedded(self,
                           embedded_inputs: Array,
                           input_mask: Array,
                           *,
                           enable_dropout: Optional[bool] = None) -> Array:
    """Runs the encoder on embedded inputs."""
    embedded_inputs = self.finalize_embeddings(
        embedded_inputs, enable_dropout=enable_dropout)
    attention_mask = dense_attention.make_attention_mask(input_mask, input_mask)
    return self.encoder_block(
        embedded_inputs,
        attention_mask=attention_mask,
        enable_dropout=enable_dropout)


class BertMlmNsp(nn.Module):
  """A BERT encoder with a pooler, MLM-head and NSP-head.

  Attributes:
    encoder: An encoder that returns a sequence of input representations.
    pooler: A sequence pooler. In the original BERT model the pooler is
      parameterized: it selects the first position (the CLS token) and applies a
        dense layer with activation.
    mlm_head: A masked language modeling head.
    nsp_head: A next sentence prediction head.
  """
  encoder: BertEncoder
  pooler: heads.BertPooler
  mlm_head: heads.MLMHead
  nsp_head: heads.NSPHead

  def __call__(self,
               token_ids: Array,
               *,
               position_ids: Array,
               segment_ids: Array,
               input_mask: Array,
               masked_positions: Optional[Array] = None,
               enable_dropout: bool = True) -> Tuple[Array, Array]:
    """Encodes the inputs with the encoder and applies the MLM and NSP heads."""
    encoded_inputs = self.encoder(
        token_ids,
        position_ids=position_ids,
        segment_ids=segment_ids,
        input_mask=input_mask,
        enable_dropout=enable_dropout)
    return self.mlm_head(
        encoded_inputs,
        masked_positions=masked_positions), self.nsp_head(encoded_inputs)


class BertClassifier(nn.Module):
  """A BERT encoder with a pooler and classification head.

  Attributes:
    encoder: An encoder that returns a sequence of input representations.
    pooler: A sequence pooler. In the original BERT model the pooler is
      parameterized: it selects the first position (the CLS token) and applies a
        dense layer with activation. Note that this module is not actually used
        by `BertClassifier`, but some heads might depend on it. For example, the
        default `heads.ClassifierHead` does have its own pooler params and does
        not need the params provided in `pooler`.
    classifier_head: A classification head. Operates on pooled encodings.
  """
  encoder: BertEncoder
  # TODO: Reduce redundancy of `pooler` and `heads.ClassifierHead`.
  pooler: heads.BertPooler
  classifier_head: heads.ClassifierHead

  def encode(self,
             token_ids,
             *,
             position_ids,
             segment_ids,
             input_mask,
             enable_dropout: bool = True):
    """Encodes the inputs with the encoder."""
    return self.encoder(
        token_ids,
        position_ids=position_ids,
        segment_ids=segment_ids,
        input_mask=input_mask,
        enable_dropout=enable_dropout)

  def classify(self, encoded_inputs, enable_dropout: bool = True):
    """Classifies the encoded inputs."""
    return self.classifier_head(encoded_inputs, enable_dropout=enable_dropout)

  def __call__(self,
               token_ids: Array,
               *,
               position_ids: Array,
               segment_ids: Array,
               input_mask: Array,
               enable_dropout: bool = True) -> Array:
    """Encodes the inputs with the encoder and applies the classifier head."""
    encoded_inputs = self.encode(
        token_ids,
        position_ids=position_ids,
        segment_ids=segment_ids,
        input_mask=input_mask,
        enable_dropout=enable_dropout)
    return self.classify(encoded_inputs, enable_dropout=enable_dropout)


def make_encoder_layer(
    attention_layer: dense_attention.DenseAttention,
    *,
    hidden_size: int,
    intermediate_dim: int,
    dtype: DType = jnp.float32,
    dropout_rate: float = 0.0,
    kernel_init: Initializer = initializers.truncated_normal(
        stddev=_DEFAULT_INIT_RANGE),
    bias_init: Initializer = nn.initializers.zeros,
    layer_norm_epsilon: float = _DEFAULT_LAYER_NORM,
    name: Optional[str] = None,
) -> EncoderLayer:
  """Returns a Bert-style transformer layer."""
  return EncoderLayer(
      attention_block=make_attention_block(
          attention_layer=attention_layer,
          hidden_size=hidden_size,
          dtype=dtype,
          dropout_rate=dropout_rate,
          kernel_init=kernel_init,
          bias_init=bias_init,
          layer_norm_epsilon=layer_norm_epsilon),
      mlp_block=make_mlp_block(
          hidden_size=hidden_size,
          intermediate_dim=intermediate_dim,
          dtype=dtype,
          dropout_rate=dropout_rate,
          kernel_init=kernel_init,
          bias_init=bias_init,
          layer_norm_epsilon=layer_norm_epsilon),
      name=name)


def make_attention_block(
    attention_layer: dense_attention.DenseAttention,
    *,
    hidden_size: int,
    dtype: DType = jnp.float32,
    dropout_rate: float = 0.0,
    kernel_init: Initializer = initializers.truncated_normal(
        stddev=_DEFAULT_INIT_RANGE),
    bias_init: Initializer = nn.initializers.zeros,
    layer_norm_epsilon: float = _DEFAULT_LAYER_NORM,
    name: Optional[str] = None,
) -> AttentionBlock:
  """Returns a Bert-style transformer attention block."""
  return AttentionBlock(
      attention_layer=attention_layer,
      dense_layer=dense.DenseGeneral(
          features=hidden_size,
          axis=(-2, -1),
          kernel_init=kernel_init,
          bias_init=bias_init,
          use_bias=True,
          kernel_axis_names=('heads', 'kv', 'embed'),
          dtype=dtype),
      # We chose to not broadcast dropout (compared to T5),
      # because of a lack of evidence that it was used by BERT).
      dropout=nn.Dropout(rate=dropout_rate),
      layer_norm=nn.LayerNorm(epsilon=layer_norm_epsilon, dtype=dtype),
      name=name)


def make_attention_layer(
    *,
    num_heads: int,
    dropout_rate: float = 0.0,
    kernel_init: Initializer = initializers.truncated_normal(
        stddev=_DEFAULT_INIT_RANGE
    ),
    bias_init: Initializer = nn.initializers.zeros,
    dtype: DType = jnp.float32,
    sow_intermediates: bool = False,
    name: Optional[str] = None,
) -> dense_attention.DenseAttention:
  """Returns a Bert-style attention layer."""
  return dense_attention.MultiHeadDotProductAttention(
      num_heads=num_heads,
      dtype=dtype,
      broadcast_dropout=False,
      dropout_rate=dropout_rate,
      kernel_init=kernel_init,
      bias_init=bias_init,
      use_bias=True,
      rescale_logits=True,
      output_projection=False,
      sow_intermediates=sow_intermediates,
      name=name,
  )


def make_mlp_block(
    *,
    hidden_size: int,
    intermediate_dim: int,
    dtype: DType = jnp.float32,
    dropout_rate: float = 0.0,
    kernel_init: Initializer = initializers.truncated_normal(
        stddev=_DEFAULT_INIT_RANGE),
    bias_init: Initializer = nn.initializers.zeros,
    layer_norm_epsilon: float = _DEFAULT_LAYER_NORM,
    name: Optional[str] = None,
) -> MlpBlock:
  """Returns a Bert-style transformer MLP block."""
  return MlpBlock(
      mlp=Mlp(
          dense_layer=dense.DenseGeneral(
              features=intermediate_dim,
              kernel_axis_names=('embed', 'mlp'),
              use_bias=True,
              dtype=dtype,
              kernel_init=kernel_init,
              bias_init=bias_init),
          activation=nn.gelu),
      dense_layer=dense.DenseGeneral(
          features=hidden_size,
          use_bias=True,
          dtype=dtype,
          kernel_init=kernel_init,
          kernel_axis_names=('mlp', 'embed'),
          bias_init=bias_init),
      dropout=nn.Dropout(
          rate=dropout_rate,
          broadcast_dims=(-2,),  # Broadcast along sequence length.
      ),
      layer_norm=nn.LayerNorm(
          epsilon=layer_norm_epsilon,
          dtype=dtype,
      ),
      name=name)
