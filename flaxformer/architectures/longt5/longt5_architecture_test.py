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

"""Tests for t5_architecture."""
import json
import pathlib
import re
from typing import Any, Optional

from absl.testing import absltest
from flax import linen as nn
from flax.core import frozen_dict
import jax
from jax import numpy as jnp
from jax import random
import numpy as np
from flaxformer.architectures.common import param_remapping
from flaxformer.architectures.longt5 import long_attention
from flaxformer.architectures.longt5 import longt5_architecture
from flaxformer.architectures.longt5 import relative_position_biases_general
from flaxformer.architectures.t5 import t5_architecture
from flaxformer.components import dense
from flaxformer.components import embedding
from flaxformer.components import layer_norm
from flaxformer.components import relative_position_biases
from flaxformer.components.attention import dense_attention
from flaxformer.types import Array


def _dense_t5_testdata_dir() -> pathlib.Path:
  return (pathlib.Path(absltest.get_default_test_srcdir()) /
          'flaxformer/architectures/t5/testdata')


def check_dense_t5_params(actual_params: frozen_dict.FrozenDict[str, Any],
                          expected_filename: str) -> None:
  actual = jax.tree.map(
      lambda x: list(x.shape),
      frozen_dict.unfreeze(param_remapping.filter_out_metadata(actual_params)))
  expected = json.load(open(_dense_t5_testdata_dir() / expected_filename))

  if actual != expected:
    print(
        re.sub(r'\[\n\s+(\d+,\s+)*\d+\s+\]',
               lambda m: ''.join(m.group(0).split()).replace(',', ', '),
               json.dumps(actual, indent=2)))
    raise AssertionError(
        f'Didn\'t match JSON params in {expected_filename}. See actual '
        'values above.')


EMBEDDING_INIT = nn.initializers.normal(stddev=1.0)
RELPOS_BIAS_INIT = nn.initializers.variance_scaling(1.0, 'fan_avg', 'uniform')
ATTENTION_KERNEL_INIT = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal')
MLP_KERNEL_INIT = nn.initializers.variance_scaling(1.0, 'fan_in',
                                                   'truncated_normal')
FINAL_KERNEL_INIT = nn.initializers.variance_scaling(1.0, 'fan_in',
                                                     'truncated_normal')
BIAS_INIT = nn.initializers.normal(stddev=1e-6)


class DegenerateLongSelfAttention(dense_attention.MultiHeadDotProductAttention,
                                  long_attention.LongSelfAttention):
  """A degenerate implementation of `LongSelfAttention` for testing.

  This just performs full self-attention after creating full attention mask
  and bias arrays.  We inherit from `MultiHeadDotProductAttention` to preserve
  the same parameter naming structure, allowing us to use the same
  json testdata as the standard ("dense") T5 architecture.
  """
  relpos_bias: Optional[nn.Module] = None

  @nn.compact
  def __call__(
      self,
      inputs: Array,
      inputs_mask: Array,
      *,
      segment_ids: Optional[Array] = None,
      positions: Optional[Array] = None,
      enable_dropout: bool = True,
  ) -> Array:
    # Make padding attention mask.
    encoder_mask = dense_attention.make_attention_mask(
        inputs_mask, inputs_mask, dtype=self.dtype)

    # Add segmentation block-diagonal attention mask if using segmented data.
    if segment_ids is not None:
      encoder_mask = dense_attention.combine_masks(
          encoder_mask,
          dense_attention.make_attention_mask(
              segment_ids, segment_ids, jnp.equal, dtype=self.dtype))

    # Shared relative position embedding attention biases.
    if self.relpos_bias:
      encoder_bias = self.relpos_bias(inputs.shape[-2], inputs.shape[-2], True)  # pylint: disable=not-callable
    else:
      encoder_bias = None

    return super().__call__(
        inputs,
        inputs,
        encoder_mask,
        encoder_bias,
        enable_dropout=enable_dropout)


def make_token_emb1(vocab_size, dtype):
  """First test configuration for token embeddings."""
  return embedding.Embed(
      num_embeddings=vocab_size,
      features=13,
      cast_input_dtype=jnp.int32,
      dtype=dtype,
      attend_dtype=jnp.float32,  # for logit training stability
      embedding_init=EMBEDDING_INIT,
      name='token_embedder')


def make_long_att_fn1(num_attn_heads, dtype):
  """First test configuration for long encoder self-attention."""

  def fn(relpos_bias):
    return DegenerateLongSelfAttention(
        num_heads=num_attn_heads,
        dtype=dtype,
        qkv_features=512,
        head_dim=None,
        kernel_init=ATTENTION_KERNEL_INIT,
        bias_init=BIAS_INIT,
        use_bias=False,
        broadcast_dropout=True,
        dropout_rate=0.1,
        relpos_bias=relpos_bias)

  return fn


def make_attention1(num_attn_heads, dtype):
  """First test configuration for attention in decoder."""
  return dense_attention.MultiHeadDotProductAttention(
      num_heads=num_attn_heads,
      dtype=dtype,
      qkv_features=512,
      head_dim=None,
      kernel_init=ATTENTION_KERNEL_INIT,
      bias_init=BIAS_INIT,
      use_bias=False,
      broadcast_dropout=True,
      dropout_rate=0.1)


def make_mlp1(dtype):
  """First test configuration for the MLP."""
  return dense.MlpBlock(
      use_bias=False,
      intermediate_dim=2048,
      activations=('relu',),
      kernel_init=MLP_KERNEL_INIT,
      bias_init=BIAS_INIT,
      intermediate_dropout_rate=0.1,
      final_dropout_rate=0.1,
      dtype=dtype)


def _make_relative_position_bias(
    num_attn_heads: int,
    dtype: Any) -> relative_position_biases.RelativePositionBiases:
  return relative_position_biases.RelativePositionBiases(
      num_buckets=32,
      max_distance=128,
      num_heads=num_attn_heads,
      dtype=dtype,
      embedding_init=RELPOS_BIAS_INIT)


def _make_relpos_bias_general(
    num_attn_heads: int,
    dtype: Any) -> relative_position_biases.RelativePositionBiases:
  return relative_position_biases_general.RelativePositionBiasesGeneral(
      num_buckets=32,
      max_distance=128,
      num_heads=num_attn_heads,
      dtype=dtype,
      embedding_init=RELPOS_BIAS_INIT)


def make_config1(
    scan_layers: bool = False,
    layer_remat: str = 'legacy') -> longt5_architecture.LongEncoderDecoder:
  """Returns a LongEncoderDecoder."""
  dtype = jnp.float32
  num_attn_heads = 8
  make_dropout = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,))
  make_layer_norm = layer_norm.T5LayerNorm

  def _make_encoder_layer(shared_relpos_bias):
    assert shared_relpos_bias is None
    return longt5_architecture.LongEncoderLayer(
        attention_factory=make_long_att_fn1(num_attn_heads, dtype),
        mlp=make_mlp1(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        relpos_bias_factory=(
            lambda: _make_relative_position_bias(num_attn_heads, dtype)),
        scanned=scan_layers)

  def _make_decoder_layer(shared_relative_position_bias):
    assert shared_relative_position_bias is None
    return t5_architecture.DecoderLayer(
        self_attention=make_attention1(num_attn_heads, dtype),
        encoder_decoder_attention=make_attention1(num_attn_heads, dtype),
        mlp=make_mlp1(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        relative_position_bias_factory=(
            lambda: _make_relative_position_bias(num_attn_heads, dtype)),
        scanned=scan_layers)

  def _make_encoder(shared_token_embedder):
    assert shared_token_embedder is None
    return longt5_architecture.LongEncoder(
        num_layers=3,
        token_embedder_factory=lambda: make_token_emb1(2_000, dtype),
        layer_factory=_make_encoder_layer,
        input_dropout_factory=make_dropout,
        output_dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        dtype=dtype,
        scan_layers=scan_layers,
        layer_remat=layer_remat,
    )

  def _make_decoder(shared_token_embedder):
    assert shared_token_embedder is None
    return t5_architecture.Decoder(
        num_layers=2,
        token_embedder_factory=lambda: make_token_emb1(2_000, dtype),
        layer_factory=_make_decoder_layer,
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        output_logits_factory=None,
        dtype=dtype,
        scan_layers=scan_layers,
        layer_remat=layer_remat,
    )

  return longt5_architecture.LongEncoderDecoder(
      shared_token_embedder_factory=lambda: None,
      encoder_factory=_make_encoder,
      decoder_factory=_make_decoder,
      dtype=dtype,
      scan_layers=scan_layers,
  )


def make_config1_original_t5(
    scan_layers: bool = False) -> t5_architecture.EncoderDecoder:
  dtype = jnp.float32
  num_attn_heads = 8
  make_dropout = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,))
  make_layer_norm = layer_norm.T5LayerNorm

  def _make_encoder_layer(shared_relative_position_bias):
    assert shared_relative_position_bias is None
    return t5_architecture.EncoderLayer(
        attention=make_attention1(num_attn_heads, dtype),
        mlp=make_mlp1(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        relative_position_bias_factory=(
            lambda: _make_relative_position_bias(num_attn_heads, dtype)),
        scanned=scan_layers)

  def _make_decoder_layer(shared_relative_position_bias):
    assert shared_relative_position_bias is None
    return t5_architecture.DecoderLayer(
        self_attention=make_attention1(num_attn_heads, dtype),
        encoder_decoder_attention=make_attention1(num_attn_heads, dtype),
        mlp=make_mlp1(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        relative_position_bias_factory=(
            lambda: _make_relative_position_bias(num_attn_heads, dtype)),
        scanned=scan_layers)

  def _make_encoder(shared_token_embedder):
    assert shared_token_embedder is None
    return t5_architecture.Encoder(
        num_layers=3,
        token_embedder_factory=lambda: make_token_emb1(2_000, dtype),
        layer_factory=_make_encoder_layer,
        input_dropout_factory=make_dropout,
        output_dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        dtype=dtype,
        scan_layers=scan_layers,
    )

  def _make_decoder(shared_token_embedder):
    assert shared_token_embedder is None
    return t5_architecture.Decoder(
        num_layers=2,
        token_embedder_factory=lambda: make_token_emb1(2_000, dtype),
        layer_factory=_make_decoder_layer,
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        output_logits_factory=None,
        dtype=dtype,
        scan_layers=scan_layers,
    )

  return t5_architecture.EncoderDecoder(
      shared_token_embedder_factory=lambda: None,
      encoder_factory=_make_encoder,
      decoder_factory=_make_decoder,
      dtype=dtype,
      scan_layers=scan_layers,
  )


# TODO: DRY up with above configs.
def make_config2_shared_relative_position_bias(
) -> longt5_architecture.LongEncoderDecoder:
  dtype = jnp.float32
  num_attn_heads = 8
  make_dropout = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,))
  make_layer_norm = layer_norm.T5LayerNorm

  def _make_encoder_layer(shared_relpos_bias):
    assert shared_relpos_bias is not None
    return longt5_architecture.LongEncoderLayer(
        attention_factory=make_long_att_fn1(num_attn_heads, dtype),
        mlp=make_mlp1(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        shared_relpos_bias=shared_relpos_bias)

  def _make_decoder_layer(shared_relative_position_bias):
    assert shared_relative_position_bias is not None
    return t5_architecture.DecoderLayer(
        self_attention=make_attention1(num_attn_heads, dtype),
        encoder_decoder_attention=make_attention1(num_attn_heads, dtype),
        mlp=make_mlp1(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        shared_relative_position_bias=shared_relative_position_bias)

  def _make_encoder(*, shared_token_embedder=None):
    assert shared_token_embedder is None
    return longt5_architecture.LongEncoder(
        num_layers=3,
        token_embedder_factory=lambda: make_token_emb1(2_000, dtype),
        layer_factory=_make_encoder_layer,
        input_dropout_factory=make_dropout,
        output_dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        shared_relpos_bias_factory=(
            lambda: _make_relative_position_bias(num_attn_heads, dtype)),
        dtype=dtype,
    )

  def _make_decoder(*, shared_token_embedder=None):
    assert shared_token_embedder is None
    return t5_architecture.Decoder(
        num_layers=2,
        token_embedder_factory=lambda: make_token_emb1(2_000, dtype),
        layer_factory=_make_decoder_layer,
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        output_logits_factory=None,
        shared_relative_position_bias_factory=(
            lambda: _make_relative_position_bias(num_attn_heads, dtype)),
        dtype=dtype,
    )

  return longt5_architecture.LongEncoderDecoder(
      shared_token_embedder_factory=lambda: None,
      encoder_factory=_make_encoder,
      decoder_factory=_make_decoder,
  )


# TODO: DRY up with above configs.
def make_config3_shared_token_embedder(
) -> longt5_architecture.LongEncoderDecoder:
  dtype = jnp.float32
  num_attn_heads = 8
  make_dropout = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,))
  make_layer_norm = layer_norm.T5LayerNorm

  def _make_encoder_layer(shared_relpos_bias):
    assert shared_relpos_bias is None
    return longt5_architecture.LongEncoderLayer(
        attention_factory=make_long_att_fn1(num_attn_heads, dtype),
        mlp=make_mlp1(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        relpos_bias_factory=(
            lambda: _make_relative_position_bias(num_attn_heads, dtype)))

  def _make_decoder_layer(shared_relative_position_bias):
    assert shared_relative_position_bias is None
    return t5_architecture.DecoderLayer(
        self_attention=make_attention1(num_attn_heads, dtype),
        encoder_decoder_attention=make_attention1(num_attn_heads, dtype),
        mlp=make_mlp1(dtype),
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        relative_position_bias_factory=(
            lambda: _make_relative_position_bias(num_attn_heads, dtype)))

  def _make_encoder(*, shared_token_embedder=None):
    return longt5_architecture.LongEncoder(
        num_layers=3,
        shared_token_embedder=shared_token_embedder,
        layer_factory=_make_encoder_layer,
        input_dropout_factory=make_dropout,
        output_dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        dtype=dtype,
    )

  def _make_decoder(*, shared_token_embedder=None):
    return t5_architecture.Decoder(
        num_layers=2,
        shared_token_embedder=shared_token_embedder,
        layer_factory=_make_decoder_layer,
        dropout_factory=make_dropout,
        layer_norm_factory=make_layer_norm,
        output_logits_factory=None,
        dtype=dtype,
    )

  return longt5_architecture.LongEncoderDecoder(
      shared_token_embedder_factory=lambda: make_token_emb1(71, dtype),
      encoder_factory=_make_encoder,
      decoder_factory=_make_decoder,
  )


class LongEncoderDecoderTest(absltest.TestCase):

  def test_encoder_shapes_with_relative_attention_per_layer(self):
    transformer = make_config1()
    inputs = np.array(
        [
            # Batch 1.
            [101, 183, 20, 75],
            # Batch 2.
            [101, 392, 19, 7],
        ],
        dtype=np.int32)
    output, variables = transformer.init_with_output(
        random.PRNGKey(0),
        inputs,
        enable_dropout=False,
        method=transformer.encode,
    )

    reformatted = transformer.apply({},
                                    variables['params'],
                                    method=transformer.to_save_format)
    check_dense_t5_params(reformatted,
                          'encoder_shapes_per_layer_relpos_bias.json')
    self.assertEqual(output.shape, (2, 4, 13))

    # Convert back to Flax module structure format and test again.
    params2 = transformer.apply({},
                                reformatted,
                                method=transformer.from_save_format)
    output2 = transformer.apply(
        {'params': params2},
        inputs,
        enable_dropout=False,
        method=transformer.encode,
    )
    np.testing.assert_allclose(output, output2, rtol=1e-8)

    # Compare with output from original T5 layers.
    transformer_t5 = make_config1_original_t5()
    output3, _ = transformer_t5.init_with_output(
        random.PRNGKey(0),
        inputs,
        enable_dropout=False,
        method=transformer_t5.encode,
    )
    np.testing.assert_allclose(output, output3, rtol=1e-8)

  def test_encoder_shapes_with_relative_attention_per_layer_scan(self):
    transformer = make_config1(scan_layers=True)
    inputs = np.array(
        [
            # Batch 1.
            [101, 183, 20, 75],
            # Batch 2.
            [101, 392, 19, 7],
        ],
        dtype=np.int32)
    output, _ = transformer.init_with_output(
        random.PRNGKey(0),
        inputs,
        enable_dropout=False,
        method=transformer.encode,
    )
    self.assertEqual(output.shape, (2, 4, 13))

    # Compare with output from original T5 layers.
    transformer_t5 = make_config1_original_t5(scan_layers=True)
    output2, _ = transformer_t5.init_with_output(
        random.PRNGKey(0),
        inputs,
        enable_dropout=False,
        method=transformer_t5.encode,
    )
    np.testing.assert_allclose(output, output2, rtol=1e-8)

  def test_encode_shared_relative_position_bias(self):
    transformer = make_config2_shared_relative_position_bias()
    inputs = np.array(
        [
            # Batch 1.
            [101, 183, 20, 75],
            # Batch 2.
            [101, 392, 19, 7],
        ],
        dtype=np.int32)
    output, variables = transformer.init_with_output(
        random.PRNGKey(0),
        inputs,
        enable_dropout=False,
        method=transformer.encode,
    )

    reformatted = transformer.apply({},
                                    variables['params'],
                                    method=transformer.to_save_format)
    check_dense_t5_params(reformatted, 'encoder_shapes_shared_relpos_bias.json')
    self.assertEqual(output.shape, (2, 4, 13))

    # Convert back to Flax module structure format and test again.
    params2 = transformer.apply({},
                                reformatted,
                                method=transformer.from_save_format)
    output2 = transformer.apply(
        {'params': params2},
        inputs,
        enable_dropout=False,
        method=transformer.encode,
    )
    np.testing.assert_allclose(output, output2, rtol=1e-8)

  def test_encoder_example_packing(self):
    transformer = make_config1()
    encoder_input_tokens = np.array(
        [
            # Batch 1.
            [101, 183, 20, 75],
            # Batch 2.
            [101, 392, 19, 0],
        ],
        dtype=np.int32)
    output, variables = transformer.init_with_output(
        random.PRNGKey(0),
        encoder_input_tokens,
        enable_dropout=False,
        method=transformer.encode,
    )

    encoder_input_tokens_packed = np.array([[101, 183, 20, 75, 101, 392, 19]],
                                           dtype=np.int32)
    encoder_segment_ids = np.array([[0, 0, 0, 0, 1, 1, 1]], dtype=np.int32)
    encoder_input_positions = np.array([[0, 1, 2, 3, 0, 1, 2]], dtype=np.int32)
    output_packed = transformer.apply(
        variables,
        encoder_input_tokens_packed,
        encoder_segment_ids=encoder_segment_ids,
        encoder_positions=encoder_input_positions,
        enable_dropout=False,
        method=transformer.encode,
    )


    # Check that the first element matches, which is entire first batch of the
    # padded setup, and the first 3 "tokens" of the packed example.
    np.testing.assert_allclose(
        output[0, :, :], output_packed[0, 0:4, :], rtol=1e-4, atol=1e-4)

    # Check that the second element matches, which is the first 3 "tokens" of
    # the padded example's second batch, and the last 3 of tokens the packed
    # example's first batch.
    np.testing.assert_allclose(
        output[1, 0:3, :], output_packed[0, 4:7, :], rtol=1e-4, atol=1e-4)

  def test_scan_and_remat(self):
    """Tests if encoder returns the same output for different scan/remat."""
    encoder_input_tokens = np.array(
        [
            # Batch 1.
            [101, 183, 20, 75],
            # Batch 2.
            [101, 392, 19, 7],
        ],
        dtype=np.int32)

    transformer1 = make_config1(scan_layers=False, layer_remat='none')
    output1, _ = transformer1.init_with_output(
        random.PRNGKey(0),
        encoder_input_tokens,
        enable_dropout=False,
        method=transformer1.encode,
    )

    transformer2 = make_config1(scan_layers=False, layer_remat='minimal')
    output2, _ = transformer2.init_with_output(
        random.PRNGKey(0),
        encoder_input_tokens,
        enable_dropout=False,
        method=transformer2.encode,
    )

    transformer3 = make_config1(scan_layers=False, layer_remat='full')
    output3, _ = transformer3.init_with_output(
        random.PRNGKey(0),
        encoder_input_tokens,
        enable_dropout=False,
        method=transformer3.encode,
    )

    transformer4 = make_config1(scan_layers=True, layer_remat='minimal')
    output4, _ = transformer4.init_with_output(
        random.PRNGKey(0),
        encoder_input_tokens,
        enable_dropout=False,
        method=transformer4.encode,
    )

    transformer5 = make_config1(scan_layers=True, layer_remat='full')
    output5, _ = transformer5.init_with_output(
        random.PRNGKey(0),
        encoder_input_tokens,
        enable_dropout=False,
        method=transformer5.encode,
    )

    # Check scan_layers=False results
    np.testing.assert_allclose(output1, output2, rtol=2e-4)
    np.testing.assert_allclose(output1, output3, atol=1e-5, rtol=1.5e-5)
    # Check scan_layers=True results
    np.testing.assert_allclose(output4, output5, rtol=1.5e-5)

  def test_entire_transformer_shared_embeds(self):
    encoder_input_tokens = np.zeros((16, 8), dtype=np.float32)
    decoder_input_tokens = np.zeros((16, 8), dtype=np.float32)
    decoder_target_tokens = np.zeros((16, 8), dtype=np.float32)

    transformer = make_config3_shared_token_embedder()
    output, variables = transformer.init_with_output(
        random.PRNGKey(0),
        encoder_input_tokens=encoder_input_tokens,
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        enable_dropout=False,
    )
    params = variables['params']
    reformatted = transformer.apply({},
                                    params,
                                    method=transformer.to_save_format)
    check_dense_t5_params(reformatted,
                          'encoder_decoder_shared_embedding_shapes.json')
    self.assertEqual(output.shape, (16, 8, 71))

    # Convert back to Flax module structure format and test again.
    params2 = transformer.apply({},
                                reformatted,
                                method=transformer.from_save_format)
    output2 = transformer.apply(
        {'params': params2},
        encoder_input_tokens=encoder_input_tokens,
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        enable_dropout=False,
    )
    np.testing.assert_allclose(output, output2, rtol=1e-8)

  def test_encoder_local_self_attention_example_packing(self):

    def make_config() -> longt5_architecture.LongEncoderDecoder:
      dtype = jnp.float32
      num_attn_heads = 8
      make_dropout = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,))
      make_layer_norm = layer_norm.T5LayerNorm

      def attention_factory(relpos_bias):
        return long_attention.EncoderLocalSelfAttention(
            num_heads=num_attn_heads,
            local_radius=2,
            dtype=dtype,
            qkv_features=512,
            head_dim=None,
            kernel_init=ATTENTION_KERNEL_INIT,
            bias_init=BIAS_INIT,
            use_bias=False,
            broadcast_dropout=True,
            dropout_rate=0.1,
            relpos_bias=relpos_bias)

      def _make_encoder_layer(shared_relpos_bias):
        assert shared_relpos_bias is None
        return longt5_architecture.LongEncoderLayer(
            attention_factory=attention_factory,
            mlp=make_mlp1(dtype),
            dropout_factory=make_dropout,
            layer_norm_factory=make_layer_norm,
            relpos_bias_factory=(
                lambda: _make_relpos_bias_general(num_attn_heads, dtype)))

      def _make_decoder_layer(shared_relative_position_bias):
        assert shared_relative_position_bias is None
        return t5_architecture.DecoderLayer(
            self_attention=make_attention1(num_attn_heads, dtype),
            encoder_decoder_attention=make_attention1(num_attn_heads, dtype),
            mlp=make_mlp1(dtype),
            dropout_factory=make_dropout,
            layer_norm_factory=make_layer_norm,
            relative_position_bias_factory=(
                lambda: _make_relative_position_bias(num_attn_heads, dtype)))

      def _make_encoder(shared_token_embedder):
        assert shared_token_embedder is None
        return longt5_architecture.LongEncoder(
            num_layers=3,
            token_embedder_factory=lambda: make_token_emb1(2_000, dtype),
            layer_factory=_make_encoder_layer,
            input_dropout_factory=make_dropout,
            output_dropout_factory=make_dropout,
            layer_norm_factory=make_layer_norm,
            dtype=dtype,
        )

      def _make_decoder(shared_token_embedder):
        assert shared_token_embedder is None
        return t5_architecture.Decoder(
            num_layers=2,
            token_embedder_factory=lambda: make_token_emb1(2_000, dtype),
            layer_factory=_make_decoder_layer,
            dropout_factory=make_dropout,
            layer_norm_factory=make_layer_norm,
            output_logits_factory=None,
            dtype=dtype,
        )

      return longt5_architecture.LongEncoderDecoder(
          shared_token_embedder_factory=lambda: None,
          encoder_factory=_make_encoder,
          decoder_factory=_make_decoder,
      )

    transformer = make_config()
    encoder_input_tokens = np.array(
        [
            # Batch 1.
            [101, 183, 20, 75],
            # Batch 2.
            [101, 392, 19, 0],
        ],
        dtype=np.int32)
    output, variables = transformer.init_with_output(
        random.PRNGKey(0),
        encoder_input_tokens,
        enable_dropout=False,
        method=transformer.encode,
    )

    encoder_input_tokens_packed = np.array([[101, 183, 20, 75, 101, 392, 19]],
                                           dtype=np.int32)
    encoder_segment_ids = np.array([[0, 0, 0, 0, 1, 1, 1]], dtype=np.int32)
    encoder_input_positions = np.array([[0, 1, 2, 3, 0, 1, 2]], dtype=np.int32)
    output_packed = transformer.apply(
        variables,
        encoder_input_tokens_packed,
        encoder_segment_ids=encoder_segment_ids,
        encoder_positions=encoder_input_positions,
        enable_dropout=False,
        method=transformer.encode,
    )


    # Check that the first element matches, which is entire first batch of the
    # padded setup, and the first 3 "tokens" of the packed example.
    np.testing.assert_allclose(
        output[0, :, :], output_packed[0, 0:4, :], rtol=1e-4)

    # Check that the second element matches, which is the first 3 "tokens" of
    # the padded example's second batch, and the last 3 of tokens the packed
    # example's first batch.
    np.testing.assert_allclose(
        output[1, 0:3, :], output_packed[0, 4:7, :], rtol=1e-4)

  def test_etc_transient_global_self_attention_example_packing(self):

    def make_config() -> longt5_architecture.LongEncoderDecoder:
      dtype = jnp.float32
      num_attn_heads = 8
      make_dropout = lambda: nn.Dropout(rate=0.1, broadcast_dims=(-2,))
      make_layer_norm = layer_norm.T5LayerNorm

      def attention_factory(relpos_bias, side_relpos_bias):
        return long_attention.EtcTransientGlobalSelfAttention(
            num_heads=num_attn_heads,
            tokens_per_block=3,
            local_radius=2,
            dtype=dtype,
            qkv_features=512,
            head_dim=None,
            kernel_init=ATTENTION_KERNEL_INIT,
            bias_init=BIAS_INIT,
            use_bias=False,
            broadcast_dropout=True,
            dropout_rate=0.1,
            relpos_bias=relpos_bias,
            side_relpos_bias=side_relpos_bias)

      def _make_encoder_layer(shared_relpos_bias):
        assert shared_relpos_bias is None
        return longt5_architecture.LongEncoderLayer(
            attention_factory=attention_factory,
            mlp=make_mlp1(dtype),
            dropout_factory=make_dropout,
            layer_norm_factory=make_layer_norm,
            relpos_bias_factory=(
                lambda: _make_relpos_bias_general(num_attn_heads, dtype)),
            side_relpos_bias_factory=(
                lambda: _make_relpos_bias_general(num_attn_heads, dtype)))

      def _make_decoder_layer(shared_relative_position_bias):
        assert shared_relative_position_bias is None
        return t5_architecture.DecoderLayer(
            self_attention=make_attention1(num_attn_heads, dtype),
            encoder_decoder_attention=make_attention1(num_attn_heads, dtype),
            mlp=make_mlp1(dtype),
            dropout_factory=make_dropout,
            layer_norm_factory=make_layer_norm,
            relative_position_bias_factory=(
                lambda: _make_relative_position_bias(num_attn_heads, dtype)))

      def _make_encoder(shared_token_embedder):
        assert shared_token_embedder is None
        return longt5_architecture.LongEncoder(
            num_layers=3,
            token_embedder_factory=lambda: make_token_emb1(2_000, dtype),
            layer_factory=_make_encoder_layer,
            input_dropout_factory=make_dropout,
            output_dropout_factory=make_dropout,
            layer_norm_factory=make_layer_norm,
            dtype=dtype,
        )

      def _make_decoder(shared_token_embedder):
        assert shared_token_embedder is None
        return t5_architecture.Decoder(
            num_layers=2,
            token_embedder_factory=lambda: make_token_emb1(2_000, dtype),
            layer_factory=_make_decoder_layer,
            dropout_factory=make_dropout,
            layer_norm_factory=make_layer_norm,
            output_logits_factory=None,
            dtype=dtype,
        )

      return longt5_architecture.LongEncoderDecoder(
          shared_token_embedder_factory=lambda: None,
          encoder_factory=_make_encoder,
          decoder_factory=_make_decoder,
      )

    transformer = make_config()
    encoder_input_tokens = np.array(
        [
            # Batch 1.
            [101, 183, 20, 75],
            # Batch 2.
            [101, 392, 19, 0],
        ],
        dtype=np.int32)
    output, variables = transformer.init_with_output(
        random.PRNGKey(0),
        encoder_input_tokens,
        enable_dropout=False,
        method=transformer.encode,
    )

    encoder_input_tokens_packed = np.array([[101, 183, 20, 75, 101, 392, 19]],
                                           dtype=np.int32)
    encoder_segment_ids = np.array([[0, 0, 0, 0, 1, 1, 1]], dtype=np.int32)
    encoder_input_positions = np.array([[0, 1, 2, 3, 0, 1, 2]], dtype=np.int32)
    output_packed = transformer.apply(
        variables,
        encoder_input_tokens_packed,
        encoder_segment_ids=encoder_segment_ids,
        encoder_positions=encoder_input_positions,
        enable_dropout=False,
        method=transformer.encode,
    )


    # Check that the first element matches, which is entire first batch of the
    # padded setup, and the first 3 "tokens" of the packed example.
    np.testing.assert_allclose(
        output[0, :, :], output_packed[0, 0:4, :], rtol=1e-4, atol=1e-6)

    # Check that the second element matches, which is the first 3 "tokens" of
    # the padded example's second batch, and the last 3 of tokens the packed
    # example's first batch.
    np.testing.assert_allclose(
        output[1, 0:3, :], output_packed[0, 4:7, :], rtol=1e-4, atol=1e-6)



if __name__ == '__main__':
  absltest.main()
