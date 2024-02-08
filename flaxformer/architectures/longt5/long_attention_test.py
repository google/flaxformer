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

"""Tests for long attention classes."""

import dataclasses
from typing import Optional

from absl.testing import absltest
from absl.testing import parameterized
from flax.core import frozen_dict
import jax
from jax import random
import jax.numpy as jnp
import numpy as np

from flaxformer.architectures.longt5 import long_attention
from flaxformer.architectures.longt5 import relative_position_biases_general
from flaxformer.components import relative_position_biases
from flaxformer.components.attention import dense_attention

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

RelativePositionBiasesGeneral = (
    relative_position_biases_general.RelativePositionBiasesGeneral)


@dataclasses.dataclass(frozen=True)
class EncoderLocalSelfAttArgs:
  num_heads: int = 1
  local_radius: int = 2
  batch_size: int = 2
  qkv_features: int = 3
  out_features: int = 4
  q_len: int = 5
  features: int = 6
  broadcast_dropout: bool = True
  dropout_rate: float = 0.1
  enable_dropout: bool = True
  use_bias: bool = True
  rescale_logits: bool = True
  float32_logits: bool = False
  split_head_kernel: bool = False
  kernels_to_fuse: Optional[str] = None  # Only 'qkv' is supported.
  relpos_bias: Optional[RelativePositionBiasesGeneral] = None

  def init_args(self):
    return dict(
        num_heads=self.num_heads,
        local_radius=self.local_radius,
        qkv_features=self.qkv_features,
        out_features=self.out_features,
        broadcast_dropout=self.broadcast_dropout,
        dropout_rate=self.dropout_rate,
        use_bias=self.use_bias,
        rescale_logits=self.rescale_logits,
        float32_logits=self.float32_logits,
        split_head_kernel=self.split_head_kernel,
        kernels_to_fuse=self.kernels_to_fuse,
        relpos_bias=self.relpos_bias)

  def apply_args(self):
    inputs = jnp.ones((self.batch_size, self.q_len, self.features))
    inputs_mask = jnp.ones((self.batch_size, self.q_len))
    return {
        'inputs': inputs,
        'inputs_mask': inputs_mask,
        'enable_dropout': self.enable_dropout
    }




class LongAttentionTest(parameterized.TestCase):


  def test_local_self_attention_shape(self):
    # This test only checks for shape but tries to make sure all code paths are
    # reached.
    dropout_rng = random.PRNGKey(0)
    batch_size, num_heads, seq_len, qk_depth, v_depth = 1, 2, 8, 3, 5
    local_radius = 1
    block_len = local_radius + 1  # 2
    num_blocks = seq_len // block_len + bool(seq_len % block_len)  # 4

    query = jnp.ones((batch_size, seq_len, num_heads, qk_depth))
    key = jnp.ones((batch_size, seq_len, num_heads, qk_depth))
    value = jnp.ones((batch_size, seq_len, num_heads, v_depth))
    bias = jnp.ones(
        (batch_size, num_blocks, num_heads, block_len, 3 * block_len))

    args = dict(
        query=query,
        key=key,
        value=value,
        local_radius=local_radius,
        bias=bias,
        rescale_logits=True,
        dropout_rng=dropout_rng,
        dropout_rate=0.5,
        enable_dropout=True,
    )

    output = long_attention._local_self_attention(
        **args, broadcast_dropout=True)
    self.assertEqual(output.shape, (batch_size, seq_len, num_heads, v_depth))

    # Make sure we also reach the code path where we don't broadcast dropout.
    output = long_attention._local_self_attention(
        **args, broadcast_dropout=False)
    self.assertEqual(output.shape, (batch_size, seq_len, num_heads, v_depth))

  def test_encoder_local_self_attention(self):
    rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    args = EncoderLocalSelfAttArgs()
    model = long_attention.EncoderLocalSelfAttention(**args.init_args())
    y, _ = model.init_with_output(rngs, **args.apply_args())
    self.assertEqual(y.shape, (args.batch_size, args.q_len, args.out_features))

  def test_encoder_local_self_attention_cast_logits_float32(self):
    rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    args = EncoderLocalSelfAttArgs(float32_logits=True)
    model = long_attention.EncoderLocalSelfAttention(**args.init_args())
    y, _ = model.init_with_output(rngs, **args.apply_args())
    self.assertEqual(y.shape, (args.batch_size, args.q_len, args.out_features))

  def test_encoder_local_self_attention_no_rescale_logits(self):
    rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    args = EncoderLocalSelfAttArgs(rescale_logits=False)
    model = long_attention.EncoderLocalSelfAttention(**args.init_args())
    y, _ = model.init_with_output(rngs, **args.apply_args())
    self.assertEqual(y.shape, (args.batch_size, args.q_len, args.out_features))

  def test_encoder_local_self_attention_no_out_features(self):
    rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    args = EncoderLocalSelfAttArgs(out_features=None)
    model = long_attention.EncoderLocalSelfAttention(**args.init_args())
    y, _ = model.init_with_output(rngs, **args.apply_args())
    self.assertEqual(y.shape, (args.batch_size, args.q_len, args.features))

  def test_encoder_local_self_attention_with_kernel_fusion(self):
    rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    args = EncoderLocalSelfAttArgs(
        split_head_kernel=True, kernels_to_fuse='qkv')
    model = long_attention.EncoderLocalSelfAttention(**args.init_args())
    y, _ = model.init_with_output(rngs, **args.apply_args())
    self.assertEqual(y.shape, (args.batch_size, args.q_len, args.out_features))

  @parameterized.named_parameters(
      ('even_blocking', 15),
      ('uneven_blocking', 16),
      ('degenerate_blocking', 35),
  )
  def test_encoder_local_self_attention_logic(self, local_radius):
    # This test checks the logic of the local attention calculations by
    # comparing with the output of `MultiHeadDotProductAttention`
    # (full attention) after manually applying the local sparsity pattern.
    # The outputs should be identical for non-padding tokens.

    keys = random.split(random.PRNGKey(0), 4)

    batch_size = 3
    seq_len = 64
    in_features = 11
    out_features = 12
    num_heads = 5
    dtype = jnp.float32

    inputs = random.normal(keys[0], (batch_size, seq_len, in_features))
    inputs_mask = random.bernoulli(keys[1], 0.9, (batch_size, seq_len))
    inputs_mask = inputs_mask.astype(jnp.bool_)
    segment_ids = jnp.cumsum(
        random.bernoulli(keys[2], 0.1, (batch_size, seq_len)), axis=-1)
    # `positions` is unused in `EncoderLocalSelfAttention`, so we set to zeros.
    positions = jnp.zeros_like(segment_ids)

    att_config = dict(
        num_heads=num_heads,
        dtype=dtype,
        qkv_features=20,
        out_features=out_features,
        use_rotary_embedding=True,
    )

    relpos_bias = RelativePositionBiasesGeneral(
        num_heads=num_heads, num_buckets=32, max_distance=128, dtype=dtype)

    local_att = long_attention.EncoderLocalSelfAttention(
        local_radius=local_radius,
        relpos_bias=relpos_bias,
        **att_config,
    )

    full_att = dense_attention.MultiHeadDotProductAttention(
        use_bias=True, **att_config
    )

    local_att_output, local_att_vars = local_att.init_with_output(
        keys[3],
        inputs,
        inputs_mask,
        segment_ids=segment_ids,
        positions=positions,
        enable_dropout=False)

    relpos_bias_vars = dict(params=local_att_vars['params']['relpos_bias'])

    # Full attention uses the same variables as local attention (ignoring
    # `relpos_bias`).
    full_att_vars = local_att_vars

    rp_bucket = relpos_bias.full_att_rp_bucket(
        qlen=seq_len, klen=seq_len, bidirectional=True)
    bias = relpos_bias.apply(relpos_bias_vars, rp_bucket)
    mask = dense_attention.make_attention_mask(
        inputs_mask, inputs_mask, dtype=dtype)
    mask = dense_attention.combine_masks(
        mask,
        dense_attention.make_attention_mask(
            segment_ids, segment_ids, jnp.equal, dtype=dtype))

    # Overlay local sparsity attention mask for full attention case.
    range_array = np.arange(seq_len)
    locality_mask = np.abs(range_array[np.newaxis, :] -
                           range_array[:, np.newaxis]) <= local_radius
    # [1, 1, seq_len, seq_len] shape
    locality_mask = locality_mask[np.newaxis, np.newaxis, :, :]
    mask = dense_attention.combine_masks(mask, locality_mask)

    full_att_output = full_att.apply(
        full_att_vars, inputs, inputs, mask, bias, enable_dropout=False)

    np.testing.assert_array_equal(local_att_output.shape,
                                  (batch_size, seq_len, out_features))
    np.testing.assert_array_equal(local_att_output.shape, full_att_output.shape)

    # Padding tokens may have different embeddings which we'll want to ignore
    # in our comparison, so we "clear" them to zero.
    def clear_padding(array):
      return array * inputs_mask[..., jnp.newaxis].astype(dtype)

    np.testing.assert_allclose(
        clear_padding(local_att_output),
        clear_padding(full_att_output),
        atol=1e-5)

  @parameterized.named_parameters(
      ('even_blocking', 15),
      ('uneven_blocking', 16),
      ('degenerate_blocking', 35),
      ('uneven_blocking_use_kernel_fusion', 16, True),
      ('even_blocking_causal', 15, False, True),
      ('uneven_blocking_causal', 16, False, True),
      ('degenerate_blocking_causal', 35, False, True),
      ('uneven_blocking_use_kernel_fusion_causal', 16, True, True),
  )
  def test_etc_transient_global_self_attention(self,
                                               local_radius,
                                               use_kernel_fusion=False,
                                               causal=False):
    # This test just makes sure the layer successfully runs with different
    # input sizes.

    keys = random.split(random.PRNGKey(0), 3)

    batch_size = 3
    seq_len = 64
    tokens_per_block = 4
    in_features = 11
    out_features = 12
    num_heads = 5
    dtype = jnp.float32

    inputs = random.normal(keys[0], (batch_size, seq_len, in_features))

    # Construct realistic packed inputs.
    new_segment_marker = random.bernoulli(keys[2], 0.1, (batch_size, seq_len))
    segment_ids = jnp.cumsum(new_segment_marker, axis=-1)
    # We make the last segment padding.
    is_padding = segment_ids == jnp.max(segment_ids, axis=-1, keepdims=True)
    inputs_mask = jnp.logical_not(is_padding)
    # Create positions based on segments.
    arange = np.broadcast_to(np.arange(seq_len), segment_ids.shape)
    positions = arange - np.maximum.accumulate(
        new_segment_marker * arange, axis=-1)
    positions *= inputs_mask

    relpos_bias = RelativePositionBiasesGeneral(
        num_heads=num_heads, num_buckets=32, max_distance=128, dtype=dtype)

    side_relpos_bias = RelativePositionBiasesGeneral(
        num_heads=num_heads, num_buckets=32, max_distance=128, dtype=dtype)

    att_layer = long_attention.EtcTransientGlobalSelfAttention(
        num_heads=num_heads,
        tokens_per_block=tokens_per_block,
        local_radius=local_radius,
        dtype=dtype,
        causal=causal,
        qkv_features=15,
        out_features=out_features,
        rescale_logits=use_kernel_fusion,
        split_head_kernel=use_kernel_fusion,
        kernels_to_fuse='kv' if use_kernel_fusion else None,
        relpos_bias=relpos_bias,
        side_relpos_bias=side_relpos_bias,
    )

    output, _ = att_layer.init_with_output(
        keys[3],
        inputs,
        inputs_mask,
        segment_ids=segment_ids,
        positions=positions,
        enable_dropout=False)

    np.testing.assert_array_equal(output.shape,
                                  (batch_size, seq_len, out_features))


  def test_make_etc_fixed_block_ids(self):
    # See this documentation for an example of what packed inputs look like:
    # https://github.com/google/seqio/blob/main/seqio/utils.py#L292
    inputs_mask = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #
        ],
        dtype=np.bool_)
    segment_ids = [
        [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3],  #
        [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 0, 0],  #
        [1, 1, 2, 3, 3, 4, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6],  #
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #
    ]
    positions = [
        [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 0, 1, 2],  #
        [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 0, 1, 2, 3, 0, 0],  #
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 2, 3, 4, 5, 6, 7],  #
        list(range(16)),  #
    ]
    block_ids, global_segment_ids = long_attention.make_etc_fixed_block_ids(
        tokens_per_block=3,
        inputs_mask=inputs_mask,
        segment_ids=segment_ids,
        positions=positions)

    np.testing.assert_array_equal(
        block_ids,
        [
            [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],  #
            [0, 0, 0, 1, 1, 1, 1, 1, -1, -1, 2, 2, 2, 2, -1, -1],  #
            [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 1, 1, 1, 1, 1],  #
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4],  #
        ])
    np.testing.assert_array_equal(
        global_segment_ids,
        [
            [1, 1, 2, 2, 3],  #
            [1, 1, 3, 0, 0],  #
            [6, 6, 0, 0, 0],  #
            [1, 1, 1, 1, 1],  #
        ])

  def test_make_etc_fixed_block_ids_without_packing(self):
    inputs_mask = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1, 0, 0],  #
            [1, 1, 1, 1, 1, 0, 0, 0],  #
        ],
        dtype=np.bool_)

    block_ids, global_segment_ids = long_attention.make_etc_fixed_block_ids(
        tokens_per_block=3, inputs_mask=inputs_mask)

    np.testing.assert_array_equal(
        block_ids,
        [
            [0, 0, 0, 1, 1, 1, 1, 1],  #
            [0, 0, 0, 1, 1, 1, -1, -1],  #
            [0, 0, 0, 0, 0, -1, -1, -1],  #
        ])
    np.testing.assert_array_equal(
        global_segment_ids,
        [
            [1, 1],  #
            [1, 1],  #
            [1, 0],  #
        ])

  def test_make_etc_fixed_block_ids_without_orphan_adoption(self):
    # See this documentation for an example of what packed inputs look like:
    # https://github.com/google/seqio/blob/main/seqio/utils.py#L292
    inputs_mask = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #
        ],
        dtype=np.bool_)
    segment_ids = [
        [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3],  #
        [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 0, 0],  #
        [1, 1, 2, 3, 3, 4, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6],  #
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #
    ]
    positions = [
        [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 0, 1, 2],  #
        [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 0, 1, 2, 3, 0, 0],  #
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 2, 3, 4, 5, 6, 7],  #
        list(range(16)),  #
    ]
    block_ids, global_segment_ids = long_attention.make_etc_fixed_block_ids(
        tokens_per_block=3,
        inputs_mask=inputs_mask,
        segment_ids=segment_ids,
        positions=positions,
        adopt_orphan_tokens=False)

    np.testing.assert_array_equal(
        block_ids,
        [
            [0, 0, 0, 1, 1, 1, -1, 2, 2, 2, 3, 3, 3, 4, 4, 4],  #
            [0, 0, 0, 1, 1, 1, -1, -1, -1, -1, 2, 2, 2, -1, -1, -1],  #
            [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1],  #
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, -1],  #
        ])
    np.testing.assert_array_equal(
        global_segment_ids,
        [
            [1, 1, 2, 2, 3],  #
            [1, 1, 3, 0, 0],  #
            [6, 6, 0, 0, 0],  #
            [1, 1, 1, 1, 1],  #
        ])

  def test_make_etc_fixed_block_ids_without_packing_nor_adoption(self):
    inputs_mask = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1, 0, 0],  #
            [1, 1, 1, 1, 1, 0, 0, 0],  #
        ],
        dtype=np.bool_)

    block_ids, global_segment_ids = long_attention.make_etc_fixed_block_ids(
        tokens_per_block=3, inputs_mask=inputs_mask, adopt_orphan_tokens=False)

    np.testing.assert_array_equal(
        block_ids,
        [
            [0, 0, 0, 1, 1, 1, -1, -1],  #
            [0, 0, 0, 1, 1, 1, -1, -1],  #
            [0, 0, 0, -1, -1, -1, -1, -1],  #
        ])
    np.testing.assert_array_equal(
        global_segment_ids,
        [
            [1, 1],  #
            [1, 1],  #
            [1, 0],  #
        ])

  def test_orphan_token_identification(self):
    inputs_mask = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1, 0, 0],  #
            [1, 1, 1, 1, 1, 0, 0, 0],  #
        ],
        dtype=np.bool_)

    orphan_tokens = long_attention.identify_orphan_tokens(
        tokens_per_block=3, inputs_mask=inputs_mask)
    np.testing.assert_array_equal(
        orphan_tokens,
        [
            [0, 0, 0, 0, 0, 0, 1, 1],  #
            [0, 0, 0, 0, 0, 0, 0, 0],  #
            [0, 0, 0, 1, 1, 0, 0, 0],  #
        ])

    inputs_mask = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #
        ],
        dtype=np.bool_)
    positions = [
        [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 0, 1, 2],  #
        [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 0, 1, 2, 3, 0, 0],  #
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 2, 3, 4, 5, 6, 7],  #
        list(range(16)),  #
    ]
    orphan_tokens = long_attention.identify_orphan_tokens(
        tokens_per_block=3, inputs_mask=inputs_mask, positions=positions)
    np.testing.assert_array_equal(
        orphan_tokens,
        [
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],  #
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  #
        ])

  def test_mask_to_bias(self):
    mask1 = np.array([1, 0], dtype=bool)
    bias1 = long_attention.mask_to_bias(mask1, dtype=np.float32)
    np.testing.assert_array_equal(bias1, np.array([0, -1e10], dtype=np.float32))
    assert bias1.dtype == np.float32

    mask2 = np.array([[1, 0], [0, 0]], dtype=bool)
    bias2 = long_attention.mask_to_bias(mask2, dtype=np.float32)
    np.testing.assert_array_equal(
        bias2, np.array([[0, -1e10], [-1e10, -1e10]], dtype=np.float32))
    assert bias2.dtype == np.float32

  def test_make_side_relpos(self):
    tokens_per_block = 3
    inputs_mask = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  #
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])  #
    positions = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  #
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0],  #
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2],  #
        [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]
    ])  #
    segment_ids = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],  #
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2],  #
        [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    ])  #

    side_relative_positions = long_attention._make_side_relpos(
        tokens_per_block,
        inputs_mask,
        segment_ids,
        positions,
        adopt_orphan_tokens=True)

    # certain biases are not important b/c they will be masked out; we represent
    # these with NaNs and ignore their positions in testing.
    x = np.nan
    expected_relative_positions = np.array([
        [
            [0, 1, 2, 3],  #
            [0, 1, 2, 3],  #
            [0, 1, 2, 3],  #
            [-1, 0, 1, 2],  #
            [-1, 0, 1, 2],  #
            [-1, 0, 1, 2],  #
            [-2, -1, 0, 1],  #
            [-2, -1, 0, 1],  #
            [-2, -1, 0, 1],  #
            [-3, -2, -1, 0],  #
            [-3, -2, -1, 0],  #
            [-3, -2, -1, 0]
        ],  #
        [
            [0, 1, 2, x],  #
            [0, 1, 2, x],  #
            [0, 1, 2, x],  #
            [-1, 0, 1, x],  #
            [-1, 0, 1, x],  #
            [-1, 0, 1, x],  #
            [-2, -1, 0, x],  #
            [-2, -1, 0, x],  #
            [-2, -1, 0, x],  #
            [-2, -1, 0, x],  #
            [x, x, x, x],  #
            [x, x, x, x]
        ],  #
        [
            [0, 1, 2, x],  #
            [0, 1, 2, x],  #
            [0, 1, 2, x],  #
            [-1, 0, 1, x],  #
            [-1, 0, 1, x],  #
            [-1, 0, 1, x],  #
            [-2, -1, 0, x],  #
            [-2, -1, 0, x],  #
            [-2, -1, 0, x],  #
            [x, x, x, 0],  #
            [x, x, x, 0],  #
            [x, x, x, 0]
        ],  #
        [
            [0, 1, 2, x],  #
            [0, 1, 2, x],  #
            [0, 1, 2, x],  #
            [-1, 0, 1, x],  #
            [-1, 0, 1, x],  #
            [-1, 0, 1, x],  #
            [-1, 0, 1, x],  #
            [-1, 0, 1, x],  #
            [x, x, 0, x],  #
            [x, x, 0, x],  #
            [x, x, 0, x],  #
            [x, x, 0, x]
        ]
    ])  #
    positions_to_compare = np.isfinite(expected_relative_positions)
    np.testing.assert_array_equal(
        side_relative_positions[positions_to_compare],
        expected_relative_positions[positions_to_compare])

    side_relative_positions = long_attention._make_side_relpos(
        tokens_per_block,
        inputs_mask,
        segment_ids,
        positions,
        adopt_orphan_tokens=False)
    expected_relative_positions = np.array([
        [
            [0, 1, 2, 3],  #
            [0, 1, 2, 3],  #
            [0, 1, 2, 3],  #
            [-1, 0, 1, 2],  #
            [-1, 0, 1, 2],  #
            [-1, 0, 1, 2],  #
            [-2, -1, 0, 1],  #
            [-2, -1, 0, 1],  #
            [-2, -1, 0, 1],  #
            [-3, -2, -1, 0],  #
            [-3, -2, -1, 0],  #
            [-3, -2, -1, 0]
        ],  #
        [
            [0, 1, 2, x],  #
            [0, 1, 2, x],  #
            [0, 1, 2, x],  #
            [-1, 0, 1, x],  #
            [-1, 0, 1, x],  #
            [-1, 0, 1, x],  #
            [-2, -1, 0, x],  #
            [-2, -1, 0, x],  #
            [-2, -1, 0, x],  #
            [-3, -2, -1, x],  #
            [x, x, x, x],  #
            [x, x, x, x]
        ],  #
        [
            [0, 1, 2, x],  #
            [0, 1, 2, x],  #
            [0, 1, 2, x],  #
            [-1, 0, 1, x],  #
            [-1, 0, 1, x],  #
            [-1, 0, 1, x],  #
            [-2, -1, 0, x],  #
            [-2, -1, 0, x],  #
            [-2, -1, 0, x],  #
            [x, x, x, 0],  #
            [x, x, x, 0],  #
            [x, x, x, 0]
        ],  #
        [
            [0, 1, 2, x],  #
            [0, 1, 2, x],  #
            [0, 1, 2, x],  #
            [-1, 0, 1, x],  #
            [-1, 0, 1, x],  #
            [-1, 0, 1, x],  #
            [-2, -1, 0, x],  #
            [-2, -1, 0, x],  #
            [x, x, 0, x],  #
            [x, x, 0, x],  #
            [x, x, 0, x],  #
            [x, x, -1, x]
        ]
    ])  #
    positions_to_compare = np.isfinite(expected_relative_positions)
    np.testing.assert_array_equal(
        side_relative_positions[positions_to_compare],
        expected_relative_positions[positions_to_compare])

    inputs_mask = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #
        ],
        dtype=np.bool_)
    segment_ids = [
        [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3],  #
        [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 0, 0],  #
        [1, 1, 2, 3, 3, 4, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6],  #
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #
    ]
    positions = [
        [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 0, 1, 2],  #
        [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 0, 1, 2, 3, 0, 0],  #
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 2, 3, 4, 5, 6, 7],  #
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  #
    ]
    side_relative_positions = long_attention._make_side_relpos(
        tokens_per_block,
        inputs_mask,
        segment_ids,
        positions,
        adopt_orphan_tokens=True)

    expected_relative_positions = np.array([
        [
            [0, 1, 2, 3, 4],  #
            [0, 1, 2, 3, 4],  #
            [0, 1, 2, 3, 4],  #
            [-1, 0, 1, 2, 3],  #
            [-1, 0, 1, 2, 3],  #
            [-1, 0, 1, 2, 3],  #
            [-1, 0, 1, 2, 3],  #
            [-2, -1, 0, 1, 2],  #
            [-2, -1, 0, 1, 2],  #
            [-2, -1, 0, 1, 2],  #
            [-3, -2, -1, 0, 1],  #
            [-3, -2, -1, 0, 1],  #
            [-3, -2, -1, 0, 1],  #
            [-4, -3, -2, -1, 0],  #
            [-4, -3, -2, -1, 0],  #
            [-4, -3, -2, -1, 0]
        ],  #
        [
            [0, 1, x, x, x],  #
            [0, 1, x, x, x],  #
            [0, 1, x, x, x],  #
            [-1, 0, x, x, x],  #
            [-1, 0, x, x, x],  #
            [-1, 0, x, x, x],  #
            [-1, 0, x, x, x],  #
            [-1, 0, x, x, x],  #
            [x, x, x, x, x],  #
            [x, x, x, x, x],  #
            [x, x, 0, x, x],  #
            [x, x, 0, x, x],  #
            [x, x, 0, x, x],  #
            [x, x, 0, x, x],  #
            [x, x, x, x, x],  #
            [x, x, x, x, x]
        ],  #
        [
            [x, x, x, x, x],  #
            [x, x, x, x, x],  #
            [x, x, x, x, x],  #
            [x, x, x, x, x],  #
            [x, x, x, x, x],  #
            [x, x, x, x, x],  #
            [x, x, x, x, x],  #
            [x, x, x, x, x],  #
            [0, 1, 2, 3, 4],  #
            [0, 1, 2, 3, 4],  #
            [0, 1, 2, 3, 4],  #
            [-1, 0, 1, 2, 3],  #
            [-1, 0, 1, 2, 3],  #
            [-1, 0, 1, 2, 3],  #
            [-1, 0, 1, 2, 3],  #
            [-1, 0, 1, 2, 3]
        ],  #
        [
            [0, 1, 2, 3, 4],  #
            [0, 1, 2, 3, 4],  #
            [0, 1, 2, 3, 4],  #
            [-1, 0, 1, 2, 3],  #
            [-1, 0, 1, 2, 3],  #
            [-1, 0, 1, 2, 3],  #
            [-2, -1, 0, 1, 2],  #
            [-2, -1, 0, 1, 2],  #
            [-2, -1, 0, 1, 2],  #
            [-3, -2, -1, 0, 1],  #
            [-3, -2, -1, 0, 1],  #
            [-3, -2, -1, 0, 1],  #
            [-4, -3, -2, -1, 0],  #
            [-4, -3, -2, -1, 0],  #
            [-4, -3, -2, -1, 0],  #
            [-4, -3, -2, -1, 0]
        ]
    ])  #

    positions_to_compare = np.isfinite(expected_relative_positions)
    np.testing.assert_array_equal(
        side_relative_positions[positions_to_compare],
        expected_relative_positions[positions_to_compare])

    side_relative_positions = long_attention._make_side_relpos(
        tokens_per_block,
        inputs_mask,
        segment_ids,
        positions,
        adopt_orphan_tokens=False)

    expected_relative_positions = np.array([
        [
            [0, 1, 2, 3, 4],  #
            [0, 1, 2, 3, 4],  #
            [0, 1, 2, 3, 4],  #
            [-1, 0, 1, 2, 3],  #
            [-1, 0, 1, 2, 3],  #
            [-1, 0, 1, 2, 3],  #
            [-2, -1, 0, 1, 2],  #
            [-2, -1, 0, 1, 2],  #
            [-2, -1, 0, 1, 2],  #
            [-2, -1, 0, 1, 2],  #
            [-3, -2, -1, 0, 1],  #
            [-3, -2, -1, 0, 1],  #
            [-3, -2, -1, 0, 1],  #
            [-4, -3, -2, -1, 0],  #
            [-4, -3, -2, -1, 0],  #
            [-4, -3, -2, -1, 0]
        ],  #
        [
            [0, 1, x, x, x],  #
            [0, 1, x, x, x],  #
            [0, 1, x, x, x],  #
            [-1, 0, x, x, x],  #
            [-1, 0, x, x, x],  #
            [-1, 0, x, x, x],  #
            [-2, -1, x, x, x],  #
            [-2, -1, x, x, x],  #
            [x, x, x, x, x],  #
            [x, x, x, x, x],  #
            [x, x, 0, x, x],  #
            [x, x, 0, x, x],  #
            [x, x, 0, x, x],  #
            [x, x, -1, x, x],  #
            [x, x, x, x, x],  #
            [x, x, x, x, x]
        ],  #
        [
            [x, x, x, x, x],  #
            [x, x, x, x, x],  #
            [x, x, x, x, x],  #
            [x, x, x, x, x],  #
            [x, x, x, x, x],  #
            [x, x, x, x, x],  #
            [x, x, x, x, x],  #
            [x, x, x, x, x],  #
            [0, 1, 2, 3, 4],  #
            [0, 1, 2, 3, 4],  #
            [0, 1, 2, 3, 4],  #
            [-1, 0, 1, 2, 3],  #
            [-1, 0, 1, 2, 3],  #
            [-1, 0, 1, 2, 3],  #
            [-2, -1, 0, 1, 2],  #
            [-2, -1, 0, 1, 2]
        ],  #
        [
            [0, 1, 2, 3, 4],  #
            [0, 1, 2, 3, 4],  #
            [0, 1, 2, 3, 4],  #
            [-1, 0, 1, 2, 3],  #
            [-1, 0, 1, 2, 3],  #
            [-1, 0, 1, 2, 3],  #
            [-2, -1, 0, 1, 2],  #
            [-2, -1, 0, 1, 2],  #
            [-2, -1, 0, 1, 2],  #
            [-3, -2, -1, 0, 1],  #
            [-3, -2, -1, 0, 1],  #
            [-3, -2, -1, 0, 1],  #
            [-4, -3, -2, -1, 0],  #
            [-4, -3, -2, -1, 0],  #
            [-4, -3, -2, -1, 0],  #
            [-5, -4, -3, -2, -1]
        ]
    ])  #

    positions_to_compare = np.isfinite(expected_relative_positions)
    np.testing.assert_array_equal(
        side_relative_positions[positions_to_compare],
        expected_relative_positions[positions_to_compare])

  def test_validate_long_att_call_params_positions_without_segment_ids(self):
    # This test ensures the `allow_positions_without_segment_ids` option
    # relaxes the default convention that requires both `positions` and
    # `segment_ids` to be given together.
    inputs = jnp.array([
        [[.1, .2], [.3, .4], [.5, .6], [.7, .8], [0., 0.]],  #
        [[.1, .2], [.3, .4], [.5, .6], [.7, .8], [.9, 0.]],  #
    ])
    inputs_mask = jnp.array([
        [1, 1, 1, 1, 0],  #
        [1, 1, 1, 1, 1],  #
    ])
    segment_ids = inputs_mask
    positions = jnp.array([
        [0, 1, 2, 3, 0],  #
        [0, 1, 2, 3, 4],  #
    ])

    # allow_positions_without_segment_ids=False (default)
    long_attention.validate_long_attention_call_parameter_shapes(
        inputs=inputs,
        inputs_mask=inputs_mask,
        positions=positions,
        segment_ids=segment_ids)
    long_attention.validate_long_attention_call_parameter_shapes(
        inputs=inputs,
        inputs_mask=inputs_mask,
        positions=None,
        segment_ids=None)
    with self.assertRaises(ValueError):
      long_attention.validate_long_attention_call_parameter_shapes(
          inputs=inputs,
          inputs_mask=inputs_mask,
          positions=positions,
          segment_ids=None)
    with self.assertRaises(ValueError):
      long_attention.validate_long_attention_call_parameter_shapes(
          inputs=inputs,
          inputs_mask=inputs_mask,
          positions=None,
          segment_ids=segment_ids)

    # allow_positions_without_segment_ids=True
    long_attention.validate_long_attention_call_parameter_shapes(
        inputs=inputs,
        inputs_mask=inputs_mask,
        positions=positions,
        segment_ids=segment_ids,
        allow_positions_without_segment_ids=True)
    long_attention.validate_long_attention_call_parameter_shapes(
        inputs=inputs,
        inputs_mask=inputs_mask,
        positions=None,
        segment_ids=None,
        allow_positions_without_segment_ids=True)
    long_attention.validate_long_attention_call_parameter_shapes(
        inputs=inputs,
        inputs_mask=inputs_mask,
        positions=positions,
        segment_ids=None,
        allow_positions_without_segment_ids=True)
    with self.assertRaises(ValueError):
      long_attention.validate_long_attention_call_parameter_shapes(
          inputs=inputs,
          inputs_mask=inputs_mask,
          positions=None,
          segment_ids=segment_ids,
          allow_positions_without_segment_ids=True)


if __name__ == '__main__':
  absltest.main()
