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

"""Tests for attention classes."""

import dataclasses
import functools
import itertools
from typing import Optional
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax_legacy.jax import quantization as aqt
from flax import linen as nn
from flax.core import freeze
from flax.core import unfreeze
from flax.linen import partitioning as flax_partitioning
import jax
from jax import dtypes
from jax import random
import jax.numpy as jnp
import numpy as np

from flaxformer import testing_utils
from flaxformer.components import dense
from flaxformer.components.attention import dense_attention
from flaxformer.types import Array

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()
AxisMetadata = flax_partitioning.AxisMetadata


class SelfAttention(dense_attention.MultiHeadDotProductAttention):
  """Self-attention special case of multi-head dot-product attention."""

  @nn.compact
  def __call__(
      self,
      inputs_q: Array,
      mask: Optional[Array] = None,
      bias: Optional[Array] = None,
      enable_dropout: bool = True,
  ):
    return super().__call__(
        inputs_q, inputs_q, mask, bias, enable_dropout=enable_dropout
    )


@dataclasses.dataclass(frozen=True)
class SelfAttentionArgs:
  num_heads: int = 1
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
  decode: bool = False
  float32_logits: bool = False
  use_rotary_embedding: bool = False

  def __post_init__(self):
    # If we are doing decoding, the query length should be 1, because are doing
    # autoregressive decoding where we feed one position at a time.
    assert not self.decode or self.q_len == 1

  def init_args(self):
    return dict(
        num_heads=self.num_heads,
        qkv_features=self.qkv_features,
        out_features=self.out_features,
        broadcast_dropout=self.broadcast_dropout,
        dropout_rate=self.dropout_rate,
        use_bias=self.use_bias,
        rescale_logits=self.rescale_logits,
        float32_logits=self.float32_logits,
        use_rotary_embedding=self.use_rotary_embedding,
    )

  def apply_args(self):
    inputs_q = jnp.ones((self.batch_size, self.q_len, self.features))
    mask = jnp.ones((self.batch_size, self.num_heads, self.q_len, self.q_len))
    bias = jnp.ones((self.batch_size, self.num_heads, self.q_len, self.q_len))
    return {
        'inputs_q': inputs_q,
        'mask': mask,
        'bias': bias,
        'enable_dropout': self.enable_dropout,
    }


class AttentionTest(parameterized.TestCase):

  def _mock_initializer(self, key, shape, dtype=jnp.float_, val=1.0):  # pylint: disable=unused-argument
    return jnp.ones(shape, dtypes.canonicalize_dtype(dtype)) * val

  def test_dot_product_attention_shape(self):
    # This test only checks for shape but tries to make sure all code paths are
    # reached.
    dropout_rng = random.PRNGKey(0)
    batch_size, num_heads, q_len, kv_len, qk_depth, v_depth = 1, 2, 3, 4, 5, 6

    query = jnp.ones((batch_size, q_len, num_heads, qk_depth))
    key = jnp.ones((batch_size, kv_len, num_heads, qk_depth))
    value = jnp.ones((batch_size, kv_len, num_heads, v_depth))
    bias = jnp.ones((batch_size, num_heads, q_len, kv_len))

    args = dict(
        query=query,
        key=key,
        value=value,
        bias=bias,
        rescale_logits=True,
        dropout_rng=dropout_rng,
        dropout_rate=0.5,
        enable_dropout=True,
    )

    output = dense_attention.dot_product_attention(
        **args, broadcast_dropout=True
    )
    self.assertEqual(output.shape, (batch_size, q_len, num_heads, v_depth))

    # Make sure we also reach the code path where we don't broadcast dropout.
    output = dense_attention.dot_product_attention(
        **args, broadcast_dropout=False
    )
    self.assertEqual(output.shape, (batch_size, q_len, num_heads, v_depth))

  def test_dot_product_attention_no_batch_dim(self):
    num_heads, q_len, kv_len, qk_depth, v_depth = 1, 2, 3, 4, 5
    query = jnp.ones((q_len, num_heads, qk_depth))
    key = jnp.ones((kv_len, num_heads, qk_depth))
    value = jnp.ones((kv_len, num_heads, v_depth))
    output = dense_attention.dot_product_attention(query, key, value)
    self.assertEqual(output.shape, (q_len, num_heads, v_depth))

  def test_self_attention(self):
    # We only test MultiHeadDotProductAttention through SelfAttention because
    # we are only shape checking anyway.
    rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    args = SelfAttentionArgs()
    model = SelfAttention(**args.init_args())
    y, _ = model.init_with_output(rngs, **args.apply_args())
    self.assertEqual(y.shape, (args.batch_size, args.q_len, args.out_features))

  def test_self_attention_cast_logits_float32(self):
    rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    args = SelfAttentionArgs(float32_logits=True)
    model = SelfAttention(**args.init_args())
    y, _ = model.init_with_output(rngs, **args.apply_args())
    self.assertEqual(y.shape, (args.batch_size, args.q_len, args.out_features))

  def test_self_attention_no_rescale_logits(self):
    rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    args = SelfAttentionArgs(rescale_logits=False)
    model = SelfAttention(**args.init_args())
    y, _ = model.init_with_output(rngs, **args.apply_args())
    self.assertEqual(y.shape, (args.batch_size, args.q_len, args.out_features))

  def test_self_attention_no_out_features(self):
    rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    args = SelfAttentionArgs(out_features=None)
    model = SelfAttention(**args.init_args())
    y, _ = model.init_with_output(rngs, **args.apply_args())
    self.assertEqual(y.shape, (args.batch_size, args.q_len, args.features))

  def test_self_attention_no_masking(self):
    rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    args = SelfAttentionArgs()
    model = SelfAttention(**args.init_args())
    apply_args = args.apply_args()
    apply_args['mask'] = None
    y, _ = model.init_with_output(rngs, **apply_args)
    self.assertEqual(y.shape, (args.batch_size, args.q_len, args.out_features))

  def test_self_attention_with_decoding(self):
    rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    args = SelfAttentionArgs(decode=True, q_len=1)
    model = SelfAttention(**args.init_args())
    apply_args = args.apply_args()
    apply_args['mask'] = None
    apply_args['bias'] = None
    params = model.init(rngs, **apply_args)
    y, _ = model.apply(
        params,
        **apply_args,
        mutable=['cache'],
        rngs={'dropout': random.PRNGKey(2)},
    )
    self.assertEqual(y.shape, (args.batch_size, args.q_len, args.out_features))

  def test_make_attention_mask_multiply_pairwise_fn(self):
    decoder_target_tokens = jnp.array([[7, 0, 0], [8, 5, 0]])
    attention_mask = dense_attention.make_attention_mask(
        decoder_target_tokens > 0, decoder_target_tokens > 0, dtype=jnp.int32
    )
    expected0 = jnp.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    expected1 = jnp.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    self.assertEqual(attention_mask.shape, (2, 1, 3, 3))
    np.testing.assert_array_equal(attention_mask[0, 0], expected0)
    np.testing.assert_array_equal(attention_mask[1, 0], expected1)

  def test_make_attention_mask_equal_pairwise_fn(self):
    segment_ids = jnp.array([[1, 1, 2, 2, 2, 0], [1, 1, 1, 2, 0, 0]])
    attention_mask = dense_attention.make_attention_mask(
        segment_ids, segment_ids, pairwise_fn=jnp.equal, dtype=jnp.int32
    )
    # Padding is not treated in a special way. So they need to be zeroed out
    # separately.
    expected0 = jnp.array([
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ])
    expected1 = jnp.array([
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1],
    ])
    self.assertEqual(attention_mask.shape, (2, 1, 6, 6))
    np.testing.assert_array_equal(attention_mask[0, 0], expected0)
    np.testing.assert_array_equal(attention_mask[1, 0], expected1)

  def test_make_causal_mask_with_padding(self):
    x = jnp.array([[7, 0, 0], [8, 5, 0]])
    y = dense_attention.make_causal_mask(x)
    self.assertEqual(y.shape, (2, 1, 3, 3))
    # Padding is not treated in a special way. So they need to be zeroed out
    # separately.
    expected_y = jnp.array(
        [[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]], jnp.float32
    )
    np.testing.assert_allclose(y[0], expected_y)
    np.testing.assert_allclose(y[1], expected_y)

  def test_make_causal_mask_extra_batch_dims(self):
    x = jnp.ones((3, 3, 5))
    y = dense_attention.make_causal_mask(x, extra_batch_dims=2)
    self.assertEqual(y.shape, (1, 1, 3, 3, 1, 5, 5))

  def test_make_causal_mask(self):
    x = jnp.ones((1, 3))
    y = dense_attention.make_causal_mask(x)
    self.assertEqual(y.shape, (1, 1, 3, 3))
    expected_y = jnp.array(
        [[[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]]], jnp.float32
    )
    np.testing.assert_allclose(y, expected_y)

  def test_combine_masks(self):
    masks = [
        jnp.array([0, 1, 0, 1], jnp.float32),
        None,
        jnp.array([1, 1, 1, 1], jnp.float32),
        jnp.array([1, 1, 1, 0], jnp.float32),
    ]
    y = dense_attention.combine_masks(*masks)
    np.testing.assert_allclose(y, jnp.array([0, 1, 0, 0], jnp.float32))

  def test_combine_biases(self):
    masks = [
        jnp.array([0, 1, 0, 1], jnp.float32),
        None,
        jnp.array([0, 1, 1, 1], jnp.float32),
        jnp.array([0, 1, 1, 0], jnp.float32),
    ]
    y = dense_attention.combine_biases(*masks)
    np.testing.assert_allclose(y, jnp.array([0, 3, 2, 2], jnp.float32))

  def test_make_decoder_mask_lm_unpacked(self):
    decoder_target_tokens = jnp.array([6, 7, 3, 0])
    mask = dense_attention.make_decoder_mask(
        decoder_target_tokens=decoder_target_tokens, dtype=jnp.float32
    )
    expected_mask = jnp.array(
        [[[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]]]
    )
    np.testing.assert_array_equal(mask, expected_mask)

  def test_make_decoder_mask_lm_packed(self):
    decoder_target_tokens = jnp.array([[6, 7, 3, 4, 5, 0]])
    decoder_segment_ids = jnp.array([[1, 1, 1, 2, 2, 0]])
    mask = dense_attention.make_decoder_mask(
        decoder_target_tokens=decoder_target_tokens,
        dtype=jnp.float32,
        decoder_segment_ids=decoder_segment_ids,
    )
    expected_mask = jnp.array([[[
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ]]])
    np.testing.assert_array_equal(mask, expected_mask)

  def test_make_decoder_mask_prefix_lm_unpacked(self):
    decoder_target_tokens = jnp.array([[5, 6, 7, 3, 4, 0]])
    decoder_causal_attention = jnp.array([[1, 1, 1, 0, 0, 0]])
    mask = dense_attention.make_decoder_mask(
        decoder_target_tokens=decoder_target_tokens,
        dtype=jnp.float32,
        decoder_causal_attention=decoder_causal_attention,
    )
    expected_mask = jnp.array(
        [[[
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ]]],
        dtype=jnp.float32,
    )
    np.testing.assert_array_equal(mask, expected_mask)

  def test_make_decoder_mask_prefix_lm_packed(self):
    decoder_target_tokens = jnp.array([[5, 6, 7, 8, 3, 4, 0]])
    decoder_segment_ids = jnp.array([[1, 1, 1, 2, 2, 2, 0]])
    decoder_causal_attention = jnp.array([[1, 1, 0, 1, 1, 0, 0]])
    mask = dense_attention.make_decoder_mask(
        decoder_target_tokens=decoder_target_tokens,
        dtype=jnp.float32,
        decoder_causal_attention=decoder_causal_attention,
        decoder_segment_ids=decoder_segment_ids,
    )
    expected_mask = jnp.array([[[
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]]])
    np.testing.assert_array_equal(mask, expected_mask)

  def test_make_decoder_mask_prefix_lm_unpacked_multiple_elements(self):
    decoder_target_tokens = jnp.array([[6, 7, 3, 0], [4, 5, 0, 0]])
    decoder_causal_attention = jnp.array([[1, 1, 0, 0], [1, 0, 0, 0]])
    mask = dense_attention.make_decoder_mask(
        decoder_target_tokens=decoder_target_tokens,
        dtype=jnp.float32,
        decoder_causal_attention=decoder_causal_attention,
    )
    expected_mask0 = jnp.array(
        [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]]
    )
    expected_mask1 = jnp.array(
        [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    )
    self.assertEqual(mask.shape, (2, 1, 4, 4))
    np.testing.assert_array_equal(mask[0, 0], expected_mask0)
    np.testing.assert_array_equal(mask[1, 0], expected_mask1)

  def test_make_decoder_mask_composite_causal_attention(self):
    decoder_target_tokens = jnp.array([[6, 7, 3, 4, 8, 9, 0]])
    decoder_causal_attention = jnp.array([[1, 1, 0, 0, 1, 1, 0]])
    mask = dense_attention.make_decoder_mask(
        decoder_target_tokens=decoder_target_tokens,
        dtype=jnp.float32,
        decoder_causal_attention=decoder_causal_attention,
    )
    expected_mask0 = jnp.array([
        [1, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])

    self.assertEqual(mask.shape, (1, 1, 7, 7))
    np.testing.assert_array_equal(mask[0, 0], expected_mask0)

  def test_make_decoder_mask_composite_causal_attention_packed(self):
    decoder_target_tokens = jnp.array([[6, 7, 3, 4, 8, 9, 2, 3, 4]])
    decoder_segment_ids = jnp.array([[1, 1, 1, 1, 1, 1, 2, 2, 2]])
    decoder_causal_attention = jnp.array([[1, 1, 0, 0, 1, 1, 1, 1, 0]])
    mask = dense_attention.make_decoder_mask(
        decoder_target_tokens=decoder_target_tokens,
        dtype=jnp.float32,
        decoder_causal_attention=decoder_causal_attention,
        decoder_segment_ids=decoder_segment_ids,
    )
    expected_mask0 = jnp.array([
        [1, 1, 0, 0, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1],
    ])

    self.assertEqual(mask.shape, (1, 1, 9, 9))
    np.testing.assert_array_equal(mask[0, 0], expected_mask0)

  @parameterized.parameters({'f': 20}, {'f': 22})
  def test_multihead_dot_product_attention(self, f):
    # b: batch, f: qkv_features, q: q_len, k: kv_len, h: num_head, d: head_dim
    b, q, h, d, k = 2, 3, 4, 5, 6

    base_args = SelfAttentionArgs(
        num_heads=h,
        qkv_features=f,
        out_features=f,
        dropout_rate=0,
        rescale_logits=False,
        use_bias=False,
    )
    args = base_args.init_args()

    if f != h * d:
      args['head_dim'] = d

    np.random.seed(0)
    inputs_q = np.random.randn(b, q, f)
    inputs_kv = np.random.randn(b, k, f)

    # Projection: [b, q, f] -> [b, q, h, d]
    # So the kernels have to be [f, h, d]
    query_kernel = np.random.randn(f, h, d)
    key_kernel = np.random.randn(f, h, d)
    value_kernel = np.random.randn(f, h, d)
    # `out` calculation: [b, q, h, d] -> [b, q, f]
    # So kernel has to be [h, d, f]
    out_kernel = np.random.randn(h, d, f)

    params = {
        'query': {'kernel': query_kernel.reshape(f, -1)},
        'key': {'kernel': key_kernel.reshape(f, -1)},
        'value': {'kernel': value_kernel.reshape(f, -1)},
        'out': {'kernel': out_kernel.reshape(-1, f)},
    }
    y = dense_attention.MultiHeadDotProductAttention(**args).apply(
        {'params': freeze(params)}, inputs_q, inputs_kv
    )

    query = np.einsum('bqf,fhd->bqhd', inputs_q, query_kernel)
    key = np.einsum('bkf,fhd->bkhd', inputs_kv, key_kernel)
    value = np.einsum('bkf,fhd->bkhd', inputs_kv, value_kernel)
    logits = np.einsum('bqhd,bkhd->bhqk', query, key)
    weights = nn.softmax(logits, axis=-1)
    combined_value = np.einsum('bhqk,bkhd->bqhd', weights, value)
    y_expected = np.einsum('bqhd,hdf->bqf', combined_value, out_kernel)
    np.testing.assert_allclose(y, y_expected, rtol=1e-5, atol=1e-5)

  def test_multihead_dot_product_attention_prefill_caching(self):
    # b: batch, f: qkv_features, k: kv_len, h: num_head, d: head_dim
    b, h, d, k = 2, 3, 4, 5
    f = h * d
    prefill_lengths = np.array([3, 1])

    base_args = SelfAttentionArgs(
        num_heads=h, qkv_features=f, out_features=f, dropout_rate=0
    )
    args = base_args.init_args()

    cache = {
        'cached_key': np.zeros((b, h, d, k)),
        'cached_value': np.zeros((b, h, d, k)),
        'cache_index': np.array([0, 0]),
    }
    inputs_q = np.random.randn(b, k, f)
    inputs_kv = np.random.randn(b, k, f)

    # Mock dense general such that q, k, v projections are replaced by simple
    # reshaping.
    def mock_dense_general(self, x, **kwargs):  # pylint: disable=unused-argument
      return x.reshape(b, -1, h, d)

    with mock.patch.object(
        dense.DenseGeneral, '__call__', new=mock_dense_general
    ):
      _, mutated = dense_attention.MultiHeadDotProductAttention(**args).apply(
          {'cache': freeze(cache)},
          inputs_q,
          inputs_kv,
          decode=False,
          prefill=True,
          prefill_lengths=prefill_lengths,
          mutable=['cache'],
      )
      updated_cache = mutated['cache']

    # Perform the same mocked projection to generate the expected cache.
    # (key|value): [b, 1, h, d]
    key = mock_dense_general(None, inputs_kv)
    value = mock_dense_general(None, inputs_kv)

    # cached_(key|value): [b, h, d, k]
    # Update the our gold cache with the key and values that are part of the
    # prefix that we are prefilling the cache with. Explicit loops here avoid a
    # confusing transpose.
    for b, prefill_length in enumerate(prefill_lengths):
      for i in range(prefill_length):
        cache['cached_key'][b, :, :, i] = key[b, i, :, :]
        cache['cached_value'][b, :, :, i] = value[b, i, :, :]
      cache['cache_index'][b] = prefill_length
    for name, array in cache.items():
      np.testing.assert_allclose(array, updated_cache[name])

  def test_multihead_dot_product_attention_caching(self):
    # b: batch, f: qkv_features, k: kv_len, h: num_head, d: head_dim
    b, h, d, k = 2, 3, 4, 5
    f = h * d

    base_args = SelfAttentionArgs(
        num_heads=h, qkv_features=f, out_features=f, dropout_rate=0
    )
    args = base_args.init_args()

    cache = {
        'cached_key': np.zeros((b, h, d, k)),
        'cached_value': np.zeros((b, h, d, k)),
        'cache_index': np.array(0),
    }
    inputs_q = np.random.randn(b, 1, f)
    inputs_kv = np.random.randn(b, 1, f)

    # Mock dense general such that q, k, v projections are replaced by simple
    # reshaping.
    def mock_dense_general(self, x, **kwargs):  # pylint: disable=unused-argument
      return x.reshape(b, -1, h, d)

    with mock.patch.object(
        dense.DenseGeneral, '__call__', new=mock_dense_general
    ):
      _, mutated = dense_attention.MultiHeadDotProductAttention(**args).apply(
          {'cache': freeze(cache)},
          inputs_q,
          inputs_kv,
          decode=True,
          mutable=['cache'],
      )
      updated_cache = mutated['cache']

    # Perform the same mocked projection to generate the expected cache.
    # (key|value): [b, 1, h, d]
    key = mock_dense_general(None, inputs_kv)
    value = mock_dense_general(None, inputs_kv)

    # cached_(key|value): [b, h, d, k]
    cache['cached_key'][:, :, :, 0] = key[:, 0, :, :]
    cache['cached_value'][:, :, :, 0] = value[:, 0, :, :]
    cache['cache_index'] = np.array(1)
    for name, array in cache.items():
      np.testing.assert_allclose(array, updated_cache[name])

  def test_dot_product_attention(self):
    # b: batch, f: qkv_features, q: q_len, k: kv_len, h: num_head, d: head_dim
    b, q, h, d, k = 2, 3, 4, 5, 6
    np.random.seed(0)
    query = np.random.randn(b, q, h, d)
    key = np.random.randn(b, k, h, d)
    value = np.random.randn(b, k, h, d)
    bias = np.random.randn(b, h, q, k)
    attn_out = dense_attention.dot_product_attention(
        query, key, value, bias=bias
    )
    logits = np.einsum('bqhd,bkhd->bhqk', query, key)
    weights = jax.nn.softmax(logits + bias, axis=-1)
    expected = np.einsum('bhqk,bkhd->bqhd', weights, value)
    np.testing.assert_allclose(attn_out, expected, atol=1e-6)

  def test_dot_product_attention_rescale_weights(self):
    b, q, h, d, k = 2, 3, 4, 5, 6
    np.random.seed(0)
    query = np.random.randn(b, q, h, d)
    key = np.random.randn(b, k, h, d)
    value = np.random.randn(b, k, h, d)
    bias = np.random.randn(b, h, q, k)
    attn_out = dense_attention.dot_product_attention(
        query, key, value, bias=bias, rescale_weights=True
    )
    logits = np.einsum('bqhd,bkhd->bhqk', query, key) / np.sqrt(d)
    weights = jax.nn.softmax(logits + bias, axis=-1)
    expected = np.einsum('bhqk,bkhd->bqhd', weights, value)
    np.testing.assert_allclose(attn_out, expected, atol=1e-6)

  def test_dot_product_attention_rescale_both(self):
    b, q, h, d, k = 2, 3, 4, 5, 6
    np.random.seed(0)
    query = np.random.randn(b, q, h, d)
    key = np.random.randn(b, k, h, d)
    value = np.random.randn(b, k, h, d)
    # Rescaling twice is not supported.
    with self.assertRaises(ValueError):
      dense_attention.dot_product_attention(
          query, key, value, rescale_logits=True, rescale_weights=True
      )

  @parameterized.parameters({'f': 20}, {'f': 22})
  def test_multiquery_dot_product_attention(self, f):
    # b: batch, f: qkv_features, q: q_len, k: kv_len, h: num_head, d: head_dim
    b, q, h, d, k = 2, 3, 4, 5, 6

    base_args = SelfAttentionArgs(
        num_heads=h,
        qkv_features=f,
        out_features=f,
        dropout_rate=0,
        rescale_logits=False,
        use_bias=False,
    )
    args = base_args.init_args()

    if f != h * d:
      args['head_dim'] = d

    np.random.seed(0)
    inputs_q = np.random.randn(b, q, f)
    inputs_kv = np.random.randn(b, k, f)

    # Projection: [b, q, f] -> [b, q, h, d]
    # So the query kernel has to be [f, h, d]
    query_kernel = np.random.randn(f, h, d)
    key_kernel = np.random.randn(f, d)
    value_kernel = np.random.randn(f, d)
    # `out` calculation: [b, q, h, d] -> [b, q, f]
    # So kernel has to be [h, d, f]
    out_kernel = np.random.randn(h, d, f)

    params = {
        'query': {'kernel': query_kernel.reshape(f, -1)},
        'key': {'kernel': key_kernel},
        'value': {'kernel': value_kernel},
        'out': {'kernel': out_kernel.reshape(-1, f)},
    }
    y = dense_attention.MultiQueryDotProductAttention(**args).apply(
        {'params': freeze(params)}, inputs_q, inputs_kv
    )

    query = np.einsum('bqf,fhd->bqhd', inputs_q, query_kernel)
    key = np.einsum('bkf,fd->bkd', inputs_kv, key_kernel)
    value = np.einsum('bkf,fd->bkd', inputs_kv, value_kernel)
    logits = np.einsum('bqhd,bkd->bhqk', query, key)
    weights = nn.softmax(logits, axis=-1)
    combined_value = np.einsum('bhqk,bkd->bqhd', weights, value)
    y_expected = np.einsum('bqhd,hdf->bqf', combined_value, out_kernel)
    np.testing.assert_allclose(y, y_expected, atol=2e-4, rtol=1e-4)

  @parameterized.named_parameters([
      dict(
          testcase_name='multi_head',
          attn_class=dense_attention.MultiHeadDotProductAttention,
      ),
      dict(
          testcase_name='multi_query',
          attn_class=dense_attention.MultiQueryDotProductAttention,
      ),
  ])
  def test_attention_prefill_logits_match_forward(self, attn_class):
    """Make sure values during a cache prefill match values from training."""
    # b: batch, k: kv_len, h: num_head, d: head_dim t: sequence length
    b, h, d, t = 2, 3, 5, 6
    ls = np.array([6, 4]).astype(np.int32)
    f = h * d

    base_args = SelfAttentionArgs(
        num_heads=h, qkv_features=f, out_features=f, dropout_rate=0
    )
    args = base_args.init_args()

    inputs_q = np.random.randn(b, t, f).astype(np.float32)
    inputs_kv = np.random.randn(b, t, f).astype(np.float32)
    bias = np.random.randn(1, h, t, t).astype(np.float32)
    mask = dense_attention.make_decoder_mask(
        (np.arange(t) < np.reshape(ls, (-1, 1))).astype(inputs_q.dtype),
        dtype=inputs_q.dtype,
    ).astype(np.float32)

    attn = attn_class(**args)
    params = attn.init(jax.random.PRNGKey(0), inputs_q, inputs_kv, mask, bias)[
        'params'
    ]

    # Calculate logits as done during training, no caching or anything.
    logits = attn.apply(
        {'params': params},
        inputs_q,
        inputs_kv,
        mask=mask,
        bias=bias,
        enable_dropout=False,
        decode=False,
        prefill=False,
    )

    # Initialize the cache.
    _, variables_with_cache = attn.apply(
        {'params': params},
        inputs_q,
        inputs_kv,
        mask=mask,
        bias=bias,
        decode=True,
        prefill=False,
        mutable=['cache'],
    )
    cache = variables_with_cache['cache']
    # Calculate the logits returned during the cache prefill step. Actions
    # taken to facilitate caching should not effect the output.
    prefill_logits, _ = attn.apply(
        {'params': params, 'cache': cache},
        inputs_q,
        inputs_kv,
        mask=mask,
        bias=bias,
        enable_dropout=False,
        decode=False,
        prefill=True,
        prefill_lengths=ls,
        mutable=['cache'],
    )

    np.testing.assert_allclose(
        prefill_logits, logits, err_msg='logits do not match.'
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='multi_head',
          attn_class=dense_attention.MultiHeadDotProductAttention,
      ),
      dict(
          testcase_name='multi_query',
          attn_class=dense_attention.MultiQueryDotProductAttention,
      ),
      dict(
          testcase_name='one_head',
          attn_class=dense_attention.MultiHeadDotProductAttention,
          num_heads=1,
      ),
  ])
  def test_rotary_embedding_attention(self, attn_class, num_heads=3):
    """Makes sure enabling rotary embeddings works."""
    # b: batch, k: kv_len, h: num_head, d: head_dim t: sequence length
    b, h, d, t = 2, num_heads, 4, 8
    ls = np.array([6, 4]).astype(np.int32)
    f = h * d

    base_args = SelfAttentionArgs(
        num_heads=h,
        qkv_features=f,
        out_features=f,
        dropout_rate=0,
        use_rotary_embedding=True,
    )
    args = base_args.init_args()

    inputs_q = np.random.randn(b, t, f).astype(np.float32)
    inputs_kv = np.random.randn(b, t, f).astype(np.float32)
    bias = np.random.randn(1, h, t, t).astype(np.float32)
    mask = dense_attention.make_decoder_mask(
        (np.arange(t) < np.reshape(ls, (-1, 1))).astype(inputs_q.dtype),
        dtype=inputs_q.dtype,
    ).astype(np.float32)

    attn = attn_class(**args)
    params = attn.init(jax.random.PRNGKey(0), inputs_q, inputs_kv, mask, bias)[
        'params'
    ]

    # Calculate logits as done during training, no caching or anything.
    logits = attn.apply(
        {'params': params},
        inputs_q,
        inputs_kv,
        mask=mask,
        bias=bias,
        enable_dropout=False,
        decode=False,
        prefill=False,
    )

    self.assertEqual(logits.shape, (b, t, f))

  @parameterized.named_parameters([
      dict(
          testcase_name='multi_head_causal',
          attn_class=dense_attention.MultiHeadDotProductAttention,
          causal=True,
      ),
      dict(
          testcase_name='multi_query_causal',
          attn_class=dense_attention.MultiQueryDotProductAttention,
          causal=True,
      ),
      dict(
          testcase_name='multi_head',
          attn_class=dense_attention.MultiHeadDotProductAttention,
          causal=False,
      ),
      dict(
          testcase_name='multi_query',
          attn_class=dense_attention.MultiQueryDotProductAttention,
          causal=False,
      ),
  ])
  def test_final_prefill_logits_match_first_decode(self, attn_class, causal):
    """Check logits of final input position matches in prefill and decode.

    The position of the final input token is a special case where the input to
    the model is the last input token but the output is the logits for the first
    output token. During decoding, we need to use these logits for the first
    outputs to select the next token to feed into the model. This means we
    cannot pre-cache this final position, it needs to be calculated as the first
    step in decode model.

    However, when using a prefix-LM with full visibility within the inputs, we
    also need to include this position in calculation of the rest of the tokens.

    This test validates that this final position is considered during prefilling
    and is calculated with the same attention mask by checking the value of the
    logits. During prefilling, this position should have full visibility to all
    previous tokens (either via bidirectional attention in the input or by
    virtue of being the last token with a causal mask). During decoding, it will
    also have full visibility via the causal mask. Therefore, the logits for
    this position that is output from the prefill call should match the
    (re-computation of this position) in decode mode.

    Args:
      attn_class: The class for the attention type we are testing.
      causal: Whether the input tokens have causal masking or bidirectional
        attention.
    """
    with jax.default_matmul_precision('float32'):
      # b: batch, k: kv_len, h: num_head, d: head_dim t: sequence length
      b, h, d, t = 2, 3, 5, 8
      lengths = np.array([6, 4]).astype(np.int32)
      f = h * d

      base_args = SelfAttentionArgs(
          num_heads=h,
          qkv_features=f,
          out_features=f,
          dropout_rate=0,
          float32_logits=True,
      )
      args = base_args.init_args()

      inputs_q = np.random.randn(b, t, f).astype(np.float32)
      inputs_kv = np.random.randn(b, t, f).astype(np.float32)
      bias = np.random.randn(1, h, t, t).astype(np.float32)
      # For this test we need the final token (at our prefill length) to be
      # considered in the attention like it will when it is the first decode
      # token.
      valid_tokens = (np.arange(t) <= np.reshape(lengths, (-1, 1))).astype(
          inputs_q.dtype
      )
      last_valid = np.take_along_axis(
          valid_tokens, np.expand_dims(lengths, axis=-1), axis=1
      )
      assert np.all(last_valid == np.ones((2, 1), dtype=last_valid.dtype))
      mask = dense_attention.make_decoder_mask(
          valid_tokens,
          # Use bidirectional attention in the input.
          decoder_causal_attention=None if causal else valid_tokens,
          dtype=inputs_q.dtype,
      )

      attn = attn_class(**args, precision='float32')
      params = attn.init(
          jax.random.PRNGKey(0), inputs_q, inputs_kv, mask, bias
      )['params']

      # Initialize the cache
      _, variables_with_cache = attn.apply(
          {'params': params},
          inputs_q,
          inputs_kv,
          decode=True,
          prefill=False,
          mutable=['cache'],
      )
      cache = variables_with_cache['cache']
      # Prefill the cache and select the logits from the position of the final
      # input token.
      prefilled_logits, vars_with_new_cache = attn.apply(
          {
              'params': params,
              'cache': cache,
          },
          inputs_q,
          inputs_kv,
          mask=mask,
          bias=bias,
          enable_dropout=False,
          decode=False,
          prefill=True,
          prefill_lengths=lengths,
          mutable=['cache'],
      )
      prefilled_cache = vars_with_new_cache['cache']

      lengths_index = jnp.reshape(lengths, (-1, 1, 1))
      final_prefilled_logits = jnp.take_along_axis(
          prefilled_logits, lengths_index, axis=1
      )

      # Do a single decode step, with the final input token as input.
      decode_logits, _ = attn.apply(
          {'params': params, 'cache': prefilled_cache},
          jnp.take_along_axis(inputs_q, lengths_index, axis=1),
          jnp.take_along_axis(inputs_kv, lengths_index, axis=1),
          mask=None,
          bias=bias,
          enable_dropout=False,
          decode=True,
          prefill=False,
          mutable=['cache'],
      )

      np.testing.assert_allclose(
          decode_logits, final_prefilled_logits, atol=1e-6
      )

  @parameterized.named_parameters([
      dict(
          testcase_name='multi_head',
          attn_class=dense_attention.MultiHeadDotProductAttention,
      ),
      dict(
          testcase_name='multi_query',
          attn_class=dense_attention.MultiQueryDotProductAttention,
      ),
  ])
  def test_attention_causal_prefill_and_decode_match_decode(self, attn_class):
    """Make sure causal prefill->decode is the same as just decode."""
    with jax.default_matmul_precision('float32'):
      # b: batch, k: kv_len, h: num_head, d: head_dim t: sequence length
      b, h, d, t = 2, 3, 5, 8
      ls = np.array([6, 4]).astype(np.int32)
      f = h * d

      base_args = SelfAttentionArgs(
          num_heads=h,
          qkv_features=f,
          out_features=f,
          dropout_rate=0,
          float32_logits=True,
      )
      args = base_args.init_args()

      inputs_q = np.random.randn(b, t, f).astype(np.float32)
      inputs_kv = np.random.randn(b, t, f).astype(np.float32)
      bias = np.random.randn(1, h, t, t).astype(np.float32)
      mask = dense_attention.make_decoder_mask(
          (np.arange(t) < np.reshape(ls, (-1, 1))).astype(inputs_q.dtype),
          dtype=inputs_q.dtype,
      ).astype(np.float32)

      attn = attn_class(**args, precision='float32')
      params = attn.init(
          jax.random.PRNGKey(0), inputs_q, inputs_kv, mask, bias
      )['params']

      # Pure Decoding
      # Initialize the cache
      _, variables_with_cache = attn.apply(
          {'params': params},
          inputs_q,
          inputs_kv,
          decode=True,
          prefill=False,
          mutable=['cache'],
      )
      decoded_cache = variables_with_cache['cache']

      # Run decoding for each input element.
      decoded_logits = []
      for i in range(t):
        logits, vars_with_new_cache = attn.apply(
            {'params': params, 'cache': decoded_cache},
            inputs_q[:, i, np.newaxis],
            inputs_kv[:, i, np.newaxis],
            mask=None,
            bias=bias,
            enable_dropout=False,
            decode=True,
            prefill=False,
            mutable=['cache'],
        )
        decoded_logits.append(logits)
        decoded_cache = vars_with_new_cache['cache']
      decoded_logits = jnp.concatenate(decoded_logits, axis=1)

      # Prefilled Cache
      # Initialize the cache
      _, variables_with_cache = attn.apply(
          {'params': params},
          inputs_q,
          inputs_kv,
          mask=mask,
          bias=bias,
          decode=True,
          prefill=False,
          mutable=['cache'],
      )
      prefilled_cache = variables_with_cache['cache']
      # Prefill the cache with values calculated via causal attention.
      prefilled_logits, vars_with_new_cache = attn.apply(
          {'params': params, 'cache': prefilled_cache},
          inputs_q,
          inputs_kv,
          mask=mask,
          bias=bias,
          enable_dropout=False,
          decode=False,
          prefill=True,
          prefill_lengths=ls,
          mutable=['cache'],
      )
      prefilled_cache = vars_with_new_cache['cache']

      # Run decoding, starting from where we finished prefilling.
      prefilled_decode_logits = []
      # The prefill step has two different lengths, so for the shorter one to
      # reach the max number of steps we need the longer one to do some extra
      # work which will be discarded. Here we pad out the input so that while we
      # are running real decode steps on the shorter sequence, the longer one
      # will have values to consume.
      decode_steps = t - np.min(ls)
      padding = decode_steps + np.max(ls) - t
      padding = np.zeros((b, padding, f), dtype=inputs_q.dtype)
      padded_inputs_q = np.concatenate([inputs_q, padding], axis=1)
      padded_inputs_kv = np.concatenate([inputs_kv, padding], axis=1)
      # Run decoding steps.
      for i in range(decode_steps):
        idx = np.reshape(ls + i, (-1, 1, 1))
        logits, vars_with_new_cache = attn.apply(
            {'params': params, 'cache': prefilled_cache},
            # Select the next element based on our cache index + the number of
            # decode steps taken.
            np.take_along_axis(padded_inputs_q, idx, axis=1),
            np.take_along_axis(padded_inputs_kv, idx, axis=1),
            mask=None,
            bias=bias,
            enable_dropout=False,
            decode=True,
            prefill=False,
            mutable=['cache'],
        )
        prefilled_cache = vars_with_new_cache['cache']
        prefilled_decode_logits.append(logits)
      prefilled_decode_logits = np.concatenate(prefilled_decode_logits, axis=1)
      prefilled_logits = np.array(prefilled_logits)
      # Copy the decode step logits into the original logits array, while
      # making sure to discard any of the busy work steps.
      for i, l in enumerate(ls):
        prefilled_logits[i, l:] = prefilled_decode_logits[i, : t - l]
        prefilled_logits[i, l:] = prefilled_decode_logits[i, : t - l]

      # `DenseGeneral`, used in the attention class to project q, k, and v, can
      # have some comparatively large difference when running on a slice with
      # a sequence length of 1 vs a the full q, k, or v. As such, our
      # comparisons need to have larger tolerances than normal.
      # Check caches match
      np.testing.assert_allclose(
          prefilled_cache['cached_key'],
          decoded_cache['cached_key'],
          atol=1e-6,
          err_msg='cached keys do not match',
      )
      np.testing.assert_allclose(
          prefilled_cache['cached_value'],
          decoded_cache['cached_value'],
          atol=1e-6,
          err_msg='cached values do not match',
      )
      # Check outputs match
      np.testing.assert_allclose(
          prefilled_logits,
          decoded_logits,
          atol=1e-6,
          err_msg='logits do not match',
      )

  def test_multiquery_dot_product_attention_prefill_caching(self):
    # b: batch, f: qkv_features, k: kv_len, h: num_head, d: head_dim
    b, h, d, k = 2, 3, 4, 5
    f = h * d
    prefill_lengths = np.array([3, 1])

    base_args = SelfAttentionArgs(
        num_heads=h, qkv_features=f, out_features=f, dropout_rate=0
    )
    args = base_args.init_args()

    cache = {
        'cached_key': np.zeros((b, d, k)),
        'cached_value': np.zeros((b, d, k)),
        'cache_index': np.array([0, 0]),
    }
    inputs_q = np.random.randn(b, k, f)
    inputs_kv = np.random.randn(b, k, f)

    def mock_dense_general(self, x, **kwargs):  # pylint: disable=unused-argument
      # For q, replace the projection with simple reshaping.
      if x is inputs_q:
        return x.reshape(b, -1, h, d)
      # For k and v, the feature dim is sliced to mimic down-projection.
      elif x is inputs_kv:
        return x[:, :, :d]

    with mock.patch.object(
        dense.DenseGeneral, '__call__', new=mock_dense_general
    ):
      _, mutated = dense_attention.MultiQueryDotProductAttention(**args).apply(
          {'cache': freeze(cache)},
          inputs_q,
          inputs_kv,
          decode=False,
          prefill=True,
          prefill_lengths=prefill_lengths,
          mutable=['cache'],
      )
      updated_cache = mutated['cache']

    # Perform the same mocked projection to generate the expected cache.
    # (key|value): [b, 1, h, d]
    key = mock_dense_general(None, inputs_kv)
    value = mock_dense_general(None, inputs_kv)

    # cached_(key|value): [b, h, d, k]
    # Update the our gold cache with the key and values that are part of the
    # prefix that we are prefilling the cache with. Explicit loops here avoid a
    # confusing transpose.
    for b, prefill_length in enumerate(prefill_lengths):
      for i in range(prefill_length):
        cache['cached_key'][b, :, i] = key[b, i, :]
        cache['cached_value'][b, :, i] = value[b, i, :]
      cache['cache_index'][b] = prefill_length
    for name, array in cache.items():
      np.testing.assert_allclose(array, updated_cache[name])

  def test_multiquery_dot_product_attention_caching(self):
    # b: batch, f: qkv_features, k: kv_len, h: num_head, d: head_dim
    b, h, d, k = 2, 3, 4, 5
    f = h * d

    base_args = SelfAttentionArgs(
        num_heads=h, qkv_features=f, out_features=f, dropout_rate=0
    )
    args = base_args.init_args()

    cache = {
        'cached_key': np.zeros((b, d, k)),
        'cached_value': np.zeros((b, d, k)),
        'cache_index': np.array(0),
    }
    inputs_q = np.random.randn(b, 1, f)
    inputs_kv = np.random.randn(b, 1, f)

    def mock_dense_general(self, x, **kwargs):  # pylint: disable=unused-argument
      # For q, replace the projection with simple reshaping.
      if x is inputs_q:
        return x.reshape(b, -1, h, d)
      # For k and v, the feature dim is sliced to mimic down-projection.
      elif x is inputs_kv:
        return x[:, :, :d]

    with mock.patch.object(
        dense.DenseGeneral, '__call__', new=mock_dense_general
    ):
      _, mutated = dense_attention.MultiQueryDotProductAttention(**args).apply(
          {'cache': freeze(cache)},
          inputs_q,
          inputs_kv,
          decode=True,
          mutable=['cache'],
      )
      updated_cache = mutated['cache']

    # Perform the same mocked projection to generate the expected cache.
    # (key|value): [b, 1, d]
    key = mock_dense_general(None, inputs_kv)
    value = mock_dense_general(None, inputs_kv)

    # cached_(key|value): [b, d, k]
    cache['cached_key'][:, :, 0] = key[:, 0, :]
    cache['cached_value'][:, :, 0] = value[:, 0, :]
    cache['cache_index'] = np.array(1)
    for name, array in cache.items():
      np.testing.assert_allclose(array, updated_cache[name])

  def test_dot_product_attention_multiquery(self):
    # b: batch, f: qkv_features, q: q_len, k: kv_len, h: num_head, d: head_dim
    b, q, h, d, k = 2, 3, 4, 5, 6
    np.random.seed(0)
    query = np.random.randn(b, q, h, d)
    key = np.random.randn(b, k, d)
    value = np.random.randn(b, k, d)
    bias = np.random.randn(b, h, q, k)
    attn_out = dense_attention.dot_product_attention_multiquery(
        query, key, value, bias=bias
    )
    logits = np.einsum('bqhd,bkd->bhqk', query, key)
    weights = jax.nn.softmax(logits + bias, axis=-1)
    expected_attn_out = np.einsum('bhqk,bkd->bqhd', weights, value)
    np.testing.assert_allclose(attn_out, expected_attn_out, atol=1e-6)

  @parameterized.parameters({'f': 20}, {'f': 22})
  def test_multihead_dot_product_attention_split_head(self, f):
    # b: batch, f: qkv_features, q: q_len, k: kv_len, h: num_head, d: head_dim
    b, q, h, d, k = 2, 3, 4, 5, 6

    base_args = SelfAttentionArgs(
        num_heads=h,
        qkv_features=f,
        out_features=f,
        dropout_rate=0,
        rescale_logits=False,
        use_bias=False,
    )
    args = base_args.init_args()

    if f != h * d:
      args['head_dim'] = d

    np.random.seed(0)
    inputs_q = np.random.randn(b, q, f)
    inputs_kv = np.random.randn(b, k, f)

    # Projection: [b, q, f] -> [b, q, h, d]
    # So the kernels have to be [f, h, d]
    query_kernel = np.random.randn(f, h, d)
    key_kernel = np.random.randn(f, h, d)
    value_kernel = np.random.randn(f, h, d)
    # `out` calculation: [b, q, h, d] -> [b, q, f]
    # So kernel has to be [h, d, f]
    out_kernel = np.random.randn(h, d, f)

    params = {
        'query': {'kernel': query_kernel.reshape(f, -1)},
        'key': {'kernel': key_kernel.reshape(f, -1)},
        'value': {'kernel': value_kernel.reshape(f, -1)},
        'out': {'kernel': out_kernel.reshape(-1, f)},
    }
    y = dense_attention.MultiHeadDotProductAttention(**args).apply(
        {'params': freeze(params)}, inputs_q, inputs_kv
    )

    params = {
        'query': {'kernel': query_kernel},
        'key': {'kernel': key_kernel},
        'value': {'kernel': value_kernel},
        'out': {'kernel': out_kernel},
    }
    args_split_head_kernel = dict(args)
    args_split_head_kernel['split_head_kernel'] = True
    y_split_head_kernel = dense_attention.MultiHeadDotProductAttention(
        **args_split_head_kernel
    ).apply({'params': freeze(params)}, inputs_q, inputs_kv)
    np.testing.assert_allclose(y, y_split_head_kernel, rtol=1e-5, atol=1e-5)

  @parameterized.parameters({'f': 20}, {'f': 22})
  def test_multihead_dot_product_attention_fuse_kernels_kv(self, f):
    # b: batch, f: qkv_features, q: q_len, k: kv_len, h: num_head, d: head_dim
    b, q, h, d, k = 2, 3, 4, 5, 6

    base_args = SelfAttentionArgs(
        num_heads=h,
        qkv_features=f,
        out_features=f,
        dropout_rate=0,
        rescale_logits=False,
        use_bias=False,
    )
    args = base_args.init_args()
    args['split_head_kernel'] = True
    args['rescale_logits'] = True

    if f != h * d:
      args['head_dim'] = d

    np.random.seed(0)
    inputs_q = np.random.randn(b, q, f)
    inputs_kv = np.random.randn(b, k, f)

    # Projection: [b, q, f] -> [b, q, h, d]
    # So the kernels have to be [f, h, d]
    query_kernel = np.random.randn(f, h, d)
    key_kernel = np.random.randn(f, h, d)
    value_kernel = np.random.randn(f, h, d)
    # `out` calculation: [b, q, h, d] -> [b, q, f]
    # So kernel has to be [h, d, f]
    out_kernel = np.random.randn(h, d, f)

    params = {
        'query': {'kernel': query_kernel},
        'key': {'kernel': key_kernel},
        'value': {'kernel': value_kernel},
        'out': {'kernel': out_kernel},
    }
    y = dense_attention.MultiHeadDotProductAttention(**args).apply(
        {'params': freeze(params)}, inputs_q, inputs_kv
    )

    fused_kernel = np.stack([key_kernel, value_kernel], axis=1)
    params = {
        'query': {'kernel': query_kernel},
        'kv_fused': {'kernel': fused_kernel},
        'out': {'kernel': out_kernel},
    }
    args_fused_kernels = dict(args)
    args_fused_kernels['kernels_to_fuse'] = 'kv'
    y_fused_kernels = dense_attention.MultiHeadDotProductAttention(
        **args_fused_kernels
    ).apply({'params': freeze(params)}, inputs_q, inputs_kv)
    np.testing.assert_allclose(y, y_fused_kernels, rtol=1e-5, atol=1e-5)

  @parameterized.parameters({'f': 20}, {'f': 22})
  def test_multihead_dot_product_attention_fuse_kernels_qkv(self, f):
    # b: batch, f: qkv_features, q: q_len, k: kv_len, h: num_head, d: head_dim
    b, q, h, d = 2, 3, 4, 5

    base_args = SelfAttentionArgs(
        num_heads=h,
        qkv_features=f,
        out_features=f,
        dropout_rate=0,
        rescale_logits=False,
        use_bias=False,
    )
    args = base_args.init_args()
    args['split_head_kernel'] = True
    args['rescale_logits'] = True

    if f != h * d:
      args['head_dim'] = d

    np.random.seed(0)
    inputs_q = np.random.randn(b, q, f)

    # Projection: [b, q, f] -> [b, q, h, d]
    # So the kernels have to be [f, h, d]
    query_kernel = np.random.randn(f, h, d)
    key_kernel = np.random.randn(f, h, d)
    value_kernel = np.random.randn(f, h, d)
    # `out` calculation: [b, q, h, d] -> [b, q, f]
    # So kernel has to be [h, d, f]
    out_kernel = np.random.randn(h, d, f)

    params = {
        'query': {'kernel': query_kernel},
        'key': {'kernel': key_kernel},
        'value': {'kernel': value_kernel},
        'out': {'kernel': out_kernel},
    }
    y = dense_attention.MultiHeadDotProductAttention(**args).apply(
        {'params': freeze(params)}, inputs_q, inputs_q
    )

    fused_kernel = np.stack([query_kernel, key_kernel, value_kernel], axis=1)
    params = {
        'qkv_fused': {'kernel': fused_kernel},
        'out': {'kernel': out_kernel},
    }
    args_fused_kernels = dict(args)
    args_fused_kernels['kernels_to_fuse'] = 'qkv'
    y_fused_kernels = dense_attention.MultiHeadDotProductAttention(
        **args_fused_kernels
    ).apply({'params': freeze(params)}, inputs_q, inputs_q)
    np.testing.assert_allclose(y, y_fused_kernels, rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters([
      ('no_fuse_kernel_none', None, False, False, False),
      ('no_fuse_kernel_qkv', None, True, False, False),
      ('no_fuse_kernel_qkv_kv', None, True, True, False),
      ('no_fuse_kernel_qkv_kv_q', None, True, True, True),
      ('qkv_fuse_kernel_none', 'qkv', False, False, False),
      ('qkv_fuse_kernel_qkv', 'qkv', True, False, False),
      ('qkv_fuse_kernel_qkv_kv', 'qkv', True, True, False),
      ('qkv_fuse_kernel_qkv_kv_q', 'qkv', True, True, True),
      ('kv_fuse_kernel_none', 'kv', False, False, False),
      ('kv_fuse_kernel_qkv', 'kv', True, False, False),
      ('kv_fuse_kernel_qkv_kv', 'kv', True, True, False),
      ('kv_fuse_kernel_qkv_kv_q', 'kv', True, True, True),
  ])
  def test_multihead_dot_product_attention_kernel_kernel_init(
      self,
      fused_kernels,
      set_qkv_kernel_init,
      set_kv_kernel_init,
      set_q_kernel_init,
  ):
    # b: batch, f: qkv_features, q: q_len, k: kv_len, h: num_head, d: head_dim
    f = 20
    b, q, h, d = 2, 3, 4, 5

    base_args = SelfAttentionArgs(
        num_heads=h,
        qkv_features=f,
        out_features=f,
        dropout_rate=0,
        rescale_logits=False,
        use_bias=False,
    )
    args = base_args.init_args()
    args['split_head_kernel'] = True
    args['rescale_logits'] = True
    args['kernel_init'] = functools.partial(self._mock_initializer, val=1.0)
    if fused_kernels:
      args['kernels_to_fuse'] = fused_kernels
    if set_qkv_kernel_init:
      args['qkv_kernel_init'] = functools.partial(
          self._mock_initializer, val=2.0
      )
    if set_kv_kernel_init:
      args['kv_kernel_init'] = functools.partial(
          self._mock_initializer, val=3.0
      )
    if set_q_kernel_init:
      args['q_kernel_init'] = functools.partial(self._mock_initializer, val=4.0)

    if f != h * d:
      args['head_dim'] = d

    np.random.seed(0)
    inputs_q = np.random.randn(b, q, f)

    params = dense_attention.MultiHeadDotProductAttention(**args).init(
        random.PRNGKey(0), inputs_q, inputs_q, enable_dropout=False
    )

    # Construct expected param
    # Projection: [b, q, f] -> [b, q, h, d]
    # So the kernels have to be [f, h, d]
    query_kernel = np.ones((f, h, d))
    key_kernel = np.ones((f, h, d))
    value_kernel = np.ones((f, h, d))
    # `out` calculation: [b, q, h, d] -> [b, q, f]
    # So kernel has to be [h, d, f]
    out_kernel = np.ones((h, d, f))
    if fused_kernels is None:
      if set_q_kernel_init:
        query_kernel = np.ones((f, h, d)) * 4.0

      expected_params = {
          'query': {'kernel': query_kernel.tolist()},
          'key': {'kernel': key_kernel.tolist()},
          'value': {'kernel': value_kernel.tolist()},
          'out': {'kernel': out_kernel.tolist()},
      }
    elif fused_kernels == 'qkv':
      if set_qkv_kernel_init:
        query_kernel = np.ones((f, h, d)) * 2.0
        key_kernel = np.ones((f, h, d)) * 2.0
        value_kernel = np.ones((f, h, d)) * 2.0

      fused_kernel = np.stack([query_kernel, key_kernel, value_kernel], axis=1)
      expected_params = {
          'qkv_fused': {'kernel': fused_kernel.tolist()},
          'out': {'kernel': out_kernel.tolist()},
      }
    elif fused_kernels == 'kv':
      if set_kv_kernel_init:
        key_kernel = np.ones((f, h, d)) * 3.0
        value_kernel = np.ones((f, h, d)) * 3.0
      if set_q_kernel_init:
        query_kernel = np.ones((f, h, d)) * 4.0

      kv_fused_kernel = np.stack([key_kernel, value_kernel], axis=1)
      expected_params = {
          'kv_fused': {'kernel': kv_fused_kernel.tolist()},
          'query': {'kernel': query_kernel.tolist()},
          'out': {'kernel': out_kernel.tolist()},
      }

    self.assertDictEqual(
        jax.tree.map(lambda a: a.tolist(), unfreeze(params['params'])),
        expected_params,
    )

  def test_decoder_logits_mask_unpacked(self):
    # [batch, length]
    decoder_input_tokens = jnp.array(
        [[0, 3, 9, 4, 1, 0, 0], [0, 8, 5, 3, 1, 0, 0]]
    )
    expected = jnp.array(
        [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0]], dtype=jnp.float32
    )
    # [batch, length, 1]
    expected = jnp.expand_dims(expected, axis=-1)
    logit_mask = dense_attention.get_decoder_logit_mask(
        decoder_input_tokens, jnp.float32
    )
    self.assertEqual(logit_mask.dtype, jnp.float32)
    np.testing.assert_array_equal(logit_mask, expected)

  def test_decoder_logits_mask_packed(self):
    # Two sequences packed together for each batch elements.
    # [batch, length]
    decoder_input_tokens = jnp.array(
        [[0, 3, 9, 0, 4, 8, 0, 0], [0, 8, 5, 8, 0, 9, 0, 0]]
    )
    expected = jnp.array(
        [[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0]], dtype=jnp.float32
    )
    # [batch, length, 1]
    expected = jnp.expand_dims(expected, axis=-1)
    logit_mask = dense_attention.get_decoder_logit_mask(
        decoder_input_tokens, jnp.float32
    )
    self.assertEqual(logit_mask.dtype, jnp.float32)
    np.testing.assert_array_equal(logit_mask, expected)


class LocalAttentionLayerTest(parameterized.TestCase):

  @parameterized.parameters(
      itertools.product(
          [True, False],
          [True, False],
          [True, False],
      )
  )
  def test_shapes(
      self,
      always_attend_to_first_position,
      first_position_attends_to_all,
      output_projection,
  ):
    """Checks the local attention layer's shapes are correct."""
    num_heads = 2
    head_dim = 5
    out_features = 11
    model = dense_attention.LocalAttentionLayer(
        dense_attention.MultiHeadDotProductAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            use_bias=True,
            dropout_rate=0.0,
            output_projection=output_projection,
            out_features=out_features if output_projection else None,
        ),
        q_chunk_width=4,
        q_chunk_stride=4,
        kv_chunk_width=6,
        kv_chunk_stride=6,
        always_attend_to_first_position=always_attend_to_first_position,
        first_position_attends_to_all=first_position_attends_to_all,
    )

    batch_size = 3
    q_len = 8
    q_features = 7
    kv_len = 12
    kv_features = 9
    inputs_q = np.ones([batch_size, q_len, q_features], dtype=np.float32)
    inputs_kv = np.ones([batch_size, kv_len, kv_features], dtype=np.float32)
    mask = np.ones([batch_size, 1, q_len, kv_len], dtype=np.int32)
    bias = np.ones([batch_size, 1, q_len, kv_len], dtype=np.int32)
    key = random.PRNGKey(0)

    outputs, _ = model.init_with_output(key, inputs_q, inputs_kv, mask, bias)
    if output_projection:
      self.assertSequenceEqual(outputs.shape, (batch_size, q_len, out_features))
    else:
      self.assertSequenceEqual(
          outputs.shape, (batch_size, q_len, num_heads, head_dim)
      )


class QuantizedAttentionTest(parameterized.TestCase):

  def test_quantization_no_params_specified(self):
    module = dense_attention.MultiQueryDotProductAttention(
        num_heads=2,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        dtype=jnp.float32,
        use_bias=True,
        use_aqt=True,
    )
    inputs_q = np.array(
        [
            # Batch 1.
            [[1, 1], [1, 1], [1, 2]],
            # Batch 2.
            [[2, 2], [3, 1], [2, 2]],
        ],
        dtype=np.float32,
    )
    inputs_kv = np.array(
        [
            # Batch 1.
            [[1, 1], [1, 1], [1, 2]],
            # Batch 2.
            [[2, 2], [3, 1], [2, 2]],
        ],
        dtype=np.float32,
    )
    with self.assertRaisesRegex(
        ValueError, 'If use_aqt is True, either of weights or acts quantization'
    ):
      module.init(random.PRNGKey(0), inputs_q, inputs_kv, enable_dropout=False)

  def test_multiquery_dot_product_attention_quantized_weights(self):
    module = dense_attention.MultiQueryDotProductAttention(
        num_heads=2,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        dtype=jnp.float32,
        use_bias=True,
        use_aqt=True,
        weight_params=aqt.QuantOps.WeightParams(
            prec=8, half_shift=False, axis=None
        ),
    )

    inputs_q = np.array(
        [
            # Batch 1.
            [[1, 1], [1, 1], [1, 2]],
            # Batch 2.
            [[2, 2], [3, 1], [2, 2]],
        ],
        dtype=np.float32,
    )
    inputs_kv = np.array(
        [
            # Batch 1.
            [[1, 1], [1, 1], [1, 2]],
            # Batch 2.
            [[2, 2], [3, 1], [2, 2]],
        ],
        dtype=np.float32,
    )

    expected_params = {
        'params': {
            'query': {
                'kernel': jnp.array(
                    [
                        [[0.89760804], [-0.7743368]],
                        [[-0.27043915], [-0.09338999]],
                    ],
                    dtype=jnp.float32,
                ),
                'bias': jnp.array(
                    [3.8685133e-07, -5.7897455e-07], dtype=jnp.float32
                ),
            },
            'key': {
                'kernel': jnp.array(
                    [[-1.2404252], [0.6276205]], dtype=jnp.float32
                ),
                'bias': jnp.array([9.180263e-07], dtype=jnp.float32),
            },
            'value': {
                'kernel': jnp.array(
                    [[-0.8634736], [-0.9621272]], dtype=jnp.float32
                ),
                'bias': jnp.array([8.859404e-07], dtype=jnp.float32),
            },
            'out': {
                'kernel': jnp.array(
                    [[0.8359484, 0.9604499], [-1.0830641, 1.0543139]],
                    dtype=jnp.float32,
                ),
                'bias': jnp.array(
                    [-9.7886084e-07, 1.3396599e-06], dtype=jnp.float32
                ),
            },
        },
        'params_axes': {
            'query': {
                'kernel_axes': AxisMetadata(names=('embed', 'heads', 'kv')),
                'bias_axes': AxisMetadata(names=('kv',)),
            },
            'key': {
                'kernel_axes': AxisMetadata(names=('embed', 'kv')),
                'bias_axes': AxisMetadata(names=('kv',)),
            },
            'value': {
                'kernel_axes': AxisMetadata(names=('embed', 'kv')),
                'bias_axes': AxisMetadata(names=('kv',)),
            },
            'out': {
                'kernel_axes': AxisMetadata(names=('joined_kv', 'embed')),
                'bias_axes': AxisMetadata(names=('embed',)),
            },
        },
    }
    result, params = module.init_with_output(
        random.PRNGKey(0), inputs_q, inputs_kv, enable_dropout=False
    )
    jax.tree.map(
        functools.partial(np.testing.assert_allclose, rtol=1e-6),
        unfreeze(params),
        expected_params,
    )

    np.testing.assert_allclose(
        result.tolist(),
        [
            [
                [0.3442336916923523, -4.3061041831970215],
                [0.3442336916923523, -4.3061041831970215],
                [0.36651411652565, -4.258667469024658],
            ],
            [
                [0.807983934879303, -7.265725612640381],
                [0.799161970615387, -7.264179706573486],
                [0.807983934879303, -7.265725612640381],
            ],
        ],
        rtol=1e-6,
    )

  def test_multiquery_dot_product_attention_materialized_weights(self):
    weight_params = aqt.QuantOps.WeightParams(
        prec=8, half_shift=False, axis=None
    )
    module = dense_attention.MultiQueryDotProductAttention(
        num_heads=2,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        dtype=jnp.float32,
        use_bias=True,
        use_aqt=True,
        weight_params=weight_params,
        possibly_use_quantized_vars=True,
    )

    # enable_dropout

    inputs_q = np.array(
        [
            # Batch 1.
            [[1, 1], [1, 1], [1, 2]],
            # Batch 2.
            [[2, 2], [3, 1], [2, 2]],
        ],
        dtype=np.float32,
    )
    inputs_kv = np.array(
        [
            # Batch 1.
            [[1, 1], [1, 1], [1, 2]],
            # Batch 2.
            [[2, 2], [3, 1], [2, 2]],
        ],
        dtype=np.float32,
    )

    result, params = module.init_with_output(
        random.PRNGKey(0), inputs_q, inputs_kv, enable_dropout=False
    )

    expected_params = {
        'params': {
            'query': {
                'qkernel': jnp.array([[[0], [0]], [[0], [0]]], dtype=jnp.int8),
                'qscale': jnp.array(
                    [[[3.8685133e-07], [-5.7897455e-07]]], dtype=jnp.float32
                ),
                'bias': jnp.array(
                    [1.1104368e-06, 2.4920448e-06], dtype=jnp.float32
                ),
            },
            'key': {
                'qkernel': jnp.array([[0], [0]], dtype=jnp.int8),
                'qscale': jnp.array([[9.180263e-07]], dtype=jnp.float32),
                'bias': jnp.array([5.054643e-07], dtype=jnp.float32),
            },
            'value': {
                'qkernel': jnp.array([[0], [0]], dtype=jnp.int8),
                'qscale': jnp.array([[8.859404e-07]], dtype=jnp.float32),
                'bias': jnp.array([4.5408714e-07], dtype=jnp.float32),
            },
            'out': {
                'qkernel': jnp.array([[0, 0], [0, 0]], dtype=jnp.int8),
                'qscale': jnp.array(
                    [[-9.7886084e-07, 1.3396599e-06]], dtype=jnp.float32
                ),
                'bias': jnp.array(
                    [-3.5336794e-07, -3.4736888e-07], dtype=jnp.float32
                ),
            },
        },
        'params_axes': {
            'query': {
                'qkernel_axes': AxisMetadata(names=('embed', 'heads', 'kv')),
                'qscale_axes': AxisMetadata(
                    names=('embed_qscale', 'heads', 'kv')
                ),
                'bias_axes': AxisMetadata(names=('kv',)),
            },
            'key': {
                'qkernel_axes': AxisMetadata(names=('embed', 'kv')),
                'qscale_axes': AxisMetadata(names=('embed_qscale', 'kv')),
                'bias_axes': AxisMetadata(names=('kv',)),
            },
            'value': {
                'qkernel_axes': AxisMetadata(names=('embed', 'kv')),
                'qscale_axes': AxisMetadata(names=('embed_qscale', 'kv')),
                'bias_axes': AxisMetadata(names=('kv',)),
            },
            'out': {
                'qkernel_axes': AxisMetadata(names=('joined_kv', 'embed')),
                'qscale_axes': AxisMetadata(
                    names=('joined_kv_qscale', 'embed')
                ),
                'bias_axes': AxisMetadata(names=('embed',)),
            },
        },
    }
    jax.tree.map(
        functools.partial(np.testing.assert_allclose, rtol=1e-6),
        unfreeze(params),
        expected_params,
    )
    self.assertDictEqual(
        testing_utils.param_dtypes_shapes_axes(
            params['params'], params['params_axes']
        ),
        {
            'key': {
                'bias': ['float32', 'kv=1'],
                'qkernel': ['int8', 'embed=2', 'kv=1'],
                'qscale': ['float32', 'embed_qscale=1', 'kv=1'],
            },
            'out': {
                'bias': ['float32', 'embed=2'],
                'qkernel': ['int8', 'joined_kv=2', 'embed=2'],
                'qscale': ['float32', 'joined_kv_qscale=1', 'embed=2'],
            },
            'query': {
                'bias': ['float32', 'kv=2'],
                'qkernel': ['int8', 'embed=2', 'heads=2', 'kv=1'],
                'qscale': ['float32', 'embed_qscale=1', 'heads=2', 'kv=1'],
            },
            'value': {
                'bias': ['float32', 'kv=1'],
                'qkernel': ['int8', 'embed=2', 'kv=1'],
                'qscale': ['float32', 'embed_qscale=1', 'kv=1'],
            },
        },
    )
    np.testing.assert_allclose(
        result.tolist(),
        [
            [
                [-3.5336794e-07, -3.4736888e-07],
                [-3.5336794e-07, -3.4736888e-07],
                [-3.5336794e-07, -3.4736888e-07],
            ],
            [
                [-3.5336794e-07, -3.4736888e-07],
                [-3.5336794e-07, -3.4736888e-07],
                [-3.5336794e-07, -3.4736888e-07],
            ],
        ],
        rtol=1e-6,
    )


if __name__ == '__main__':
  absltest.main()
