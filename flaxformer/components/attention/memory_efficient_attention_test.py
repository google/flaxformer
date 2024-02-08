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
from typing import Any, Callable, Optional

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from flax.core import freeze
import jax
from jax import random
import jax.numpy as jnp
import numpy as np

from flaxformer.components import dense
from flaxformer.components.attention import dense_attention
from flaxformer.components.attention import memory_efficient_attention
from flaxformer.types import Array

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class MultiQueryDotProductAttention(
    dense_attention.MultiQueryDotProductAttention
):
  """Memory-efficient multi-query dot-product attention."""

  attention_fn: Callable[[Array, Array, Array], Array] = (
      memory_efficient_attention.dot_product_attention_multiquery
  )


@dataclasses.dataclass(frozen=True)
class SelfAttentionArgs:
  num_heads: int = 1
  batch_size: int = 2
  qkv_features: int = 8
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

  def apply_args(self, dtype=jnp.float32):
    inputs_q = jnp.ones(
        (self.batch_size, self.q_len, self.features), dtype=dtype
    )
    mask = jnp.ones(
        (self.batch_size, self.num_heads, self.q_len, self.q_len), dtype=dtype
    )
    bias = jnp.ones(
        (self.batch_size, self.num_heads, self.q_len, self.q_len), dtype=dtype
    )
    return {
        'inputs_q': inputs_q,
        'mask': mask,
        'bias': bias,
        'enable_dropout': self.enable_dropout,
    }


class AttentionTest(parameterized.TestCase):

  def test_memory_efficient_attention_shape(self):
    # This test only checks for shape but tries to make sure all code paths are
    # reached.
    dropout_rng = random.PRNGKey(0)
    batch_size, num_heads, q_len, kv_len, qk_depth, v_depth = 1, 2, 3, 4, 5, 6

    query = jnp.ones((batch_size, q_len, num_heads, qk_depth))
    key = jnp.ones((batch_size, kv_len, qk_depth))
    value = jnp.ones((batch_size, kv_len, v_depth))
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

    output = memory_efficient_attention.dot_product_attention_multiquery(
        **args, broadcast_dropout=True
    )
    self.assertEqual(output.shape, (batch_size, q_len, num_heads, v_depth))

    # Make sure we also reach the code path where we don't broadcast dropout.
    output = memory_efficient_attention.dot_product_attention_multiquery(
        **args, broadcast_dropout=False
    )
    self.assertEqual(output.shape, (batch_size, q_len, num_heads, v_depth))

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
    y = MultiQueryDotProductAttention(**args).apply(
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
      _, mutated = MultiQueryDotProductAttention(**args).apply(
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

  @parameterized.parameters(
      {'bias_gen': lambda *args: np.zeros(args)}, {'bias_gen': np.random.randn}
  )
  def test_dot_product_attention_multiquery(self, bias_gen: ...):
    # b: batch, f: qkv_features, q: q_len, k: kv_len, h: num_head, d: head_dim
    b, q, h, d, k = 2, 3, 4, 5, 6
    np.random.seed(0)
    query = np.random.randn(b, q, h, d)
    key = np.random.randn(b, k, d)
    value = np.random.randn(b, k, d)
    bias = bias_gen(b, h, q, k)
    attn_out = memory_efficient_attention.dot_product_attention_multiquery(
        query, key, value, bias=bias
    )
    logits = np.einsum('bqhd,bkd->bhqk', query, key)
    weights = jax.nn.softmax(logits + bias, axis=-1)
    expected_attn_out = np.einsum('bhqk,bkd->bqhd', weights, value)
    np.testing.assert_allclose(attn_out, expected_attn_out, atol=1e-6)

  @parameterized.parameters(
      {},  # defaults
      {'use_extra_logit': True},
      {
          'key_chunk_size': 1,
      },
      {
          'query_chunk_size': 1,
      },
      {'key_chunk_size': 2, 'k': 12},
      {'key_chunk_size': 2, 'k': 12, 'use_extra_logit': True},
      {'query_chunk_size': 2, 'q': 6},
      {'init_fn': lambda *args: np.zeros(args)},
      {'causal_mask': True, 'b': 1, 'k': 2, 'q': 2, 'h': 1, 'd': 2},
      {'causal_mask': True, 'k': 8, 'q': 8},
      # Trigger the code path where some chunks are skipped.
      {
          'causal_mask': True,
          'b': 1,
          'k': 8,
          'q': 8,
          'h': 1,
          'd': 7,
          'key_chunk_size': 2,
          'query_chunk_size': 2,
      },
      {'use_bias': False},
      {'b': 1, 'k': 8192, 'q': 2048, 'h': 1, 'd': 128, 'use_extra_logit': True},
  )
  def test_memory_efficient_same_as_default(
      self,
      use_extra_logit: bool = False,
      b: int = 2,
      q: int = 3,
      h: int = 5,
      d: int = 7,
      k: int = 11,
      key_chunk_size: Optional[int] = None,
      query_chunk_size: Optional[int] = None,
      init_fn: Callable[[...], Any] = np.random.randn,
      use_bias: bool = True,
      causal_mask: bool = False,
  ):
    # b: batch, f: qkv_features, q: q_len, k: kv_len, h: num_head, d: head_dim
    np.random.seed(0)
    query = init_fn(b, q, h, d)
    key = init_fn(b, k, d)
    value = init_fn(b, k, d)
    bias = None
    if use_bias:
      bias = init_fn(b, h, q, k)
    attn_bias = bias
    if causal_mask:
      attn_bias += memory_efficient_attention._causal_bias(q, k, 0)
    attn = dense_attention.dot_product_attention_multiquery(
        query,
        key,
        value,
        bias=attn_bias,
        use_extra_logit=use_extra_logit,
    )
    m_attn_fn = memory_efficient_attention.dot_product_attention_multiquery
    if key_chunk_size is not None:
      m_attn_fn = functools.partial(m_attn_fn, key_chunk_size=key_chunk_size)
    if query_chunk_size is not None:
      m_attn_fn = functools.partial(
          m_attn_fn, query_chunk_size=query_chunk_size
      )
    m_attn = m_attn_fn(
        query,
        key,
        value,
        # We use the bias version WITHOUT the causal mask, to test the
        # causal_mask flag.
        bias=bias,
        use_extra_logit=use_extra_logit,
        causal_mask=causal_mask,
    )
    np.testing.assert_allclose(attn, m_attn, atol=1e-5, rtol=1e-2)

  @parameterized.parameters(
      {'dropout_rate': 0.00001},
      {'dropout_rate': 0.5},
      {'dropout_rate': 1.0},
      {'key_chunk_size': 2, 'k': 4, 'dropout_rate': 0.5},
      {'query_chunk_size': 2, 'q': 4, 'dropout_rate': 0.5},
  )
  def test_dropout(
      self,
      dropout_rate: float,
      b=2,
      q=3,
      h=5,
      d=3,
      k=5,
      key_chunk_size=None,
      query_chunk_size=None,
      causal_mask: bool = False,
      use_bias: bool = True,
  ):
    # smoketest only
    dropout_rng = jax.random.PRNGKey(0)
    np.random.seed(0)
    query = np.random.randn(b, q, h, d)
    key = np.random.randn(b, k, d)
    value = np.random.randn(b, k, d)
    bias = None
    if use_bias:
      bias = np.random.randn(b, h, q, k)
    m_attn_fn = memory_efficient_attention.dot_product_attention_multiquery
    if key_chunk_size is not None:
      m_attn_fn = functools.partial(m_attn_fn, key_chunk_size=key_chunk_size)
    if query_chunk_size is not None:
      m_attn_fn = functools.partial(
          m_attn_fn, query_chunk_size=query_chunk_size
      )
    m_attn_fn = functools.partial(
        m_attn_fn,
        query,
        key,
        value,
        bias=bias,
        causal_mask=causal_mask,
        dropout_rate=dropout_rate,
    )
    m_attn = m_attn_fn(dropout_rng=dropout_rng)
    if dropout_rate > 0.1 and dropout_rate < 0.9:
      alt_dropout_rng = jax.random.PRNGKey(1)
      alt_m_attn = m_attn_fn(dropout_rng=alt_dropout_rng)
      if np.allclose(m_attn, alt_m_attn, atol=1e-6):
        self.fail(
            f'm_attn and alt_m_attn should differ:\n{m_attn=}\n{alt_m_attn=}'
        )


class SelfAttention(dense_attention.MultiHeadDotProductAttention):
  """Self-attention special case of multi-head dot-product attention."""

  attention_fn: Callable[[Array, Array, Array], Array] = (
      memory_efficient_attention.dot_product_attention_multihead
  )

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


class MHAttentionTest(parameterized.TestCase):

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

    output = memory_efficient_attention.dot_product_attention_multihead(
        **args, broadcast_dropout=True
    )
    self.assertEqual(output.shape, (batch_size, q_len, num_heads, v_depth))

    # Make sure we also reach the code path where we don't broadcast dropout.
    output = memory_efficient_attention.dot_product_attention_multihead(
        **args, broadcast_dropout=False
    )
    self.assertEqual(output.shape, (batch_size, q_len, num_heads, v_depth))

  def test_dot_product_attention_no_batch_dim(self):
    num_heads, q_len, kv_len, qk_depth, v_depth = 1, 2, 3, 4, 5
    query = jnp.ones((q_len, num_heads, qk_depth))
    key = jnp.ones((kv_len, num_heads, qk_depth))
    value = jnp.ones((kv_len, num_heads, v_depth))
    output = memory_efficient_attention.dot_product_attention_multihead(
        query, key, value
    )
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

  @parameterized.product(dtype=['float32', 'bfloat16'])
  def test_self_attention_with_decoding(self, dtype):
    rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    args = SelfAttentionArgs(decode=True, q_len=1)
    init_args = args.init_args()
    init_args['dtype'] = dtype
    model = SelfAttention(**init_args)
    apply_args = args.apply_args(dtype=dtype)
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

  def test_dot_product_attention(self):
    # b: batch, f: qkv_features, q: q_len, k: kv_len, h: num_head, d: head_dim
    b, q, h, d, k = 2, 3, 4, 5, 6
    np.random.seed(0)
    query = np.random.randn(b, q, h, d)
    key = np.random.randn(b, k, h, d)
    value = np.random.randn(b, k, h, d)
    bias = np.random.randn(b, h, q, k)
    attn_out = memory_efficient_attention.dot_product_attention_multihead(
        query, key, value, bias=bias
    )
    logits = np.einsum('bqhd,bkhd->bhqk', query, key)
    weights = jax.nn.softmax(logits + bias, axis=-1)
    expected = np.einsum('bhqk,bkhd->bqhd', weights, value)
    np.testing.assert_allclose(attn_out, expected, atol=1e-6)

  def test_with_rope_and_bfloat16(self):
    rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    args = SelfAttentionArgs()
    init_args = args.init_args()
    init_args.update({'dtype': 'bfloat16', 'use_rotary_embedding': True})
    model = SelfAttention(**init_args)
    y, _ = model.init_with_output(rngs, **args.apply_args(dtype='bfloat16'))
    self.assertEqual(y.shape, (args.batch_size, args.q_len, args.out_features))


if __name__ == '__main__':
  absltest.main()
