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

"""Tests for token_hierarchy.py."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

from flaxformer.architectures.h_transformer import token_hierarchy


class OneDimTokenCoarseningTest(parameterized.TestCase):
  """Test cases for OneDimTokenCoarsening."""

  @parameterized.named_parameters(
      ('sample', token_hierarchy.TokenCoarseningMethod.SAMPLE,
       np.array([[[1.], [3.], [5.]]])),
      ('sum', token_hierarchy.TokenCoarseningMethod.SUM,
       np.array([[[3.], [7.], [11.]]])),
      ('const_average', token_hierarchy.TokenCoarseningMethod.CONST_AVERAGE,
       np.array([[[1.5], [3.5], [5.5]]])),
      ('linear_average', token_hierarchy.TokenCoarseningMethod.LINEAR_AVERAGE,
       np.array([[[2.], [4.], [4.25]]])),
  )
  def test_coarsening(self, method, expected_result):
    batch_size = 1
    seq_len = 6
    feature_size = 1
    seq_shape = [batch_size, seq_len, feature_size]
    data_size = np.prod(seq_shape)
    inputs = jnp.arange(1, data_size + 1).reshape(tuple(seq_shape))

    coasening_fn = token_hierarchy.OneDimTokenCoarsening(
        conv_kernel_size=2, method=method)
    result = coasening_fn(inputs)
    logging.info('method = %s', method)
    logging.info('result = %s', result)
    logging.info('expected_result = %s', expected_result)
    np.testing.assert_array_almost_equal(result, expected_result)


class OneDimInterpolationTest(parameterized.TestCase):
  """Test cases for OneDimTokenInterpolation."""

  @parameterized.named_parameters(
      ('const', token_hierarchy.ConvKernelType.CONST, False,
       np.array([[[1.], [1.], [2.], [2.], [3.], [3.]]])),
      ('linear_no_correction', token_hierarchy.ConvKernelType.LINEAR, False,
       np.array([[[1.], [1.5], [2.], [2.5], [3.], [1.5]]])),
      ('linear_with_correction', token_hierarchy.ConvKernelType.LINEAR, True,
       np.array([[[1.], [1.5], [2.], [2.5], [3.], [3.]]])),
  )
  def test_interpolation(self, conv_kernel_type, use_edge_correction,
                         expected_result):
    batch_size = 1
    seq_len = 3
    feature_size = 1
    seq_shape = [batch_size, seq_len, feature_size]
    data_size = np.prod(seq_shape)
    inputs = jnp.arange(1, data_size + 1).reshape(tuple(seq_shape))

    interpolation_fn = token_hierarchy.OneDimTokenInterpolation(
        conv_kernel_size=2,
        conv_kernel_type=conv_kernel_type,
        use_edge_correction=use_edge_correction)
    result = interpolation_fn(inputs)
    logging.info('conv_kernel_type = %s', conv_kernel_type)
    logging.info('result = %s', result)
    logging.info('expected_result = %s', expected_result)
    np.testing.assert_array_almost_equal(result, expected_result)


class OneDimTokenHierarchyTest(parameterized.TestCase):
  """Test cases for OneDimTokenHierarchy."""

  @parameterized.named_parameters(
      ('odd_num_cluster', 3, 4),
      ('wrong_num_block', 3, 7),
  )
  def test_bad_attribute_value(self, num_cluster, num_block):
    seq_len = num_cluster * num_block
    with self.assertRaises(ValueError):
      token_hierarchy.OneDimTokenHierarchy(
          seq_len=seq_len, num_cluster=num_cluster)

  @parameterized.named_parameters(('causal_mask', True, {
      token_hierarchy.TokenBlockName.ANCHOR:
          np.array([[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]],
                     [[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]]]]),
      token_hierarchy.TokenBlockName.LEFT:
          np.array([[[[0., 0.], [0., 0.]], [[5., 6.], [7., 8.]],
                     [[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]],
                     [[0., 0.], [0., 0.]], [[9., 10.], [13., 14.]]]]),
  }), ('non_causal_mask', False, {
      token_hierarchy.TokenBlockName.ANCHOR:
          np.array([[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]],
                     [[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]]]]),
      token_hierarchy.TokenBlockName.LEFT:
          np.array([[[[0., 0.], [0., 0.]], [[5., 6.], [7., 8.]],
                     [[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]],
                     [[0., 0.], [0., 0.]], [[10., 11.], [14., 15.]]]]),
      token_hierarchy.TokenBlockName.RIGHT:
          np.array([[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]],
                     [[9., 10.], [11., 12.]], [[0., 0.], [0., 0.]],
                     [[2., 3.], [6., 7.]], [[0., 0.], [0., 0.]]]]),
  }))
  def test_hierarchical_coarsen_without_padding(self, causal_mask,
                                                expected_coarse_query):
    batch_size = 1
    feature_size = 2
    num_cluster = 2
    num_block = 4
    seq_len = num_block * num_cluster

    seq_shape = [batch_size, seq_len, feature_size]
    inputs_q = jnp.arange(1, 17).reshape(tuple(seq_shape))

    hierarchy = token_hierarchy.OneDimTokenHierarchy(
        seq_len=seq_len,
        num_cluster=num_cluster,
        for_self_attention=True,
        causal_mask=causal_mask)
    results = hierarchy.hierarchical_coarsen(inputs_q, inputs_q)
    coarse_qkv = results.packed_coarse_qkv
    coarse_query, coarse_key, coarse_value = coarse_qkv.values()

    partitioned_shape = tuple(
        (batch_size, num_block, num_cluster, feature_size))
    diag_qkv = inputs_q.reshape(partitioned_shape)
    expected_coarse_key = {
        token_hierarchy.TokenBlockName.ANCHOR:
            diag_qkv,
        token_hierarchy.TokenBlockName.LEFT:
            np.array([[[[0., 0.], [0., 0.]], [[1., 2.], [3., 4.]],
                       [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]],
                       [[0., 0.], [0., 0.]], [[2., 3.], [6., 7.]]]]),
        token_hierarchy.TokenBlockName.RIGHT:
            np.array([[[[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]],
                       [[13., 14.], [15., 16.]], [[0., 0.], [0., 0.]],
                       [[10., 11.], [14., 15.]], [[0., 0.], [0., 0.]]]])
    }
    expected_coarse_value = {
        token_hierarchy.TokenBlockName.ANCHOR:
            diag_qkv,
        token_hierarchy.TokenBlockName.LEFT:
            np.array([[[[0., 0.], [0., 0.]], [[1., 2.], [3., 4.]],
                       [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]],
                       [[0., 0.], [0., 0.]], [[4., 6.], [12., 14.]]]]),
        token_hierarchy.TokenBlockName.RIGHT:
            np.array([[[[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]],
                       [[13., 14.], [15., 16.]], [[0., 0.], [0., 0.]],
                       [[20., 22.], [28., 30.]], [[0., 0.], [0., 0.]]]])
    }

    for dict_key, coarse_q in coarse_query.items():
      np.testing.assert_array_almost_equal(coarse_q,
                                           expected_coarse_query[dict_key])
      np.testing.assert_array_almost_equal(coarse_key[dict_key],
                                           expected_coarse_key[dict_key])
      np.testing.assert_array_almost_equal(coarse_value[dict_key],
                                           expected_coarse_value[dict_key])

  @parameterized.named_parameters(
      ('boolean_mask', True),
      ('int_mask', False),
  )
  def test_hierarchical_coarsen_with_padding(self, boolean_mask):
    batch_size = 1
    num_head = 1
    head_dim = 2
    feature_size = num_head * head_dim
    num_level = 2
    num_cluster = 2
    num_block = int(np.exp2(num_level))
    seq_len = num_block * num_cluster
    padding_len = 3

    seq_shape = [batch_size, seq_len, feature_size]
    data_size = np.prod(seq_shape)
    inputs_q = jnp.arange(1, data_size + 1).reshape(tuple(seq_shape))

    padding_mask = np.ones((batch_size, seq_len, 1))
    padding_mask[:, -padding_len:] = 0
    if boolean_mask:
      padding_mask = padding_mask > 0

    hierarchy = token_hierarchy.OneDimTokenHierarchy(
        seq_len=seq_len,
        num_cluster=num_cluster,
        for_self_attention=True,
        causal_mask=False)
    results = hierarchy.hierarchical_coarsen(
        inputs_q,
        inputs_q,
        query_padding_mask=padding_mask,
        key_padding_mask=padding_mask)
    coarse_qkv = results.packed_coarse_qkv
    aggregated_padding_mask = results.packed_aggregated_key_padding_mask

    coarse_query, _, _ = coarse_qkv.values()

    partitioned_shape = tuple(
        (batch_size, num_block, num_cluster, feature_size))
    inputs_q *= padding_mask
    diag_qkv = inputs_q.reshape(partitioned_shape)
    expected_coarse_query = {
        token_hierarchy.TokenBlockName.ANCHOR:
            diag_qkv,
        token_hierarchy.TokenBlockName.LEFT:
            np.array([[[[0., 0.], [0., 0.]], [[5., 6.], [7., 8.]],
                       [[9., 10.], [0., 0.]], [[0., 0.], [0., 0.]],
                       [[0., 0.], [0., 0.]], [[9., 10.], [0., 0.]]]]),
        token_hierarchy.TokenBlockName.RIGHT:
            np.array([[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]],
                       [[9., 10.], [0., 0.]], [[0., 0.], [0., 0.]],
                       [[2., 3.], [6., 7.]], [[0., 0.], [0., 0.]]]])
    }

    partitioned_mask_shape = tuple((batch_size, num_block, num_cluster, 1))
    diag_padding_mask = padding_mask.reshape(partitioned_mask_shape).astype(
        jnp.float32)
    expected_padding_mask = {
        token_hierarchy.TokenBlockName.ANCHOR:
            diag_padding_mask,
        token_hierarchy.TokenBlockName.LEFT:
            np.array([[[[0.], [0.]], [[1.], [1.]], [[1.], [1.]], [[1.], [0.]],
                       [[0.], [0.]], [[2.], [2.]]]]),
        token_hierarchy.TokenBlockName.RIGHT:
            np.array([[[[1.], [1.]], [[1.], [0.]], [[0.], [0.]], [[0.], [0.]],
                       [[1.], [0.]], [[0.], [0.]]]]),
    }

    for dict_key, coarse_q in coarse_query.items():
      np.testing.assert_array_almost_equal(coarse_q,
                                           expected_coarse_query[dict_key])
      np.testing.assert_array_equal(aggregated_padding_mask[dict_key],
                                    expected_padding_mask[dict_key])

  @parameterized.named_parameters(
      ('const', token_hierarchy.ConvKernelType.CONST,
       np.array([0, 0, 1, 1, 3, 3, 4, 4, 6, 6, 7, 7, 9, 9, 10, 10]).reshape(
           (1, 16, 1))),
      ('linear_with_correction', token_hierarchy.ConvKernelType.LINEAR,
       np.array([
           0, 0.75, 1.5, 2.25, 3, 3.75, 4.5, 5.25, 6, 6.75, 7.5, 8.25, 9, 9.5,
           10, 10
       ]).reshape((1, 16, 1))),
  )
  def test_interpolate_cumulative_sum(self, interpolation_kernel_type,
                                      expected_results):
    num_level = 3
    num_cluster = 2
    num_block = int(np.exp2(num_level))
    seq_len = num_block * num_cluster

    hierarchy = token_hierarchy.OneDimTokenHierarchy(
        seq_len=seq_len,
        num_cluster=num_cluster,
        interpolation_kernel_type=interpolation_kernel_type,
        for_self_attention=True,
        causal_mask=False)

    coarse_y = np.array([[[[0], [1]], [[2], [3]], [[4], [5]], [[6], [7]],
                          [[0], [1]], [[2], [3]]]])
    actual_results = hierarchy.interpolate_cumulative_sum(coarse_y)
    logging.info('interpolation_kernel_type=%s', interpolation_kernel_type)
    logging.info('expected_results=%s', expected_results)
    logging.info('actual_results=%s', actual_results)
    np.testing.assert_array_almost_equal(actual_results, expected_results)


if __name__ == '__main__':
  absltest.main()
