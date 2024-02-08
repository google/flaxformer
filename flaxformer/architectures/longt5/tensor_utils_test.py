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

"""Tests for tensor_utils."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

from flaxformer.architectures.longt5 import tensor_utils


class TensorUtilsTest(parameterized.TestCase):

  def test_pad_to_multiple_1d(self):
    array = np.arange(3) + 1

    np.testing.assert_array_equal(
        tensor_utils.pad_to_multiple(array, factor=3, axis=0), [1, 2, 3])

    np.testing.assert_array_equal(
        tensor_utils.pad_to_multiple(array, factor=5, axis=0), [1, 2, 3, 0, 0])

    np.testing.assert_array_equal(
        tensor_utils.pad_to_multiple(array, factor=7, axis=0),
        [1, 2, 3, 0, 0, 0, 0])

    np.testing.assert_array_equal(
        tensor_utils.pad_to_multiple(array, factor=2, axis=0), [1, 2, 3, 0])

    np.testing.assert_array_equal(
        tensor_utils.pad_to_multiple(array, factor=1, axis=0), [1, 2, 3])

    np.testing.assert_array_equal(
        tensor_utils.pad_to_multiple(array, factor=jnp.array(7), axis=0),
        [1, 2, 3, 0, 0, 0, 0])

    np.testing.assert_array_equal(
        tensor_utils.pad_to_multiple(array, factor=jnp.array(1), axis=0),
        [1, 2, 3])

  @parameterized.named_parameters(
      ('int_factor', 5),
      ('array_factor', np.array(5)),
  )
  def test_pad_to_multiple_padding_mode(self, factor):
    array = np.arange(3) + 1

    np.testing.assert_array_equal(
        tensor_utils.pad_to_multiple(
            array, factor=factor, axis=0, mode='reflect'), [1, 2, 3, 2, 1])

    np.testing.assert_array_equal(
        tensor_utils.pad_to_multiple(
            array, factor=factor, axis=0, mode='symmetric'), [1, 2, 3, 3, 2])

    np.testing.assert_array_equal(
        tensor_utils.pad_to_multiple(
            array, factor=factor, axis=0, constant_values=-1),
        [1, 2, 3, -1, -1])

  @parameterized.named_parameters(
      ('int_factor', 4),
      ('array_factor', np.array(4)),
  )
  def test_pad_to_multiple_2d(self, factor):
    array = np.ones([3, 5], dtype=np.float32)

    np.testing.assert_array_equal(
        tensor_utils.pad_to_multiple(array, factor=factor, axis=0),
        [
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1],  #
            [0, 0, 0, 0, 0],  #
        ])

    np.testing.assert_array_equal(
        tensor_utils.pad_to_multiple(array, factor=factor, axis=-1),
        [
            [1, 1, 1, 1, 1, 0, 0, 0],  #
            [1, 1, 1, 1, 1, 0, 0, 0],  #
            [1, 1, 1, 1, 1, 0, 0, 0],  #
        ])

    np.testing.assert_array_equal(
        tensor_utils.pad_to_multiple(array, factor=factor + 1, axis=1), array)

  @parameterized.named_parameters(
      ('int_factor', 3),
      ('array_factor', np.array(3)),
  )
  def test_pad_to_multiple_3d(self, factor):
    array = np.ones([2, 3, 5], dtype=np.float32)

    np.testing.assert_array_equal(
        tensor_utils.pad_to_multiple(array, factor=factor, axis=0),
        [
            [
                [1, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 1],  #
            ],  #
            [
                [1, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 1],  #
            ],  #
            [
                [0, 0, 0, 0, 0],  #
                [0, 0, 0, 0, 0],  #
                [0, 0, 0, 0, 0],  #
            ]
        ])

    np.testing.assert_array_equal(
        tensor_utils.pad_to_multiple(array, factor=factor, axis=-2), array)

    np.testing.assert_array_equal(
        tensor_utils.pad_to_multiple(array, factor=factor, axis=-1),
        [
            [
                [1, 1, 1, 1, 1, 0],  #
                [1, 1, 1, 1, 1, 0],  #
                [1, 1, 1, 1, 1, 0],  #
            ],  #
            [
                [1, 1, 1, 1, 1, 0],  #
                [1, 1, 1, 1, 1, 0],  #
                [1, 1, 1, 1, 1, 0],  #
            ]
        ])

  @parameterized.named_parameters(
      ('int_block_len', lambda x: x),
      ('array_block_len', jnp.array),
  )
  def test_split_into_blocks_1d(self, wrap_fn):
    array = np.arange(6) + 1

    np.testing.assert_array_equal(
        tensor_utils.split_into_blocks(array, block_len=wrap_fn(2), axis=0),
        [[1, 2], [3, 4], [5, 6]])

    np.testing.assert_array_equal(
        tensor_utils.split_into_blocks(array, block_len=wrap_fn(3), axis=0),
        [[1, 2, 3], [4, 5, 6]])

    np.testing.assert_array_equal(
        tensor_utils.split_into_blocks(array, block_len=wrap_fn(4), axis=0),
        [[1, 2, 3, 4], [5, 6, 0, 0]])

    np.testing.assert_array_equal(
        tensor_utils.split_into_blocks(array, block_len=wrap_fn(5), axis=0),
        [[1, 2, 3, 4, 5], [6, 0, 0, 0, 0]])

    np.testing.assert_array_equal(
        tensor_utils.split_into_blocks(array, block_len=wrap_fn(6), axis=0),
        [[1, 2, 3, 4, 5, 6]])

    np.testing.assert_array_equal(
        tensor_utils.split_into_blocks(array, block_len=wrap_fn(7), axis=0),
        [[1, 2, 3, 4, 5, 6, 0]])

    np.testing.assert_array_equal(
        tensor_utils.split_into_blocks(
            array, block_len=wrap_fn(8), axis=0, pad_value=-1),
        [[1, 2, 3, 4, 5, 6, -1, -1]])

  @parameterized.named_parameters(
      ('int_block_len', lambda x: x),
      ('array_block_len', jnp.array),
  )
  def test_split_into_blocks_3d(self, wrap_fn):
    # shape: [2, 4, 2]
    array = [
        [[1, -1], [2, -2], [3, -3], [4, -4]],  #
        [[11, 21], [12, 22], [13, 23], [14, 24]]
    ]

    np.testing.assert_array_equal(
        tensor_utils.split_into_blocks(array, block_len=wrap_fn(2), axis=-2),
        [
            [
                [[1, -1], [2, -2]],  #
                [[3, -3], [4, -4]],  #
            ],
            [
                [[11, 21], [12, 22]],  #
                [[13, 23], [14, 24]],  #
            ]
        ])

    np.testing.assert_array_equal(
        tensor_utils.split_into_blocks(array, block_len=wrap_fn(3), axis=1),
        [
            [
                [[1, -1], [2, -2], [3, -3]],  #
                [[4, -4], [0, 0], [0, 0]],  #
            ],
            [
                [[11, 21], [12, 22], [13, 23]],  #
                [[14, 24], [0, 0], [0, 0]],  #
            ]
        ])

    np.testing.assert_array_equal(
        tensor_utils.split_into_blocks(array, block_len=wrap_fn(3), axis=-1),
        [
            [
                [[1, -1, 0]],  #
                [[2, -2, 0]],  #
                [[3, -3, 0]],  #
                [[4, -4, 0]],  #
            ],
            [
                [[11, 21, 0]],  #
                [[12, 22, 0]],  #
                [[13, 23, 0]],  #
                [[14, 24, 0]],  #
            ],
        ])

  def test_concat_3_blocks(self):
    # shape: [batch=2, num_blocks=3, block_len=2, hidden_size=2]
    blocked_seq = [
        [
            [[1, -1], [2, -2]],  #
            [[3, -3], [4, -4]],  #
            [[5, -5], [6, -6]],  #
        ],  #
        [
            [[.1, -.1], [.2, -.2]],  #
            [[.3, -.3], [.4, -.4]],  #
            [[.5, -.5], [.6, -.6]],  #
        ],  #
    ]

    np.testing.assert_allclose(
        tensor_utils.concat_3_blocks(blocked_seq, block_axis=-3, seq_axis=-2),
        [
            [
                [[0, 0], [0, 0], [1, -1], [2, -2], [3, -3], [4, -4]],  #
                [[1, -1], [2, -2], [3, -3], [4, -4], [5, -5], [6, -6]],  #
                [[3, -3], [4, -4], [5, -5], [6, -6], [0, 0], [0, 0]],  #
            ],  #
            [
                [[0, 0], [0, 0], [.1, -.1], [.2, -.2], [.3, -.3], [.4, -.4]],  #
                [[.1, -.1], [.2, -.2], [.3, -.3], [.4, -.4], [.5, -.5],
                 [.6, -.6]],  #
                [[.3, -.3], [.4, -.4], [.5, -.5], [.6, -.6], [0, 0], [0, 0]],  #
            ],  #
        ])

  def test_concat_3_blocks_with_extra_dim(self):
    # shape: [batch=1, num_blocks=3, block_len=2, num_heads=1, size_per_head=2]
    blocked_seq = [[
        [[[1, -1]], [[2, -2]]],  #
        [[[3, -3]], [[4, -4]]],  #
        [[[5, -5]], [[6, -6]]],  #
    ]]

    np.testing.assert_array_equal(
        tensor_utils.concat_3_blocks(blocked_seq, block_axis=1, seq_axis=2),
        [[
            [[[0, 0]], [[0, 0]], [[1, -1]], [[2, -2]], [[3, -3]], [[4, -4]]],  #
            [[[1, -1]], [[2, -2]], [[3, -3]], [[4, -4]], [[5, -5]], [[6, -6]]],
            [[[3, -3]], [[4, -4]], [[5, -5]], [[6, -6]], [[0, 0]], [[0, 0]]],  #
        ]])

  @parameterized.parameters(
      dict(shape=(2, 3, 4, 5), block_axis=-3, seq_axis=-2),
      dict(shape=(1, 2, 3, 4, 5), block_axis=-4, seq_axis=-2),
  )
  def test_concat_3_blocks_one_hot(self, shape, block_axis, seq_axis):
    # Make sure the output from `concat_3_blocks_one_hot` is the same as
    # `concat_3_blocks`.
    seed = 1234
    np.random.seed(seed)
    array = np.random.randn(*shape)

    output = tensor_utils.concat_3_blocks_one_hot(array, block_axis, seq_axis)
    expected = tensor_utils.concat_3_blocks(array, block_axis, seq_axis)

    np.testing.assert_array_equal(output, expected)

  def test_make_3block_local_att_mask_no_segment_ids(self):
    input_mask = np.array(
        [
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 0, 0],  #
        ],
        dtype=np.bool_)

    np.testing.assert_array_equal(
        tensor_utils.make_3block_local_att_mask(2, input_mask),
        np.array(
            [
                [
                    [[0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0]],  #
                    [[0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0]],  #
                    [[0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]],  #
                ],  #
                [
                    [[0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0]],  #
                    [[0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]],  #
                    [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],  #
                ],  #
            ],
            dtype=np.bool_))

  def test_make_3block_local_att_mask_w_causal_mask(self):
    input_mask = np.array(
        [
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 0, 0],  #
        ],
        dtype=np.bool_)

    np.testing.assert_array_equal(
        tensor_utils.make_3block_local_att_mask(
            2, input_mask, use_causal_mask=True),
        np.array(
            [
                [
                    [[0, 0, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0]],  #
                    [[0, 1, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0]],  #
                    [[0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]],  #
                ],  #
                [
                    [[0, 0, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0]],  #
                    [[0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]],  #
                    [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],  #
                ],  #
            ],
            dtype=np.bool_))

  def test_make_3block_local_att_mask_with_segment_ids(self):
    input_mask = np.array(
        [
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 0, 0],  #
        ],
        dtype=np.bool_)
    segment_ids = [
        [1, 2, 2, 3, 3],  #
        [1, 1, 2, 0, 0],  #
    ]

    np.testing.assert_array_equal(
        tensor_utils.make_3block_local_att_mask(2, input_mask, segment_ids),
        np.array(
            [
                [
                    [[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0]],  #
                    [[0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0]],  #
                    [[0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]],  #
                ],  #
                [
                    [[0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0]],  #
                    [[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]],  #
                    [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],  #
                ],  #
            ],
            dtype=np.bool_))

  def test_make_3block_local_att_mask_no_segment_ids_full_block(self):
    input_mask = np.array(
        [
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 0, 0],  #
        ],
        dtype=np.bool_)

    np.testing.assert_array_equal(
        tensor_utils.make_3block_local_att_mask(
            2, input_mask, use_full_block_att=True),
        np.array(
            [
                [
                    [[0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]],  #
                    [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]],  #
                    [[1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]],  #
                ],  #
                [
                    [[0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 1, 0]],  #
                    [[1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]],  #
                    [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],  #
                ],  #
            ],
            dtype=np.bool_))

  def test_make_3block_local_att_mask_with_segment_ids_full_block(self):
    input_mask = np.array(
        [
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 0, 0],  #
        ],
        dtype=np.bool_)
    segment_ids = [
        [1, 2, 2, 3, 3],  #
        [1, 1, 2, 0, 0],  #
    ]

    np.testing.assert_array_equal(
        tensor_utils.make_3block_local_att_mask(
            2, input_mask, segment_ids, use_full_block_att=True),
        np.array(
            [
                [
                    [[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0]],  #
                    [[0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0]],  #
                    [[0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]],  #
                ],  #
                [
                    [[0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0]],  #
                    [[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]],  #
                    [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],  #
                ],  #
            ],
            dtype=np.bool_))

  def test_make_3block_relative_position(self):
    np.testing.assert_array_equal(
        tensor_utils.make_3block_relative_position(3),
        [
            [-3, -2, -1, 0, 1, 2, 3, 4, 5],  #
            [-4, -3, -2, -1, 0, 1, 2, 3, 4],  #
            [-5, -4, -3, -2, -1, 0, 1, 2, 3],  #
        ])

  def test_make_custom_3block_relative_position_simple_input(self):
    positions = np.arange(3, dtype=np.int32)[np.newaxis, :]

    # Unlike `make_3block_relative_position`, the "position" for all
    # padding tokens is set to -1, but this shouldn't matter since attention
    # to padding tokens should be masked out.
    np.testing.assert_array_equal(
        tensor_utils.make_custom_3block_relative_position(3, positions),
        [[[
            [-1, -1, -1, 0, 1, 2, -1, -1, -1],  #
            [-2, -2, -2, -1, 0, 1, -2, -2, -2],  #
            [-3, -3, -3, -2, -1, 0, -3, -3, -3],  #
        ]]])

  def test_make_custom_3block_relative_position_customized_input(self):
    positions = [
        [5, 4, 3, 2, 1, 0],  #
        [3, 4, 0, 1, 2, 5],  #
    ]

    np.testing.assert_array_equal(
        tensor_utils.make_custom_3block_relative_position(2, positions),
        [
            [
                [
                    [-6, -6, 0, -1, -2, -3],  #
                    [-5, -5, 1, 0, -1, -2],  #
                ],
                [
                    [2, 1, 0, -1, -2, -3],  #
                    [3, 2, 1, 0, -1, -2],  #
                ],
                [
                    [2, 1, 0, -1, -2, -2],  #
                    [3, 2, 1, 0, -1, -1],  #
                ],
            ],
            [
                [
                    [-4, -4, 0, 1, -3, -2],  #
                    [-5, -5, -1, 0, -4, -3],  #
                ],
                [
                    [3, 4, 0, 1, 2, 5],  #
                    [2, 3, -1, 0, 1, 4],  #
                ],
                [
                    [-2, -1, 0, 3, -3, -3],  #
                    [-5, -4, -3, 0, -6, -6],  #
                ],
            ],
        ])

  def test_positions_from_segment_ids(self):
    segment_ids = [
        [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3],  #
        [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 0, 0],  #
        [1, 1, 2, 3, 3, 4, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6],  #
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #
    ]

    expected_positions = [
        [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 0, 1, 2],  #
        [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 0, 1, 2, 3, 0, 0],  #
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 2, 3, 4, 5, 6, 7],  #
        list(range(16)),  #
    ]

    np.testing.assert_array_equal(
        expected_positions,
        tensor_utils.positions_from_segment_ids(segment_ids))



if __name__ == '__main__':
  absltest.main()
