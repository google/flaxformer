# Copyright 2021 Google LLC.
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

"""Tests for flaxformer.embedding."""
from absl.testing import absltest
from absl.testing import parameterized

from flax import linen as nn
from flax.linen import partitioning as flax_partitioning
import jax
import numpy as np

from flaxformer.components import embedding
from flaxformer.components import initializers


class EmbedTest(parameterized.TestCase):

  def test_embedder_raises_exception_for_incorrect_input_type(self):
    """Tests that inputs are integers and that an exception is raised if not."""
    embed = embedding.Embed(num_embeddings=10, features=5)
    inputs = np.expand_dims(np.arange(5, dtype=np.int64), 1)
    variables = embed.init(jax.random.PRNGKey(0), inputs)
    bad_inputs = inputs.astype(np.float32)
    with self.assertRaisesRegex(
        ValueError, 'Input type must be an integer or unsigned integer.'):
      _ = embed.apply(variables, bad_inputs)

  @parameterized.named_parameters(
      {
          'testcase_name': 'with_ones',
          'init_fn': jax.nn.initializers.ones,
          'num_embeddings': 10,
          'features': 5,
          'matrix_sum': 5 * 10,
      }, {
          'testcase_name': 'with_zeros',
          'init_fn': jax.nn.initializers.zeros,
          'num_embeddings': 10,
          'features': 5,
          'matrix_sum': 0,
      })
  def test_embedding_initializes_correctly(self, init_fn, num_embeddings,
                                           features, matrix_sum):
    """Tests if the Embed class initializes with the requested initializer."""
    embed = embedding.Embed(
        num_embeddings=num_embeddings,
        features=features,
        embedding_init=init_fn)
    inputs = np.expand_dims(np.arange(5, dtype=np.int64), 1)
    variables = embed.init(jax.random.PRNGKey(0), inputs)
    embedding_matrix = variables['params']['embedding']
    self.assertEqual(int(np.sum(embedding_matrix)), matrix_sum)

  def test_embedding_matrix_shape(self):
    """Tests that the embedding matrix has the right shape."""
    num_embeddings = 10
    features = 5
    embed = embedding.Embed(num_embeddings=num_embeddings, features=features)
    inputs = np.expand_dims(np.arange(features, dtype=np.int64), 1)
    variables = embed.init(jax.random.PRNGKey(0), inputs)
    embedding_matrix = variables['params']['embedding']
    self.assertEqual((num_embeddings, features), embedding_matrix.shape)

  def test_embedding_attend(self):
    """Tests that attending with ones returns sum of embedding vectors."""
    features = 5
    embed = embedding.Embed(num_embeddings=10, features=features)
    inputs = np.array([[1]], dtype=np.int64)
    variables = embed.init(jax.random.PRNGKey(0), inputs)
    query = np.ones(features, dtype=np.float32)
    result = embed.apply(variables, query, method=embed.attend)
    expected = np.sum(variables['params']['embedding'], -1)
    np.testing.assert_array_almost_equal(result, expected)

  def test_embedding_axis_names(self):
    rules = [
        ('my_batch_dim', 'data'),
        ('embed', None),
        ('vocab', None),
    ]
    with flax_partitioning.axis_rules(rules):
      embed = embedding.Embed(num_embeddings=10, features=5)
      inputs = np.array([1], dtype=np.int64)
      embed.init(
          jax.random.PRNGKey(0), inputs, input_axis_names=('my_batch_dim',))


class MultiEmbedTest(parameterized.TestCase):

  def test_multi_embed_returns_correct_shape(self):
    """Tests that we can build a generic combined embedder."""
    features = 5
    embedders = {
        'token_embed': nn.Embed(num_embeddings=10, features=features),
        'segment_embed': nn.Embed(num_embeddings=2, features=features),
        'position_embed': nn.Embed(num_embeddings=12, features=features)
    }
    model = embedding.MultiEmbed(embedders)

    token_ids = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    segment_ids = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.int64)
    position_ids = np.arange(3, dtype=np.int64)[None]
    output, _ = model.init_with_output(
        jax.random.PRNGKey(0),
        token_embed=token_ids,
        segment_embed=segment_ids,
        position_embed=position_ids)
    self.assertEqual(output.shape, token_ids.shape + (features,))

  @parameterized.named_parameters(
      {
          'testcase_name': 'with_sum_method',
          'method': embedding.EmbedCombineMethod.SUM,
          'features': 7,
          'expected_shape': (2, 3, 7),
      }, {
          'testcase_name': 'with_concat_method',
          'method': embedding.EmbedCombineMethod.CONCAT,
          'features': 7,
          'expected_shape': (2, 3, 14),
      })
  def test_multi_embed_combines_embeddings_correctly(self, method, features,
                                                     expected_shape):
    """Tests that embeddings are correctly summed or concatenated."""
    embedders = {
        'token_embed': nn.Embed(num_embeddings=10, features=features),
        'segment_embed': nn.Embed(num_embeddings=2, features=features)
    }
    model = embedding.MultiEmbed(embedders)

    token_ids = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    segment_ids = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.int64)
    variables = model.init(
        jax.random.PRNGKey(0), token_embed=token_ids, segment_embed=segment_ids)
    embeddings = model.apply(
        variables,
        token_embed=token_ids,
        segment_embed=segment_ids,
        method=model.get_individual_embeddings)
    output = model.apply(
        variables,
        embeddings,
        combine_method=method,
        method=model.combine_embeddings)
    self.assertEqual(output.shape, expected_shape)

  def test_multi_embed_combine_embeddings_raises_exception(self):
    """Tests that an exception is raised for an invalid combine method."""
    model = embedding.MultiEmbed({
        'token_embed': nn.Embed(num_embeddings=10, features=5),
    })
    token_ids = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    variables = model.init(jax.random.PRNGKey(0), token_embed=token_ids)
    embeddings = {'token_embed': np.ones([2, 3, 5], dtype=np.float32)}
    with self.assertRaises(ValueError):
      _ = model.apply(
          variables,
          embeddings,
          combine_method='invalid_combine_method',
          method=model.combine_embeddings)

  def test_multi_embed_can_return_individual_embeddings(self):
    """Tests that MultiEmbed returns a dictionary with each embedded input."""
    features = 5
    embedders = {
        'token_embed': nn.Embed(num_embeddings=10, features=features),
        'segment_embed': nn.Embed(num_embeddings=2, features=features),
        'position_embed': nn.Embed(num_embeddings=12, features=features)
    }
    model = embedding.MultiEmbed(embedders)

    token_ids = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    segment_ids = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.int64)
    position_ids = np.arange(3, dtype=np.int64)[None]
    variables = model.init(
        jax.random.PRNGKey(0),
        token_embed=token_ids,
        segment_embed=segment_ids,
        position_embed=position_ids)
    embeddings = model.apply(
        variables,
        token_embed=token_ids,
        segment_embed=segment_ids,
        position_embed=position_ids,
        method=model.get_individual_embeddings)
    self.assertIn('token_embed', embeddings)
    self.assertIn('segment_embed', embeddings)
    self.assertIn('position_embed', embeddings)


class EmbeddingTest(parameterized.TestCase):

  def test_add_position_embs(self):
    """Tests that positional embeddings are correctly applied."""
    positions = np.array([[[0, 1, 2], [0, 1, 2]]], dtype=np.int32)
    position_embedder = embedding.PositionEmbed(
        num_embeddings=10,
        features=4,
        dtype=np.float32,
        embedding_init=jax.nn.initializers.ones)

    variables = position_embedder.init(jax.random.PRNGKey(0), inputs=positions)
    output_embeddings = position_embedder.apply(variables, inputs=positions)

    np.testing.assert_array_equal(output_embeddings, 1)

  def test_add_position_embs_decoder(self):
    """Tests that position embeddings are correctly applied in decoding mode."""
    positions = np.array([[[0, 1, 2], [0, 1, 2]]], dtype=np.int32)
    position_embedder = embedding.PositionEmbed(
        num_embeddings=10,
        features=3,
        dtype=np.float32,
        embedding_init=jax.nn.initializers.ones)
    variables = position_embedder.init(
        jax.random.PRNGKey(0), inputs=positions, decode=True)
    state, params = variables.pop('params')

    output_embeddings, state = position_embedder.apply(
        {
            'params': params,
            **state
        },
        inputs=positions,
        decode=True,
        mutable=['cache'])

    np.testing.assert_array_equal(output_embeddings, 1)
    np.testing.assert_array_equal(state['cache']['position_embedder_index'], 1)

    # Test that repeated access increments the cache_index.
    output_embeddings, state = position_embedder.apply(
        {
            'params': params,
            **state
        },
        inputs=positions,
        decode=True,
        mutable=['cache'])

    np.testing.assert_array_equal(output_embeddings, 1)
    np.testing.assert_array_equal(state['cache']['position_embedder_index'], 2)

  def test_add_sinusoidal_position_embs(self):
    """Tests that sinusoidal positional embeddings are applied."""
    positions = np.array([[0, 1]])
    sinusoid_embedder = embedding.FixedEmbed(
        embedding_init=initializers.sinusoidal(),
        features=5,
        max_length=10,
        dtype=np.float32)

    variables = sinusoid_embedder.init(jax.random.PRNGKey(0), inputs=positions)
    output_embeddings = sinusoid_embedder.apply(variables, inputs=positions)

    expected_output_embeddings = np.array(
        [[[0.0, 0.0, 1.0, 1.0, 0.0], [0.84147, 0.0001, 0.540302, 1.0, 0.0]]],
        dtype=np.float32)
    np.testing.assert_array_almost_equal(output_embeddings,
                                         expected_output_embeddings)

  def test_regression_add_position_embeddings_sine(self):
    sequence_length = 7
    hidden_dim = 5
    positions = np.arange(sequence_length)[None, :]
    embed = embedding.FixedEmbed(
        embedding_init=initializers.sinusoidal(),
        features=hidden_dim,
        max_length=32,
        dtype=np.float32)
    outputs, params = embed.init_with_output(
        jax.random.PRNGKey(0), inputs=positions)
    self.assertEqual(params, {})
    expected = np.array([
        [
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [
                0.8414709568023682, 9.999999747378752e-05, 0.5403022766113281,
                1.0, 0.0
            ],
            [
                0.9092974066734314, 0.00019999999494757503, -0.416146844625473,
                1.0, 0.0
            ],
            [
                0.14112000167369843, 0.00029999998514540493,
                -0.9899924993515015, 0.9999999403953552, 0.0
            ],
            [
                -0.756802499294281, 0.00039999998989515007, -0.6536436080932617,
                0.9999999403953552, 0.0
            ],
            [
                -0.9589242935180664, 0.0004999999655410647, 0.28366219997406006,
                0.9999998807907104, 0.0
            ],
            [
                -0.279415488243103, 0.0005999999702908099, 0.9601702690124512,
                0.9999998211860657, 0.0
            ],
        ],
    ])
    np.testing.assert_allclose(outputs, expected, rtol=1e-5)

  def test_regression_add_position_embeddings_learned(self):
    sequence_length = 7
    hidden_dim = 5
    positions = np.arange(sequence_length)[None, :]
    embeds = embedding.PositionEmbed(
        num_embeddings=32,
        features=hidden_dim,
        embedding_init=jax.nn.initializers.normal(
            stddev=1e-6),  # Use learned embeds.
        dtype=np.float32)
    outputs, params = embeds.init_with_output(
        jax.random.PRNGKey(0), inputs=positions)
    param_shapes = jax.tree_map(lambda x: x.shape, params)
    self.assertEqual(param_shapes['params'], {
        'pos_embedding': (32, 5),
    })
    np.testing.assert_allclose(
        outputs, [
            [
                [
                    6.417410531867063e-07, -9.799798306175944e-08,
                    5.555829716286098e-07, -1.6317341078320169e-06,
                    -3.149363578813791e-07
                ],
                [
                    1.046715851771296e-06, 1.139310299436147e-07,
                    1.0767863045657577e-07, -1.2913797036162578e-06,
                    -5.156626912139473e-07
                ],
                [
                    1.3656243709192495e-06, -1.7889256014314014e-06,
                    5.730560133088147e-07, -1.1404511042201193e-06,
                    6.165008699099417e-07
                ],
                [
                    -2.014825213336735e-07, -6.456507435359526e-07,
                    -3.109490762653877e-07, -7.383717814946067e-08,
                    -1.905724502648809e-06
                ],
                [
                    1.604741726168868e-07, 7.540054411947494e-07,
                    5.639509481625282e-07, -1.6647313714202028e-06,
                    9.162973242382577e-07
                ],
                [
                    2.2114620890079095e-07, -1.906814873109397e-06,
                    -1.786764187272638e-06, -2.2142488376175606e-07,
                    -1.5123981711440138e-06
                ],
                [
                    -2.9822021474501526e-07, 2.7386813599150628e-06,
                    -2.150136197087704e-06, 2.099384346365696e-06,
                    4.0360737330047414e-07
                ],
            ],
        ],
        rtol=1e-5)


class HashEmbedTest(parameterized.TestCase):

  def test_hash_embedder(self):
    """Checks the parameters and return value shapes."""
    num_tables = 2
    num_embeddings_per_table = 32
    features = 16
    embedder = embedding.HashEmbed(
        features=features,
        num_embeddings_per_table=num_embeddings_per_table,
        num_tables=num_tables)

    key = jax.random.PRNGKey(0)
    batch_size = 2
    seq_len = 8
    ids = np.ones([batch_size, seq_len], dtype=np.int32)
    outputs, variables = embedder.init_with_output(key, ids)
    self.assertSequenceEqual(outputs.shape, (batch_size, seq_len, features))

    param_shapes = jax.tree_map(lambda x: list(x.shape), variables['params'])
    self.assertSameStructure(
        param_shapes, {
            'hash_embedder_table_0': {
                'embedding': [num_embeddings_per_table, features // num_tables]
            },
            'hash_embedder_table_1': {
                'embedding': [num_embeddings_per_table, features // num_tables]
            },
        })

  def test_hash_embedder_4d(self):
    """Checks the parameters and return value shapes."""
    num_tables = 2
    num_embeddings_per_table = 32
    features = 16
    embedder = embedding.HashEmbed(
        features=features,
        num_embeddings_per_table=num_embeddings_per_table,
        num_tables=num_tables)

    key = jax.random.PRNGKey(0)
    batch_size = 2
    seq_len = 8
    another_dim = 4
    ids = np.ones([batch_size, seq_len, another_dim], dtype=np.int32)
    outputs, variables = embedder.init_with_output(key, ids)
    self.assertSequenceEqual(outputs.shape,
                             (batch_size, seq_len, another_dim, features))

    param_shapes = jax.tree_map(lambda x: list(x.shape), variables['params'])
    self.assertSameStructure(
        param_shapes, {
            'hash_embedder_table_0': {
                'embedding': [num_embeddings_per_table, features // num_tables]
            },
            'hash_embedder_table_1': {
                'embedding': [num_embeddings_per_table, features // num_tables]
            },
        })


class NgramHashEmbedTest(parameterized.TestCase):

  @parameterized.product(
      batch_sizes=[(2,), (2, 3)],
      use_segment_ids=[False, True],
  )
  def test_hash_embedder(self, batch_sizes, use_segment_ids):
    """Checks the parameters and return value shapes."""
    num_tables = 2
    num_embeddings_per_table = 32
    features = 16
    embedder = embedding.NgramHashEmbed(
        ngram_orders=[1, 3, 4],
        padding_id=0,
        features=features,
        num_embeddings_per_table=num_embeddings_per_table,
        num_tables=num_tables)

    key = jax.random.PRNGKey(0)
    seq_len = 8
    ids = np.ones([*batch_sizes, seq_len], dtype=np.int32)
    segment_ids = (
        np.tile([[0] * 5 + [1] * 3],
                (*batch_sizes, 1)) if use_segment_ids else None)
    outputs, variables = (
        embedder.init_with_output(key, ids, segment_ids=segment_ids))
    self.assertSequenceEqual(outputs.shape, (*batch_sizes, seq_len, features))

    param_shapes = jax.tree_map(lambda x: list(x.shape), variables['params'])
    expected_table_shape = [num_embeddings_per_table, features // num_tables]
    self.assertSameStructure(
        param_shapes, {
            '1gram_hash_embed_table_0': {
                'embedding': expected_table_shape
            },
            '1gram_hash_embed_table_1': {
                'embedding': expected_table_shape
            },
            '3gram_hash_embed_table_0': {
                'embedding': expected_table_shape
            },
            '3gram_hash_embed_table_1': {
                'embedding': expected_table_shape
            },
            '4gram_hash_embed_table_0': {
                'embedding': expected_table_shape
            },
            '4gram_hash_embed_table_1': {
                'embedding': expected_table_shape
            },
        })

  @parameterized.product(
      batch_sizes=[(2,), (2, 3)],)
  def test_packing_correctness(self, batch_sizes):
    """Checks the parameters and return value shapes."""
    num_tables = 2
    num_embeddings_per_table = 32
    features = 16
    embedder = embedding.NgramHashEmbed(
        ngram_orders=[1, 3, 4],
        padding_id=0,
        features=features,
        num_embeddings_per_table=num_embeddings_per_table,
        num_tables=num_tables)

    key = jax.random.PRNGKey(0)
    seq_len = 8
    segment_ids = np.tile([[0] * 5 + [1] * 3], (*batch_sizes, 1))

    ids1 = np.ones([*batch_sizes, seq_len], dtype=np.int32)
    outputs1, variables = (
        embedder.init_with_output(key, ids1, segment_ids=segment_ids))

    # Run the embedder again, with the same IDs passed in for segment=0 IDs, but
    # different IDs for segment=1.
    ids2 = ids1 + segment_ids
    outputs2 = embedder.apply(variables, ids2, segment_ids=segment_ids)

    # Verify that the change to segment=1 didn't alter the outputs of segment=0.
    np.testing.assert_allclose(outputs1[..., :5, :], outputs2[..., :5, :])


if __name__ == '__main__':
  absltest.main()
