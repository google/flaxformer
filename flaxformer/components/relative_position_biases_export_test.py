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

"""Tests for relative_position_biases."""
from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
import jax
from jax.experimental import jax2tf  # type: ignore[import]
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from flaxformer.components import relative_position_biases


class RelativePositionBiasesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.num_heads = 3
    self.query_len = 5
    self.key_len = 7

  @parameterized.product(on_device_computation=[True, False])
  def test_relative_position_with_on_device_computation_exporting(
      self, on_device_computation):

    class TestModule(nn.Module):
      """A test Module that simply uses relpos layer."""

      @nn.compact
      def __call__(self, x):
        return relative_position_biases.RelativePositionBiases(
            num_buckets=12,
            max_distance=10,
            num_heads=3,
            dtype=jnp.float32,
            on_device_computation=on_device_computation,
        )(qlen=x.shape[0], klen=x.shape[-1], bidirectional=True, decode=False)

    inputs = np.ones((self.query_len, self.key_len))
    test_module = TestModule()
    params = test_module.init(jax.random.PRNGKey(0), inputs)

    class ExportableModule(tf.Module):
      """A mini Module that is exportable to TensorFlow."""

      def __init__(self, params, apply_fn):

        def create_var(value):
          return tf.Variable(value)

        self._params = jax.tree.map(create_var, params)
        # Use jax2tf graph serialization because test inspects the graph.
        self._apply = jax2tf.convert(apply_fn, native_serialization=False)
        self._apply = tf.autograph.experimental.do_not_convert(self._apply)

      def __call__(self, x):
        return self._apply(self._params, x)

    # Export the module to SavedModel.
    module = ExportableModule(params=params, apply_fn=test_module.apply)

    @tf.function(autograph=False)
    def forward(x):
      return module(x)

    to_save = tf.Module()
    to_save.forward = forward
    to_save.params = list(module.variables)
    signatures = {
        'serving_default':
            forward.get_concrete_function(
                tf.TensorSpec((self.query_len, self.key_len), dtype=tf.int32)),
    }

    export_dir = self.create_tempdir('export_test').full_path
    tf.saved_model.save(
        to_save,
        export_dir,
        signatures=signatures,
    )

    # Inspect whether the graph has a constant embedded.
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
      metagraph = tf.compat.v1.saved_model.loader.load(sess, ['serve'],
                                                       export_dir)
    has_embedded_const_tensor = False
    for f in metagraph.graph_def.library.function:
      for n in f.node_def:
        if n.op == 'Const':
          if [d.size for d in n.attr['value'].tensor.tensor_shape.dim
             ] == [1, self.query_len, self.key_len]:
            has_embedded_const_tensor = True
            break

    # Using on_device_computation should give us a graph without constants.
    self.assertEqual(has_embedded_const_tensor, not on_device_computation)


if __name__ == '__main__':
  absltest.main()
