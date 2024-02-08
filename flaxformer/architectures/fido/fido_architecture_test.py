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

"""Tests for FiDO architecture."""

import functools

from absl.testing import absltest
from flax import linen as nn
from jax import random
import jax.numpy as jnp
import numpy as np

from flaxformer import testing_utils
from flaxformer.architectures.fido import fido_architecture
from flaxformer.architectures.t5 import t5_architecture
from flaxformer.architectures.t5 import t5_common_layers
from flaxformer.components import embedding
from flaxformer.components import layer_norm

expected_files = testing_utils.ExpectedJsonFiles(
    'flaxformer/architectures/fido/testdata'
)
check_params = expected_files.check_params_shapes_only

# TODO mutate scanned test

BATCH_SIZE = 2
SRC_LEN = 6
TARGET_LEN = 4
NUM_HEADS = 8
EMBED_DIM = 13
MLP_DIM = 32
HEAD_DIM = 3
DTYPE = jnp.float32
NUM_LAYERS = 2
DROPOUT_RATE = 0.1
ACTIVATIONS = ('gelu', 'linear')
VOCAB_SIZE = 4

dropout_factory = lambda: nn.Dropout(rate=DROPOUT_RATE, broadcast_dims=(-2,))
embedding_factory = lambda: embedding.Embed(VOCAB_SIZE, EMBED_DIM)
layer_norm_factory = layer_norm.T5LayerNorm
relative_position_bias_factory = functools.partial(
    t5_common_layers.relative_position_bias, num_heads=NUM_HEADS, dtype=DTYPE
)


def decoder_layer_factory(shared_relative_position_bias=None, scanned=False):
  return t5_architecture.DecoderLayer(
      self_attention=t5_common_layers.attention_layer(
          num_heads=NUM_HEADS,
          head_dim=HEAD_DIM,
          dropout_rate=DROPOUT_RATE,
          dtype=DTYPE,
      ),
      encoder_decoder_attention=t5_common_layers.attention_layer(
          num_heads=NUM_HEADS,
          head_dim=HEAD_DIM,
          dropout_rate=DROPOUT_RATE,
          dtype=DTYPE,
      ),
      mlp=t5_common_layers.mlp_block(
          mlp_dim=MLP_DIM,
          dropout_rate=DROPOUT_RATE,
          activations=ACTIVATIONS,
          dtype=DTYPE,
      ),
      dropout_factory=dropout_factory,
      layer_norm_factory=layer_norm_factory,
      shared_relative_position_bias=shared_relative_position_bias,
      scanned=scanned,
  )


def fido_decoder_factory(encoder_decoder_attention_period=1):
  return fido_architecture.Decoder(
      layer_factory=decoder_layer_factory,
      dropout_factory=dropout_factory,
      layer_norm_factory=layer_norm_factory,
      num_layers=2,
      token_embedder_factory=embedding_factory,
      shared_relative_position_bias_factory=relative_position_bias_factory,
      dtype=DTYPE,
      encoder_decoder_attention_period=encoder_decoder_attention_period,
  )  # pytype: disable=wrong-keyword-args


def t5_decoder_factory():
  return t5_architecture.Decoder(
      layer_factory=decoder_layer_factory,
      dropout_factory=dropout_factory,
      layer_norm_factory=layer_norm_factory,
      num_layers=2,
      token_embedder_factory=embedding_factory,
      shared_relative_position_bias_factory=relative_position_bias_factory,
      dtype=DTYPE,
  )


def decoder_block_factory(num_layers, scanned=False, **kwargs):
  return fido_architecture.DecoderLayerBlock(
      num_layers=num_layers,
      layer_factory=functools.partial(decoder_layer_factory),
      scanned=scanned,
      **kwargs,
  )


def decoder_with_blocks_factory(block_size, scan_layers):
  return t5_architecture.Decoder(
      layer_factory=functools.partial(
          decoder_block_factory, num_layers=block_size, scanned=scan_layers
      ),
      dropout_factory=dropout_factory,
      layer_norm_factory=layer_norm_factory,
      num_layers=2 // block_size,
      token_embedder_factory=embedding_factory,
      dtype=DTYPE,
      scan_layers=scan_layers,
  )  # pytype: disable=wrong-keyword-args


class FiDODecoderTest(absltest.TestCase):
  """Tests for FiDO decoder."""

  def test_shape(self):
    decoder_lsa1 = fido_decoder_factory(encoder_decoder_attention_period=1)
    encoder_outputs = np.ones(
        (BATCH_SIZE, SRC_LEN, EMBED_DIM), dtype=np.float32
    )
    decoder_input_tokens = np.ones((BATCH_SIZE, TARGET_LEN), dtype=np.int32)
    output_logits_lsa1, variables_lsa1 = decoder_lsa1.init_with_output(
        random.PRNGKey(0),
        encoder_outputs=encoder_outputs,
        decoder_input_tokens=decoder_input_tokens,
        enable_dropout=False,
    )
    check_params(variables_lsa1['params'], 'decoder_shapes_lsa1.json')
    self.assertEqual(
        output_logits_lsa1.shape, (BATCH_SIZE, TARGET_LEN, VOCAB_SIZE)
    )

    decoder_lsa2 = fido_decoder_factory(encoder_decoder_attention_period=2)
    encoder_outputs = np.ones(
        (BATCH_SIZE, SRC_LEN, EMBED_DIM), dtype=np.float32
    )
    decoder_input_tokens = np.ones((BATCH_SIZE, TARGET_LEN), dtype=np.int32)
    output_logits_lsa2, variables_lsa2 = decoder_lsa2.init_with_output(
        random.PRNGKey(0),
        encoder_outputs=encoder_outputs,
        decoder_input_tokens=decoder_input_tokens,
        enable_dropout=False,
    )
    check_params(variables_lsa2['params'], 'decoder_shapes_lsa2.json')
    self.assertEqual(
        output_logits_lsa2.shape, (BATCH_SIZE, TARGET_LEN, VOCAB_SIZE)
    )

  def test_consistent_t5(self):
    encoder_outputs = np.ones(
        (BATCH_SIZE, SRC_LEN, EMBED_DIM), dtype=np.float32
    )
    decoder_input_tokens = np.ones((BATCH_SIZE, TARGET_LEN), dtype=np.int32)
    decoder_lsa1 = fido_decoder_factory(encoder_decoder_attention_period=1)

    output_logits_lsa1, _ = decoder_lsa1.init_with_output(
        random.PRNGKey(0),
        encoder_outputs=encoder_outputs,
        decoder_input_tokens=decoder_input_tokens,
        enable_dropout=False,
    )

    decoder_t5 = t5_decoder_factory()
    output_logits_t5, _ = decoder_t5.init_with_output(
        random.PRNGKey(0),
        encoder_outputs=encoder_outputs,
        decoder_input_tokens=decoder_input_tokens,
        enable_dropout=False,
    )

    np.testing.assert_allclose(output_logits_lsa1, output_logits_t5, rtol=1e-8)


class FiDOScanTest(absltest.TestCase):
  """Tests for scanned FiDO decoder."""

  def test_shape(self):
    decoder_lsa1 = decoder_with_blocks_factory(block_size=1, scan_layers=True)
    encoder_outputs = np.ones(
        (BATCH_SIZE, SRC_LEN, EMBED_DIM), dtype=np.float32
    )
    decoder_input_tokens = np.ones((BATCH_SIZE, TARGET_LEN), dtype=np.int32)
    output_logits_lsa1, variables_lsa1 = decoder_lsa1.init_with_output(
        random.PRNGKey(0),
        encoder_outputs=encoder_outputs,
        decoder_input_tokens=decoder_input_tokens,
        enable_dropout=False,
    )
    check_params(variables_lsa1['params'], 'decoder_shapes_blocklsa1.json')
    self.assertEqual(
        output_logits_lsa1.shape, (BATCH_SIZE, TARGET_LEN, VOCAB_SIZE)
    )

    decoder_lsa2 = decoder_with_blocks_factory(block_size=2, scan_layers=True)
    output_logits_lsa2, variables_lsa2 = decoder_lsa2.init_with_output(
        random.PRNGKey(0),
        encoder_outputs=encoder_outputs,
        decoder_input_tokens=decoder_input_tokens,
        enable_dropout=False,
    )
    check_params(variables_lsa2['params'], 'decoder_shapes_blocklsa2.json')
    self.assertEqual(
        output_logits_lsa2.shape, (BATCH_SIZE, TARGET_LEN, VOCAB_SIZE)
    )

    decoder_lsa2_noscan = decoder_with_blocks_factory(
        block_size=2, scan_layers=False
    )
    output_logits_lsa2_noscan, variables_lsa2_noscan = (
        decoder_lsa2_noscan.init_with_output(
            random.PRNGKey(0),
            encoder_outputs=encoder_outputs,
            decoder_input_tokens=decoder_input_tokens,
            enable_dropout=False,
        )
    )
    check_params(
        variables_lsa2_noscan['params'], 'decoder_shapes_blocklsa2_noscan.json'
    )
    self.assertEqual(
        output_logits_lsa2_noscan.shape, (BATCH_SIZE, TARGET_LEN, VOCAB_SIZE)
    )


if __name__ == '__main__':
  absltest.main()
