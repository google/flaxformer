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

"""A library with embedding classes and functions."""

import abc
import collections
import dataclasses
import enum
import functools
import math
from typing import (Any, Callable, DefaultDict, Dict, Generic, List, Mapping,
                    Optional, Sequence, TypeVar, Union)

import chex
from flax import linen as nn
from flax.linen import partitioning
import jax
from jax import lax
from jax import numpy as jnp
from jax import tree_util

from flaxformer.components import initializers
from flaxformer.types import Array
from flaxformer.types import DType
from flaxformer.types import Initializer

# Note: We don't use this in real models, but keep the default initializers the
# same as in Flax.
default_embed_init = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal', out_axis=0)


@enum.unique
class EmbedCombineMethod(enum.IntEnum):
  # We use IntEnum here so that this class is serializable. Enum isn't.
  SUM = 1
  CONCAT = 2


_Inputs = TypeVar('_Inputs')


class Embedder(Generic[_Inputs], metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def __call__(self,
               inputs: _Inputs,
               *,
               segment_ids: Optional[Array] = None,
               decode: bool = False,
               enable_dropout: bool = True) -> Array:
    """Returns embeddings of the inputs.

    The generic type parameter `_Inputs` allows this interface to be used for
    embedding any type. For example, the base-level `Embed` class defined below
    inherits from `Embedder[Array]` since it is an embedder of `Array`s. At the
    other end of the spectrum, one could define a custom dataclass with that
    holds a combination of text, audio, and image inputs and interit from
    `Embedder[MyDataclass]`.

    Args:
      inputs: The inputs to embed.
      segment_ids: Input segmentation info for packed examples.
      decode: True if running in single-position autoregressive decode mode.
      enable_dropout: Enables dropout if set to True.

    Returns:
      The embedded inputs.
    """


class InspectableMultiEmbedder(Generic[_Inputs], Embedder[_Inputs]):
  """Interface for embedders that provide hooks for interpretability tools."""

  def __call__(self,
               inputs: _Inputs,
               *,
               segment_ids: Optional[Array] = None,
               decode: bool = False,
               enable_dropout: bool = True) -> Array:
    """Embeds inputs using get_individual_embeddings and combine_embeddings."""
    # Embed the inputs and pass results directly into `combine_embeddings`.
    return self.combine_embeddings(
        self.get_individual_embeddings(
            inputs,
            segment_ids=segment_ids,
            decode=decode,
            enable_dropout=enable_dropout))

  @abc.abstractmethod
  def get_individual_embeddings(self,
                                inputs: Array,
                                *,
                                segment_ids: Optional[Array] = None,
                                decode: bool = False,
                                enable_dropout: bool = True) -> Sequence[Array]:
    """Embeds the contents of each input array and returns the results."""

  @abc.abstractmethod
  def combine_embeddings(self, embeddings: Sequence[Array]) -> Array:
    """Combines the separate embeddings into a single array."""


class DictEmbedder(nn.Module, InspectableMultiEmbedder[Mapping[str, Any]]):
  """Embeds any number of inputs and combines them for further processing.

  Attributes:
    embedders: A dictionary with the name of the embedders as keys, and their
      embedding modules as values (usually Embed for input tokens, but can be
      any module). To embed inputs with these embedders, the dict used to call
      this class need to match the names of the embedders in this dictionary. If
      the resulting embeddings are to be summed, the `embedding_size` attributes
      of all embedders need to match, but that is not a requirement in case they
      are to be concatenated.
    embeddings_combiner: A function that determines how the results of the
      individual embedders should be combined.
  """
  embedders: Mapping[str, Embedder[Any]]
  embeddings_combiner: Callable[[Sequence[Array]], Array] = (
      sum  # pytype: disable=annotation-type-mismatch  # jax-ndarray
  )

  def get_individual_embeddings(
      self,  # pytype: disable=signature-mismatch  # jax-ndarray
      inputs: Mapping[str, Any],
      *,
      segment_ids: Optional[Array] = None,
      decode: bool = False,
      enable_dropout: bool = True,
  ) -> Sequence[Array]:
    """Embeds each keyword argument with its corresponding embedder.

    Args:
      inputs: The inputs to be embedded. All keys in `inputs` must be present in
        `self.embedders`. The shape of each input tensor should be <int>[...,
        seq_len]. When using a first batch dimension, the batch dimensions also
        need to match, or be 1 (to broadcast).
      segment_ids: Input segmentation info for packed examples.
      decode: Decoding parameter to pass through to all embedders.
      enable_dropout: Enables dropout if set to True.

    Returns:
      A list of individual embeddings, in the iteration order `inputs`. A tensor
      for an embedder with name `k` is shaped <float32>[..., embedding_size_k].
      If the embeddings are to be summed by `combine_embeddings`, then their
      embedding sizes should match.
    """
    if inputs.keys() != self.embedders.keys():
      raise ValueError(f'Expected input keys {self.embedders.keys()}, '
                       f'but got {inputs.keys()}')

    embeddings = []
    for k, v in inputs.items():
      embeddings.append(self.embedders[k](
          v,
          segment_ids=segment_ids,
          decode=decode,
          enable_dropout=enable_dropout))
    return embeddings

  def combine_embeddings(self, embeddings: Sequence[Array]) -> Array:
    """Combines the dictionary of embeddings using the combine method."""
    return self.embeddings_combiner(embeddings)


class EmbedderWithDecode(Generic[_Inputs], Embedder[_Inputs]):
  """Denotes embedder classes that support the `decode` parameter."""
  pass


class EmbedderWithDeterministic(Generic[_Inputs], Embedder[_Inputs]):
  """Denotes embedder classes that support the `deterministic` parameter."""
  pass


class Embed(nn.Module, Embedder[Array]):
  """An embedder for `Array`s.

  A parameterized function from integers [0, n) to d-dimensional vectors.

  Attributes:
    num_embeddings: number of embeddings.
    features: number of feature dimensions for each embedding.
    dtype: the dtype of the embedding vectors (default: float32).
    embedding_init: embedding initializer.
    one_hot: performs the gather with a one-hot contraction rather than a true
      gather. This is currently needed for SPMD partitioning.
    axes: default axis metadata names for the embedding table.
    input_axis_names: default axis metadata names for the input activations.
  """
  num_embeddings: int
  features: int
  cast_input_dtype: Optional[DType] = None
  dtype: DType = jnp.float32
  attend_dtype: Optional[DType] = None
  embedding_init: Initializer = default_embed_init  # pytype: disable=annotation-type-mismatch  # jax-types
  one_hot: bool = False
  axes: Sequence[str] = ('vocab', 'embed')
  input_axis_names: Sequence[str] = ('batch', 'length')
  embedding: Array = dataclasses.field(init=False)

  def setup(self):
    self.embedding = partitioning.param_with_axes(
        'embedding',
        self.embedding_init,
        (self.num_embeddings, self.features),
        jnp.float32,
        axes=tuple(self.axes),
    )

  def __call__(
      self,
      inputs: Array,
      *,
      segment_ids: Optional[Array] = None,
      decode: bool = False,
      enable_dropout: bool = True,
      input_axis_names: Optional[Sequence[str]] = None,
  ) -> Array:
    """Embeds the inputs along the last dimension.

    Args:
      inputs: input data, all dimensions are considered batch dimensions.
      segment_ids: Input segmentation info for packed examples.
      decode: True if running in single-position autoregressive decode mode.
      enable_dropout: Enables dropout if set to True.
      input_axis_names: Names of axes of input array. Used for logical
        activation sharding annotation. If None, then no output sharding
        annotation will be generated.

    Returns:
      Output which is embedded input data.  The output shape follows the input,
      with an additional `features` dimension appended.
    """
    del segment_ids  # Unused.
    if input_axis_names is None:
      input_axis_names = self.input_axis_names


    if self.cast_input_dtype:
      inputs = inputs.astype(self.cast_input_dtype)
    if not jnp.issubdtype(inputs.dtype, jnp.integer):
      raise ValueError('Input type must be an integer or unsigned integer.')
    if self.one_hot:
      iota = lax.iota(jnp.int32, self.num_embeddings)
      one_hot = jnp.array(inputs[..., jnp.newaxis] == iota, dtype=self.dtype)
      if input_axis_names is not None and self.axes:
        one_hot = partitioning.with_sharding_constraint(
            one_hot,
            tuple(input_axis_names) + (self.axes[0],))
      output = jnp.dot(one_hot, jnp.asarray(self.embedding, self.dtype))
    else:
      output = jnp.asarray(self.embedding, self.dtype)[inputs]
    if input_axis_names is not None and self.axes:
      output = partitioning.with_sharding_constraint(
          output,
          tuple(input_axis_names) + (self.axes[1],))
    return output

  def attend(self, query: Array) -> Array:
    """Attend over the embedding using a query array.

    Args:
      query: array with last dimension equal the feature depth `features` of the
        embedding.

    Returns:
      An array with final dim `num_embeddings` corresponding to the batched
      inner-product of the array of query vectors against each embedding.
      Commonly used for weight-sharing between embeddings and logit transform
      in NLP models.
    """
    dtype = self.attend_dtype if self.attend_dtype is not None else self.dtype
    return jnp.dot(query, jnp.asarray(self.embedding, dtype).T)


# DEPRECATED.
# TODO: Delete this in favor of the type-safe `DictEmbedder`.
class MultiEmbed(nn.Module):
  """Embeds any number of inputs and combines them for further processing.

  Attributes:
    embedders: A dictionary with the name of the embedders as keys, and their
      embedding modules as values (usually Embed for input tokens, but can be
      any module). To embed inputs with these embedders, the keyword arguments
      provided to the __call__-method of this class need to match the names of
      the embedders in this dictionary. If the resulting embeddings are to be
      summed, the `embedding_size` attributes of all embedders need to match,
      but that is not a requirement in case they are to be concatenated.
    sow_intermediates: whether to track intermediates using Module.sow.
    capture_gradients: whether to track input gradients using a variable in the
      `grads` collection. This captures the gradient of the (combined) embedded
      inputs, i.e. the output of this module which is usually the input to the
      first encoder layer.
  """
  embedders: Dict[str, Union[Embedder[Array], Callable[[Array], Array]]]
  sow_intermediates: bool = False
  capture_gradients: bool = False

  def get_individual_embeddings(
      self,
      decode: bool = False,
      deterministic: bool = False,
      segment_ids: Optional[Array] = None,
      **input_kwargs: Mapping[str, Array]) -> Dict[str, Array]:
    """Embeds each keyword argument with its corresponding embedder.

    The names of the keyword arguments need to match. To embed the input keyword
    argument 'word_embed', self.embedders['word_embed'] needs to exist.

    Args:
      decode: Decoding parameter to pass through to all embedders.
      deterministic: Deterministic parameter to pass through to all embedders.
      segment_ids: Input segmentation info for packed examples.
      **input_kwargs: The input tensors to be embedded, with a name that matches
        the embedder in self.embedders. The shape of each input tensor should be
        <int64>[..., seq_len]. When using a first batch dimension, the batch
        dimensions also need to match, or be 1 (to broadcast).

    Returns:
      A dictionary mapping the input keys to their embedded inputs. A tensor for
      an embedder with name `k` is shaped <float32>[..., embedding_size_k].
      If the embeddings are to be summed by `combine_embeddings`, then their
      embedding sizes should match.
    """
    if 'segment_ids' in self.embedders:
      if segment_ids is not None:
        input_kwargs = dict(**input_kwargs, segment_ids=segment_ids)

    embeddings = {}
    for k, v in input_kwargs.items():
      embedder: Callable[..., Array] = self.embedders[k]
      passthru_kwargs = {}
      if isinstance(embedder, EmbedderWithDecode):
        passthru_kwargs['decode'] = decode
      if isinstance(embedder, EmbedderWithDeterministic):
        passthru_kwargs['deterministic'] = deterministic
      if isinstance(embedder, Embedder):
        passthru_kwargs['segment_ids'] = segment_ids
      embeddings[k] = embedder(v, **passthru_kwargs)
    return embeddings

  def combine_embeddings(
      self,
      embeddings: Dict[str, Array],
      combine_method: EmbedCombineMethod = EmbedCombineMethod.SUM) -> Array:
    """Combines the dictionary of embeddings using the combine method.

    Args:
      embeddings: A dictionary containing the embeddings to be combined, with
        the names of the embeddings as keys, and embedding tensors as values.
        Each embedding `k` is shaped <float32>[..., seq_len, embedding_size_k].
        Embedding sizes need to match if they are to be summed.
      combine_method: The method used for combination: sum or concat.

    Returns:
      A tensor with the combined embeddings <float32>[..., embedding_size] in
      case of summing, and <float32>[..., size_1 + size_2 + ..] in case of
      concatenation.

    Raises:
      ValueError: If the given combine_method is unknown.
    """
    if combine_method == EmbedCombineMethod.SUM:
      return jax.tree_util.tree_reduce(
          lambda total, embedding: total + embedding, embeddings)
    elif combine_method == EmbedCombineMethod.CONCAT:
      return jnp.concatenate(tree_util.tree_leaves(embeddings), axis=-1)
    else:
      raise ValueError((
          f'Invalid combine_method {combine_method} given to combine_embeddings'
          '. Allowed values: sum, concat.'))

  @nn.compact
  def __call__(self,
               combine_method: EmbedCombineMethod = EmbedCombineMethod.SUM,
               *,
               segment_ids: Optional[Array] = None,
               decode: bool = False,
               deterministic: bool = False,
               **input_kwargs: Mapping[str, Array]) -> Array:
    """Embeds each input with its corresponding embedder and combines them.

    Args:
      combine_method: The method used for combination: sum or concat.
      segment_ids: Input segmentation info for packed examples.
      decode: Parameter to pass through to all embedders.
      deterministic: Parameter to pass through to all embedders.
      **input_kwargs: The input tensors to be embedded, with a name that matches
        the embedder in self.embedders, and each shaped: <int64>[..., seq_len].

    Returns:
      A tensor with the combined embeddings <float32>[..., embedding_size] in
      case of summing, and
      <float32>[..., embedding_size_1 + embedding_size_2 + ..] in case of
      concatenation.
    """
    y = self.combine_embeddings(
        self.get_individual_embeddings(
            segment_ids=segment_ids,
            decode=decode,
            deterministic=deterministic,
            **input_kwargs),
        combine_method=combine_method)

    # We sow the embedded (continuous) inputs and grads for feature attribution.
    if self.sow_intermediates:
      self.sow('intermediates', 'output', y)

    if not self.sow_intermediates and self.capture_gradients:
      raise ValueError('Must sow intermediates when capture_gradients is True.')
    # Capture the gradients by adding a zeros variable that will catch grads.
    # We only do this when `grads` is mutable because that prevents the grads
    # variable from being added for `Model.predict`.
    if (self.sow_intermediates and self.capture_gradients and
        self.scope.is_mutable_collection('grads')):
      eps = partitioning.variable_with_axes(
          'grads',
          'output_grad',
          lambda: jnp.zeros_like(y),
          axes=('batch', 'length', 'embed'))
      y = y + eps.value

    return y


class FixedEmbed(nn.Module, EmbedderWithDecode[Array]):
  """Fixed (not learnable) embeddings specified by the initializer function.

  Note: This embedding is not currently compatible with using prefixes when
  decoding because it assumes that the decoding loop starts at position 0.

  Attributes:
    init_fn: The initializer function that defines the embeddings.
    max_length: The maximum supported length.
    dtype: The DType to use for the embeddings.
  """
  features: int
  max_length: int = 2048
  embedding_init: Initializer = initializers.sinusoidal()
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    # The key is set to None because sinusoid init is deterministic.
    shape = (self.max_length, self.features)
    self.embedding = self.embedding_init(None, shape, self.dtype)  # pytype: disable=wrong-arg-types  # jax-ndarray

  @nn.compact
  def __call__(self,
               inputs,
               *,
               segment_ids: Optional[Array] = None,
               decode: bool = False,
               enable_dropout: bool = True):
    """Returns the fixed position embeddings specified by the initializer.

    Args:
      inputs: <int>[batch_size, seq_len] input position indices.
      segment_ids: Input segmentation info for packed examples.
      decode: True if running in single-position autoregressive decode mode.
      enable_dropout: Enables dropout if set to True.

    Returns:
      The fixed position embeddings <float32>[batch_size, seq_len, features].
    """
    del segment_ids  # Unused.

    # We use a cache position index for tracking decoding position.
    # TODO: Keep track of this index in the decoder instead.
    if decode:
      position_embedder_index = self.variable(
          'cache',
          'position_embedder_index',
          lambda: jnp.array([-1], dtype=jnp.uint32),
      )
      inputs = position_embedder_index.value[:, jnp.newaxis]
      position_embedder_index.value += 1

    return jnp.take(self.embedding, inputs, axis=0)


class PositionEmbed(nn.Module, EmbedderWithDecode[Array]):
  """Learned absolute positional embeddings for the inputs.

  Note: This embedding is not currently compatible with using prefixes when
  decoding because it assumes that the decoding loop starts at position 0.

  Attributes:
    num_embeddings: The maximum supported length. We learn this many positions.
    features: The number of features (size) for each position embedding.
    dtype: The DType to use for the position embeddings.
    embedding_init: Initialize the position embeddings with this function.
  """
  num_embeddings: int
  features: int
  dtype: DType = jnp.float32
  embedding_init: Initializer = default_embed_init  # pytype: disable=annotation-type-mismatch  # jax-types

  def setup(self):
    shape = (self.num_embeddings, self.features)
    self.pos_embedding = partitioning.param_with_axes(
        'pos_embedding',
        self.embedding_init,
        shape,
        jnp.float32,
        axes=('abspos_buckets', 'embed'))

  @nn.compact
  def __call__(self,
               inputs,
               *,
               segment_ids: Optional[Array] = None,
               decode: bool = False,
               enable_dropout: bool = True):
    """Applies PositionEmbed module.

    Args:
      inputs: <int>[batch_size, seq_len] input position indices.
      segment_ids: Input segmentation info for packed examples.
      decode: True if running in single-position autoregressive decode mode.
      enable_dropout: Enables dropout if set to True.

    Returns:
      The position embeddings <float32>[batch_size, seq_len, features].
    """
    del segment_ids  # Unused.

    # We use a cache position index for tracking decoding position.
    # TODO: Keep track of this index in the decoder instead.
    if decode:
      position_embedder_index = self.variable(
          'cache',
          'position_embedder_index',
          lambda: jnp.array([-1], dtype=jnp.uint32),
      )
      inputs = position_embedder_index.value[:, jnp.newaxis]
      position_embedder_index.value += 1

    return jnp.take(self.pos_embedding, inputs, axis=0)


def rotate_half(x):
  """Helper that splits a tensor at last dim into half and rotate it."""
  x1, x2 = jnp.split(x, 2, axis=-1)
  x = jnp.concatenate([-x2, x1], axis=-1)
  return x


@functools.partial(jax.jit, static_argnums=(4,))
def apply_rotary_embedding(q, k, cos, sin, decode=False, rotary_index=None):
  """Helper function to apply Rotary Embeddings."""
  if len(k.shape) == 3:
    # for multi query attention
    k = jnp.expand_dims(k, 2)
    multiquery = True
  else:
    multiquery = False

  batch, qlen, qheads, d = q.shape
  kbatch, klen, kheads, kd = k.shape
  assert batch == kbatch, f'{batch} != {kbatch}'
  assert d == kd, f'{d} != {kd}'

  # cos: [len, d]
  # sin: [len, d]
  # rotary_index: [batch]

  if decode and qlen == 1 and rotary_index is not None:
    # we check qlen == 1 so that we don't do this when initializing cache.
    qcos = cos[rotary_index, :]
    qsin = sin[rotary_index, :]
    # qcos, qsin: [batch, d]
    qcos = jax.lax.broadcast_in_dim(qcos, (batch, qlen, qheads, d), (0, 3))
    qsin = jax.lax.broadcast_in_dim(qsin, (batch, qlen, qheads, d), (0, 3))
    # qcos, qsin: [batch, qlen, qheads, d]
  else:
    qcos, qsin = cos[:qlen, :], sin[:qlen, :]
    # qcos, qsin: [qlen, d]
    qcos = jax.lax.broadcast_in_dim(qcos, (batch, qlen, qheads, d), (1, 3))
    qsin = jax.lax.broadcast_in_dim(qsin, (batch, qlen, qheads, d), (1, 3))
    # qcos, qsin: [batch, qlen, qheads, d]

  kcos, ksin = cos[:klen, :], sin[:klen, :]
  # kcos, ksin: [klen, d]
  kcos = jax.lax.broadcast_in_dim(kcos, (batch, klen, kheads, d), (1, 3))
  ksin = jax.lax.broadcast_in_dim(ksin, (batch, klen, kheads, d), (1, 3))
  # kcos, ksin: [batch, klen, kheads, d]

  out_q = (q * qcos) + (rotate_half(q) * qsin)
  out_k = (k * kcos) + (rotate_half(k) * ksin)
  if multiquery:
    out_k = jnp.squeeze(out_k, 2)
  return out_q, out_k


def generate_fixed_pos_embedding(features,
                                 length,
                                 min_timescale=1.0,
                                 max_timescale=10000.0):
  """Generate Sin/Cos for Rotary Embeddings.

  Generates sinusoids at (features//2) different timescales, where the
  timescales form a gemetric series from min_timescale to max_timescale
  (max_timescale is not included, but would be the next element in the series).

  Sinusoids are evaluated at integer positions i in [0, length).

  The outputs are computed as:

    output_sin[i, j] = sin(i / timescale[j])
    output_cos[i, j] = cos(i / timescale[j])

  Finally, the outputs are tiled twice in the features dimension.

  Args:
    features: an integer
    length: an integer
    min_timescale: an optional float
    max_timescale: an optional float

  Returns:
    output_sin: a float32 Tensor with shape [length, features]
    output_cos: a float32 Tensor with shape [length, features]
  """
  fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
  timescale = min_timescale * (max_timescale / min_timescale)**fraction
  rotational_frequency = 1. / timescale
  # Must use high precision einsum here, since rounding off to a bfloat16 is
  # catastrophic. bfloat16 rounds 257 to 256, but sin(257) is very different
  # from sin(256).
  sinusoid_inp = jnp.einsum(
      'i , j -> i j',
      jnp.arange(length),
      rotational_frequency,
      precision=jax.lax.Precision.HIGHEST)
  sinusoid_inp = jnp.concatenate([sinusoid_inp, sinusoid_inp], axis=-1)
  return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def _generate_primes(num_primes: int) -> Sequence[int]:
  result = []
  i = 2
  while len(result) < num_primes:
    if _is_prime(i):
      result.append(i)
    i += 1
  return result


def _is_prime(n: int) -> int:
  for i in range(2, int(math.sqrt(n)) + 1):
    if n % i == 0:  # A number >=2 evenly divides `n`, so it is not prime.
      return False
  return True


class HashEmbed(nn.Module, Embedder[Array]):
  """Embeds integer identifiers using multiple hashing.

  Each input identifier's embedding vector is the concatenation of multiple
  shards, with each shard coming from a separate embedding table. To reduce the
  effect of hash collisions, the identifier is retrieved from each different
  embedding table using a different hash function.

  Attributes:
    features: Dimensionality of final embedding.
    num_embeddings_per_table: Size ("vocabulary") of each embedding table.
    num_tables: Number of embedding tables (a.k.a. hash functions / shards).
    cast_input_dtype: DType to cast input to.
    dtype: DType of resulting embeddings.
    embedding_init: Initializer for embeddings.
    one_hot: Performs the gather with a one-hot contraction rather than a true
      gather. This is currently needed for SPMD partitioning.
  """
  features: int
  num_embeddings_per_table: int
  num_tables: int = 8
  cast_input_dtype: Optional[DType] = None
  dtype: DType = jnp.float32
  embedding_init: Initializer = default_embed_init  # pytype: disable=annotation-type-mismatch  # jax-types
  one_hot: bool = False

  _tables: Sequence[Embed] = dataclasses.field(init=False)
  _primes: Sequence[int] = dataclasses.field(init=False)

  def setup(self):
    if self.features % self.num_tables != 0:
      raise ValueError(f'Expected `features` ({self.features}) % '
                       f'`num_tables` ({self.num_tables}) == 0')

    if self.num_tables <= 8:
      # For compatibility with the public Canine checkpoints.
      self._primes = [31, 43, 59, 61, 73, 97, 103, 113][:self.num_tables]
    else:
      self._primes = _generate_primes(self.num_tables)
    shard_embedding_size = self.features // self.num_tables

    tables = []
    for i in range(self.num_tables):
      tables.append(
          Embed(
              name=f'hash_embedder_table_{i}',
              num_embeddings=self.num_embeddings_per_table,
              features=shard_embedding_size,
              dtype=self.dtype,
              embedding_init=self.embedding_init,
              one_hot=self.one_hot))
    self._tables = tables

  def __call__(self,
               input_ids: Array,
               *,
               segment_ids: Optional[Array] = None,
               decode: bool = False,
               enable_dropout: bool = True) -> Array:
    """Converts IDs into embeddings via multiple hashing.

    Args:
      input_ids: The IDs to be hashed. <int>[..., seq_length]
      segment_ids: Input segmentation info for packed examples.
      decode: True if running in single-position autoregressive decode mode.
      enable_dropout: Enables dropout if set to True.

    Returns:
      The emeddings (concatenated across hash shards).
      <float>[..., seq_length, features]
    """
    if self.cast_input_dtype:
      input_ids = input_ids.astype(self.cast_input_dtype)

    embedding_shards = []
    for table, prime in zip(self._tables, self._primes):
      # `hash_bucket_ids`: <int>[batch, seq]
      hash_bucket_ids = (((input_ids + 1) * prime) %
                         self.num_embeddings_per_table)
      # `shard_embeddings`: <float>[batch, seq, features/num_tables]
      shard_embeddings: Array = table(
          hash_bucket_ids,
          segment_ids=segment_ids,
          decode=decode,
          enable_dropout=enable_dropout)
      embedding_shards.append(shard_embeddings)
    # RESULT: <float>[batch, seq, features]
    return jnp.concatenate(embedding_shards, axis=-1)


class NgramHashEmbed(nn.Module, Embedder[Array]):
  """Produces embeddings for ngrams of identifiers.

  This is similar to `HashEmbed`, but instead of embedding just the input IDs,
  it embeds ngrams of those IDs.

  Attributes:
    ngram_orders: The sizes of ngrams to embed.
    padding_id: The ID to be used for padding the ends of the inputs.
    features: Dimensionality of final embedding.
    num_embeddings_per_table: Size ("vocabulary") of each embedding table.
    num_tables: Number of embedding tables (a.k.a. hash functions / shards).
    cast_input_dtype: DType to cast input to.
    dtype: DType of resulting embeddings.
    embedding_init: Initializer for embeddings.
    one_hot: Performs the gather with a one-hot contraction rather than a true
      gather. This is currently needed for SPMD partitioning.
  """
  ngram_orders: Sequence[int]
  padding_id: int
  features: int
  num_embeddings_per_table: int
  num_tables: int = 8
  cast_input_dtype: Optional[DType] = None
  dtype: DType = jnp.float32
  embedding_init: Initializer = default_embed_init  # pytype: disable=annotation-type-mismatch  # jax-types
  one_hot: bool = False

  _tables_by_order: Mapping[str, Sequence[Embed]] = (
      dataclasses.field(init=False))
  _primes_by_table: Sequence[int] = dataclasses.field(init=False)

  def setup(self):
    if self.features % self.num_tables != 0:
      raise ValueError(f'Expected `features` ({self.features}) % '
                       f'`num_tables` ({self.num_tables}) == 0')
    self._primes_by_table = _generate_primes(self.num_tables)
    shard_embedding_size = self.features // self.num_tables

    tables_by_order: DefaultDict[int, List[Embed]] = (
        collections.defaultdict(list))
    for order in self.ngram_orders:
      for i in range(self.num_tables):
        tables_by_order[order].append(
            Embed(
                name=f'{order}gram_hash_embed_table_{i}',
                num_embeddings=self.num_embeddings_per_table,
                features=shard_embedding_size,
                dtype=self.dtype,
                embedding_init=self.embedding_init,
                one_hot=self.one_hot))
    self._tables_by_order = {
        str(order): table for order, table in tables_by_order.items()
    }

  def __call__(self,
               input_ids: Array,
               *,
               segment_ids: Optional[Array] = None,
               decode: bool = False,
               enable_dropout: bool = True) -> Array:
    """Converts IDs to ngram embeddings via multiple hashing.

    This can run entirely on the TPU and so requires no modifications to the
    CPU-side input function. Rather than computing string-wise n-grams, this
    function approximates n-grams by using multiple hash functions over a window
    of character IDs.

    Args:
      input_ids: The IDs to be hashed. <int>[batch..., seq_length]
      segment_ids: Segment IDs for packed examples. <int>[batch..., seq_length]
      decode: True if running in single-position autoregressive decode mode.
      enable_dropout: Enables dropout if set to True.

    Returns:
      The emeddings. <float>[batch..., seq_length, features]
    """
    if self.cast_input_dtype:
      input_ids = input_ids.astype(self.cast_input_dtype)
      if segment_ids is not None:
        chex.assert_shape(segment_ids, input_ids.shape)
        segment_ids = segment_ids.astype(self.cast_input_dtype)

    if segment_ids is not None:
      # Create an array that, when multiplied by the input, will zero out the
      # final position of each segment.
      boundary_mask = segment_ids == self._shift_left(segment_ids)
      # Create an array that contains `self.padding_id` at every boundary
      # position, and zeros elsewhere.
      boundary_padding = jnp.logical_not(boundary_mask) * self.padding_id

    # Compute hash values for all orders of ngrams of `input_ids`, for each
    # embedding lookup table. Note that the initial (empty) value ensures that,
    # unigram hashes will be at index 1, bigrams at index 2, etc.
    hashes_by_table_by_order: List[List[Array]] = [[]]
    cur_ids = input_ids
    for order in range(1, max(self.ngram_orders) + 1):
      hashes_by_table = []
      for table_idx in range(self.num_tables):
        # Each `n`-gram's hash value is computed by "extending" the `n-1`-gram's
        # hash value with the `n`th ID and re-hashing.
        prev_hash = hashes_by_table_by_order[-1][table_idx] if order > 1 else 0
        prime: int = self._primes_by_table[table_idx]
        hashed: Array = (prev_hash + cur_ids) * prime
        hashes_by_table.append(hashed)
      hashes_by_table_by_order.append(hashes_by_table)
      cur_ids = self._shift_left(cur_ids)
      if segment_ids is not None:
        # Prevent leaking information across segments by zeroing out each
        # position that contains an ID that crossed from another segment, and
        # then replacing (only) those zeros with `self.padding_id`.
        cur_ids *= boundary_mask
        cur_ids += boundary_padding

    # Construct a mapping from ngram orders to lists of arrays, where each
    # <int>[batch..., seq_len] array contains the hashed ngram lookup keys for a
    # particular embedding table.
    hash_keys_by_table_by_order: Dict[int, List[Array]] = {}
    for order in self.ngram_orders:
      hash_keys_by_table_by_order[order] = [
          hashed % self.num_embeddings_per_table
          for hashed in hashes_by_table_by_order[order]
      ]

    # `ngram_embeddings`: A <float>[batch..., seq, dim] array for each order.
    ngram_embeddings: List[Array] = []
    for order in self.ngram_orders:
      tables: Sequence[Embed] = self._tables_by_order[str(order)]
      hash_keys_by_table: Sequence[Array] = hash_keys_by_table_by_order[order]
      embedding_shards: List[Array] = []
      for table, hash_keys in zip(tables, hash_keys_by_table):
        embedding_shards.append(
            table(
                hash_keys,
                segment_ids=segment_ids,
                decode=decode,
                enable_dropout=enable_dropout))
      ngram_embeddings.append(jnp.concatenate(embedding_shards, axis=-1))

    # TODO: Fancier aggregation function?
    result = sum(ngram_embeddings)
    chex.assert_shape(result, (*input_ids.shape, self.features))
    return result

  def _shift_left(self, ids: Array) -> Array:
    """Shifts `ids` left by one sequence position, padding the right."""
    sliced_ids = ids[..., 1:]
    batch_sizes = ids.shape[:-1]
    padding = jnp.expand_dims(jnp.tile(self.padding_id, batch_sizes), axis=-1)
    result = jnp.concatenate([sliced_ids, padding], axis=-1)
    chex.assert_shape(result, ids.shape)
    return result
