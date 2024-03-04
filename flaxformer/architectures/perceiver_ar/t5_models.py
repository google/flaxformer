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

"""This file contains "model" classes for T5 models."""

import enum
import functools
from typing import Any, Callable, Mapping, MutableMapping, Optional, Tuple

from absl import logging
import flax
from flax import linen as nn
from flax import traverse_util
from flax.training import common_utils
import jax
from jax import lax
import jax.numpy as jnp
import seqio
from t5x import decoding
from t5x import losses
from t5x import models
from t5x import optimizers
from flaxformer.architectures.perceiver_ar import slicing


PyTree = Any


def _crop_sequences(sequences: jnp.ndarray,
                    lengths: jnp.ndarray) -> jnp.ndarray:
  """Crop sequences by replacing positions beyond length with padding."""
  return jnp.where(
      jnp.arange(sequences.shape[-1])[jnp.newaxis, :] < lengths[:, jnp.newaxis],
      sequences, 0)


class CroppingMethod(enum.Enum):
  """Perceiver AR training cropping methods.

  NONE: Cropping will be done in the data pipeline, so no online cropping
    is needed.

  FULL_LATENTS: Random placement of latents between the beginning and end
    of the sequence where loss is calculated. As many latents as possible are
    allocated positions.
    Advantage: Loss over as many tokens as possible, better use of compute.
    Disadvantage: May bias against learning to generate positions toward the
      beginning or end of sequences because they will be selected less
      frequently. For prefix tasks, does not match latent positions at inference
      time.

  FULL_LATENTS_WITH_PREFIX: Same as FULL_LATENTS, but allows the beginning
    of the window to a prefix where loss is not calculated, up to the point
    where only 1 position has loss. This matches inference behavior for a prefix
    task because (depending on the decoding_latent_reset_fill setting) the first
    inferred position can utilize all previous latents allocated to the prefix.
    Advantage: Loss over as many tokens as possible while still matching
      inference latent placement.
    Disadvantage: Prefix positions do not have loss calculated, so there are
      fewer positions with loss than with FULL_LATENTS. Also still has
      some of the bias issues fixed with EQUAL_POSITION_LIKELIHOOD.

  EQUAL_POSITION_LIKELIHOOD: Random placement of latents such that every
    sequence position within the loss mask is equally likely to have loss
    calculated on it. Achieved by letting the latent "window" extend beyond the
    edges of the sequence and then cropping/masking any invalid positions.
    Advantage: Every position is equally likely to be trained.
    Disadvantage: Loss over fewer positions, wasted compute. For example, with
      a sequence length of 8192 and 2048 latent positions, each training batch
      will be only 80% non-padding tokens.
  """
  NONE = 1
  FULL_LATENTS = 2
  FULL_LATENTS_WITH_PREFIX = 3
  EQUAL_POSITION_LIKELIHOOD = 4


def crop_train_batch(
    rng: Optional[jax.Array],
    batch: Mapping[str, jnp.ndarray],
    cropping_method: CroppingMethod,
    num_latents: int,
) -> Mapping[str, jnp.ndarray]:
  """Apply random cropping to a training batch.

  Perceiver AR can utilize a longer input sequence than the number of latents
  and therefore outputs positions for loss. In order to train the model to be
  able to generate outputs with a variety of input context lengths, random
  cropping of the input sequence is used.

  Args:
    rng: PRNG key.
    batch: T5X training batch.
    cropping_method: Type of cropping method to use.
    num_latents: Number of latents in the Perceiver AR model.

  Returns:
    A cropped batch.
  """
  first_loss_idx = jnp.argmax(batch['decoder_loss_weights'] == 1, axis=-1)
  last_loss_idx = batch['decoder_loss_weights'].shape[-1] - 1 - jnp.argmax(
      jnp.flip(batch['decoder_loss_weights'] == 1, axis=-1), axis=-1)

  logging.info('Using cropping method "%s".', cropping_method)
  if cropping_method == CroppingMethod.NONE:
    return batch
  if cropping_method == CroppingMethod.FULL_LATENTS:
    # "naive" crop selection. always results in a full batch.
    min_crop_start = first_loss_idx
    max_crop_start = last_loss_idx - num_latents + 1
  elif cropping_method == CroppingMethod.FULL_LATENTS_WITH_PREFIX:
    # FULL_LATENTS, but allows including all but 1 latent in the prefix portion.
    min_crop_start = first_loss_idx - num_latents + 1
    min_crop_start = jnp.maximum(min_crop_start, 0)
    max_crop_start = last_loss_idx - num_latents + 1
  elif cropping_method == CroppingMethod.EQUAL_POSITION_LIKELIHOOD:
    # "fair" crop selection. all positions equally likely.
    min_crop_start = first_loss_idx - num_latents + 1
    max_crop_start = last_loss_idx
  else:
    raise ValueError(f'Unknown cropping method: {cropping_method}')

  seq_crop_first_idx = jax.random.randint(
      rng, [batch['decoder_loss_weights'].shape[0]], min_crop_start,
      max_crop_start + 1)

  seq_crop_end = jnp.minimum(seq_crop_first_idx + num_latents,
                             last_loss_idx + 1)
  seq_crop_start = jnp.maximum(seq_crop_first_idx, 0)

  batch = jax.tree.map(
      functools.partial(_crop_sequences, lengths=seq_crop_end), batch)

  # Handle the loss weights specifically to ensure that loss isn't
  # calculated for positions before seq_crop_start. This ensures that all
  # token positions have an equal likelihood of being counted in the loss.
  # Specifically, it handles cases where the crop over a sequence of length
  # 8192 is something like [8000:8192]. Even if there are 2048 latents
  # allocated to [6144:8192], loss is only calculated on [8000:8192].
  batch['decoder_loss_weights'] = jnp.where(
      jnp.arange(batch['decoder_loss_weights'].shape[-1])[jnp.newaxis, :] >=
      seq_crop_start[:, jnp.newaxis], batch['decoder_loss_weights'], 0)

  return batch


class PerceiverARModel(models.DecoderOnlyModel):
  """Model class for Perceiver AR decoder-only model.

  Implements Perceiver AR as described in https://arxiv.org/abs/2202.07765.

  Decouples input length from most of the compute requirements by utilizing
  an initial cross-attention layer over the inputs to a smaller number of
  latents for processing with the self-attention stack.
  """

  def __init__(
      self,
      module: nn.Module,
      vocabulary: seqio.Vocabulary,
      optimizer_def: optimizers.OptimizerDefType,
      num_latents: int,
      decoding_latent_reset_fill: Optional[int] = None,
      disable_fast_decoding_cache: bool = False,
      decode_fn: models.DecodeFnCallable = decoding.temperature_sample,
      inputs_bidirectional_attention: bool = False,
      feature_converter_cls: Optional[Callable[...,
                                               seqio.FeatureConverter]] = None,
      label_smoothing: float = 0.0,
      z_loss: float = 0.0,
      loss_normalizing_factor: Optional[float] = None,
      train_cropping_method: CroppingMethod = CroppingMethod.FULL_LATENTS,
  ):
    self._num_latents = num_latents
    self._disable_fast_decoding_cache = disable_fast_decoding_cache

    self._configured_decoding_latent_reset_fill = decoding_latent_reset_fill

    self._train_cropping_method = train_cropping_method
    super().__init__(
        module=module,
        vocabulary=vocabulary,
        optimizer_def=optimizer_def,
        decode_fn=decode_fn,
        inputs_bidirectional_attention=inputs_bidirectional_attention,
        feature_converter_cls=feature_converter_cls,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        loss_normalizing_factor=loss_normalizing_factor,
    )

  def get_decoding_latent_reset_fill(self, input_length: int) -> int:
    if self._configured_decoding_latent_reset_fill is not None:
      decoding_latent_reset_fill = self._configured_decoding_latent_reset_fill
    else:
      # If not specified, use some reasonable defaults that try to pick a good
      # balance between using as many latents as possible (more "compute" per
      # predicted token) and doing as few reset steps as possible (full forward
      # passes that are more expensive).
      # For large numbers of latents, fill all but the final 128 positions.
      # Example: 2048 latents, 1920 reset fill.
      # For small numbers of latents, just do half.
      decoding_latent_reset_fill = max(self._num_latents - 128,
                                       self._num_latents // 2, 1)

    # For shorter sequences, make sure we use the largest fill possible.
    # For example, if there are 2048 latents, the default reset fill from above
    # would be 1920. But if the sequence length is 2049, then we'll have to do
    # 1 reset step, so we might as well use the full 2048 latents and get the
    # maximum "compute" available.
    decoding_latent_reset_fill = max(
        decoding_latent_reset_fill,
        self._num_latents - max(0, input_length - self._num_latents - 1))

    if decoding_latent_reset_fill <= 0:
      raise ValueError(f'decoding_latent_reset_fill must be > 0, but got '
                       f'{decoding_latent_reset_fill}')

    if decoding_latent_reset_fill > self._num_latents:
      raise ValueError(
          f'decoding_latent_reset_fill must be <= num_latents '
          f'({self._num_latents}), but got {decoding_latent_reset_fill}')

    logging.info(
        'decoding_latent_reset_fill: for configured fill %r, num_latents %d, '
        'and input length %d, using fill of %d.',
        self._configured_decoding_latent_reset_fill, self._num_latents,
        input_length, decoding_latent_reset_fill)
    return decoding_latent_reset_fill

  def eval_fn(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
  ) -> Tuple[jnp.ndarray, models.MetricsMap]:
    """Computes loss and metrics during the evaluation.

    Args:
      params: model parameters.
      batch: a batch of inputs.

    Returns:
      loss: the loss computed for the given inputs and parameters.
      aux:
        weight_sum: sum of the per-token weights applied to the loss.
        metrics: a mapping of metrics computed for this batch.
    """
    return self.loss_fn(
        params=params,
        batch=batch,
        dropout_rng=None,
        is_eval=True,
    )

  def loss_fn(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.Array],
      is_eval: bool = False,
  ) -> Tuple[jnp.ndarray, models.MetricsMap]:
    """Loss function used for training with a cross-entropy loss."""
    if dropout_rng is None:
      # TODO: Add RNG ability to T5X during eval.
      # TODO: In addition to random crops during eval, perhaps also take
      # decoding_latent_reset_fill into account and only report eval loss on
      # the final positions since this will more closely match what will
      # happen during inference and scoring.
      if is_eval:
        logging.info(
            'Eval loss_fn: no RNG key present, so cropping method of "%s" '
            'will not occur.', self._train_cropping_method)
      else:
        raise ValueError('Required dropout_rng was not supplied.')
    else:
      crop_train_rng, dropout_rng = jax.random.split(dropout_rng)
      batch = crop_train_batch(
          crop_train_rng,
          batch=batch,
          cropping_method=self._train_cropping_method,
          num_latents=self._num_latents)

    logits = self._compute_logits(params, batch, dropout_rng)

    sequence_lengths = slicing.get_sequence_lengths(
        batch['decoder_target_tokens'])
    assert self._num_latents == logits.shape[-2]

    targets = slicing.slice_sequences_vmap(
        batch['decoder_target_tokens'],
        sequence_lengths=sequence_lengths,
        num_latents=self._num_latents,
        axis_within_vmap=-1)
    weights = slicing.slice_sequences_vmap(
        batch['decoder_loss_weights'],
        sequence_lengths=sequence_lengths,
        num_latents=self._num_latents,
        axis_within_vmap=-1)

    loss_normalizing_factor, weights = losses.get_loss_normalizing_factor_and_weights(
        self._loss_normalizing_factor,
        batch={
            'decoder_target_tokens': targets,
            'decoder_loss_weights': weights
        })

    loss, z_loss, _ = losses.compute_weighted_cross_entropy(
        logits,
        targets=targets,
        weights=weights,
        label_smoothing=self._label_smoothing,
        z_loss=self._z_loss,
        loss_normalizing_factor=loss_normalizing_factor)
    metrics = self._compute_metrics(
        logits=logits, targets=targets, mask=weights, loss=loss, z_loss=z_loss)
    return loss, metrics

  def _compute_logits_from_slice(
      self,
      decoding_state: decoding.DecodingState,
      params: PyTree,
      decoder_causal_attention: jnp.ndarray,
      max_decode_length: int,
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Token slice to logits from decoder model."""
    # Implement a cache reset step as described in Appendix E.3 of the Perceiver
    # AR paper (https://arxiv.org/pdf/2202.07765.pdf)

    decoding_latent_reset_fill = self.get_decoding_latent_reset_fill(
        decoding_state.sequences.shape[1])

    def get_cache_by_layers(cache):
      return traverse_util.flatten_dict(
          cache, is_leaf=lambda k, x: 'cache_index' in x)

    def tree_map_self_att_cache(map_fn, cache):
      """Map a function over just the self-attention cache layers."""
      cache_by_layers = get_cache_by_layers(cache)
      new_cache_by_layers = {}
      for layer_name, layer_cache in cache_by_layers.items():
        # Only modify params that have 'layer' in the name to avoid things like
        # position encodings.
        # The first layer is cross-attention, so don't modify it.
        if 'layer' in '/'.join(layer_name) and 'layers_0' not in layer_name:
          layer_cache = jax.tree.map(map_fn, layer_cache)
        new_cache_by_layers[layer_name] = layer_cache
      return flax.core.freeze(traverse_util.unflatten_dict(new_cache_by_layers))

    def reset_step():
      assert self._num_latents >= decoding_latent_reset_fill

      # Create a version of the kv cache that has
      # decoding_latent_reset_fill positions instead of self._num_latents
      # positions.
      def prepare_reset_cache(x):
        # Modify key and value, but not index.
        if x.ndim > 1 and x.shape[-1] == self._num_latents:
          return x[..., :decoding_latent_reset_fill] * 0
        else:
          return x

      reset_cache = tree_map_self_att_cache(prepare_reset_cache,
                                            decoding_state.cache)

      # Note that it's possible to reuse the cached activations for the
      # cross-attention layer, but that would be fairly difficult to do with
      # the current cache API.

      # To ensure masking is calculated correctly, construct target_ids by
      # shifting inputs left and adding a placeholder value for the current
      # position.
      target_ids = jnp.pad(decoding_state.sequences[:, 1:], [[0, 0], [0, 1]])
      target_ids = jax.vmap(lambda x, y: x.at[y].set(1))(
          target_ids, decoding_state.cur_index)

      # Do a full forward pass of the model to predict the next tokens, filling
      # in the partial cache with the smaller number of latents as wel do.
      logits, new_vars = self.module.apply(
          {
              'params': params,
              'cache': reset_cache,
          },
          decoder_input_tokens=decoding_state.sequences,
          decoder_target_tokens=target_ids,
          enable_dropout=False,
          decoder_causal_attention=decoder_causal_attention,
          decode=False,
          max_decode_length=max_decode_length,
          prefill=True,
          prefill_lengths=decoding_state.cur_index + 1,
          mutable=['cache'],
          num_latents=decoding_latent_reset_fill)

      # Now expand the kv cache size back to self._num_latents.
      def expand_reset_cache(x):
        # Modify key and value, but not index.
        if x.ndim > 1 and x.shape[-1] == decoding_latent_reset_fill:
          padding = [(0, 0)] * x.ndim
          padding[-1] = (0, self._num_latents - decoding_latent_reset_fill)
          return jnp.pad(x, padding)
        else:
          return x

      new_cache = tree_map_self_att_cache(expand_reset_cache, new_vars['cache'])

      logits_idx = jnp.minimum(logits.shape[-2] - 1, decoding_state.cur_index)
      flat_logits = jax.vmap(
          functools.partial(lax.dynamic_slice_in_dim, slice_size=1,
                            axis=-2))(logits, logits_idx)
      return flat_logits, new_cache

    def regular_step():
      flat_logits, new_vars = self.module.apply(
          {
              'params': params,
              'cache': decoding_state.cache
          },
          decoding_state.cur_token,
          decoding_state.cur_token,
          enable_dropout=False,
          decode=True,
          max_decode_length=max_decode_length,
          mutable=['cache'])
      return flat_logits, new_vars['cache']

    # Determine if a reset step is needed based on whether the kv cache in
    # a self-attention layer is full.
    needs_reset = False
    for layer_name, layer_cache in get_cache_by_layers(
        decoding_state.cache).items():
      # Ignore non-layer parameters like position encodings.
      if 'layer' not in '/'.join(layer_name):
        continue
      # Ignore the cross-attention layer since it never gets "full".
      if 'layers_0' in layer_name:
        continue
      needs_reset |= (layer_cache['cache_index'] >=
                      layer_cache['cached_key'].shape[-1]).any()

    if self._disable_fast_decoding_cache:
      logging.info(
          'Fast decoding is disabled, always using reset steps with a latent'
          'fill of %d positions', decoding_latent_reset_fill)
      flat_logits, new_flat_cache = reset_step()
    elif decoding_state.sequences.shape[-1] > self._num_latents:
      logging.info('Using a reset step latent fill of %d positions',
                   decoding_latent_reset_fill)

      flat_logits, new_flat_cache = lax.cond(needs_reset, reset_step,
                                             regular_step)
    elif decoding_state.sequences.shape[-1] == self._num_latents:
      # If num_latents is the same as sequence length, there's no need to
      # use or compile reset_setp.
      logging.info('Using regular decoding without any reset steps.')
      flat_logits, new_flat_cache = regular_step()
    else:
      raise ValueError(
          f'Sequence length ({decoding_state.sequences.shape[-1]}) < '
          f'num_latents ({self._num_latents}) is not currently supported.')

    # Remove sequence length dimension since it's always 1 during decoding.
    flat_logits = jnp.squeeze(flat_logits, axis=1)

    return flat_logits, new_flat_cache

  def predict_batch_with_aux(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      rng: Optional[jax.Array] = None,
      *,
      return_all_decodes: bool = False,
      num_decodes: int = 1,
      decoder_params: Optional[MutableMapping[str, Any]] = None,
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Predict with prefix.

    Mostly copied from DecoderOnlyModel with minor modifications for preparing
    the tokens_ids_to_logits function.

    `decoder_params` can be used to pass dynamic configurations to
    `self.decode_fn`. An example usage is to pass different random seed (i.e.,
    `jax.random.PRNGKey(seed)` with different `seed` value). This can be done by
    setting `decoder_params['decode_rng'] = jax.random.PRNGKey(seed)`.

    Although this method is short, there are a few subtle points that. We use a
    running example to make these points clear.

    ```
    Example
      inputs = [9, 4, 6, 1]
      targets = [3, 9, 1]

      seqio.DecoderFeatureConverter will generate these set of features

         decoder_target_tokens = [9, 4, 6, 1, 3, 9, 1, 0, 0]
          decoder_input_tokens = [0, 9, 4, 6, 1, 3, 9, 1, 0]
      decoder_causal_attention = [1, 1, 1, 1, 1, 0, 0, 0, 0]

      The output of this function is (`a` through `e` are the sampled token
      ids):

             sampled_sequences = [9, 4, 6, 1, a, b, c, d, e].
    ```

    Given these set of features, we make a few important observation.

    1) When a decoder-only model is used for a supervised learning with "inputs"
       and "targets", one way to handle this is to concatenate the "inputs" and
       "targets". For training, we use teacher forcing for the entire
       concatenated sequence. For inference, on the other hand, we don't have
       the targets. This requires that we use teacher forcing on the "inputs"
       portion while using the generated token as the input token for the next
       decoding step. For evaluation, we do have "targets" but we only want to
       use them for computing metrics, i.e., by comparing to the sequence
       generated by the model.

       This function is currently used for evaluation mode, but by ignoring
       "targets", it can be extended for the inference mode.

    2) During evaluation mode, the targets portion is zeroed out and they are
       filled with the sampled token ids. The inputs portion is kept intact.

    3) Note that `decoder_causal_attention` has an additional 1 after the final
       "inputs" token. This is because the position where the last "inputs"
       token (in this case 1) is input and the output is the first "target"
       token (in this case 3) can be included in the non-causal attention
       region.

       This results in an alignment between `decoder_input_tokens` and
       `decoder_causal_attention` because the former is shifted to the right by
       one position. So we use `decoder_causal_attention` as a binary mask to
       zero out the target tokens in `decoder_input_tokens`.

    Note:
      In order to use a custom self._decode_fn with this model it must support:

      1) Decoding from a partially decoded state by accepting a vector of
         `initial_indices` that specify where in the input to start decoding
         from.
      2) Using a vector as the loop counter to support different examples being
         a different number of steps into their decoding loop.
      3) Be able to handle one batch element reaching `max_decode_length`
         before the others without it causing the model to prematurely stop
         decoding.

    Args:
      params: Model parameters.
      batch: Batch element with the model features specified in
        seqio.DecoderFeatureConverter.
      rng: An optional RNG key to use during prediction, which is passed as
        'decode_rng' to the decoding function.
      return_all_decodes: If True, will return all batch_size * num_decodes
        samples from the model as an array of shape [batch_size, num_decodes,
        sequence_length]. Otherwise returns only the most likely samples as an
        array of shape [batch_size, sequence_length].
      num_decodes: Number of decoded sequences to be returned.
      decoder_params: Additional (model-independent) parameters for the decoder.

    Returns:
      Sampled sequences, an array of shape [batch, max_decode_length].
    """
    if 'decoder_causal_attention' not in batch:
      raise ValueError(
          'Batch does not have the right format for text generation: probably '
          'because `task_feature_lengths` passed to the feature converter does '
          'not have both `inputs` and `targets`.'
      )
    # since decoder_input_tokens is shifted to the right and
    # `decoder_causal_attention` has one more 1 than the number of inputs
    # tokens, this masks out targets portion of the decoder_input_tokens.
    inputs = batch['decoder_input_tokens'] * batch['decoder_causal_attention']

    # TODO: Minor decoding performance improvement: Ideally
    # _compute_kv_cache would prefill the cache with enough space left over to
    # not immediately trigger a cache reset step if the sequence length is
    # already longer than self._num_latents.

    prefilled_cache, initial_index = self._compute_kv_cache(
        params, inputs, batch['decoder_causal_attention']
    )

    target_shape = batch['decoder_input_tokens'].shape
    max_decode_length = target_shape[1]

    # Note that the version of decoder_causal_attention to be passed to the
    # model during inference needs to be calculated by
    # _get_decoder_causal_attention, which will correctly set it to None if
    # inputs_bidirectional_attention is False.
    tokens_ids_to_logits = functools.partial(
        self._compute_logits_from_slice,
        params=params,
        decoder_causal_attention=self._get_decoder_causal_attention(batch),
        max_decode_length=max_decode_length)

    if decoder_params is None:
      decoder_params = {}
    if rng is not None:
      if decoder_params.get('decode_rng') is not None:
        raise ValueError(
            f'Got RNG both from the `rng` argument ({rng}) and '
            f"`decoder_params['decode_rng']` ({decoder_params['decode_rng']}). "
            'Please specify one or the other.')
      decoder_params['decode_rng'] = rng

    # Using the above-defined single-step decoder function, run temperature
    # sampling with the prefix.
    # [batch, max_decode_length]
    scanned = hasattr(self.module, 'scan_layers') and self.module.scan_layers

    if 'eos_id' not in decoder_params:
      decoder_params['eos_id'] = self.output_vocabulary.eos_id
    decoded_sequences, scores = self._decode_fn(
        inputs=inputs,
        cache=prefilled_cache,
        tokens_to_logits=tokens_ids_to_logits,
        num_decodes=num_decodes,
        initial_index=initial_index,
        cache_offset=1 if scanned else 0,
        **decoder_params,
    )

    if not return_all_decodes:
      # Search returns [n_batch, n_beam/decodes, n_length] with the beam/decode
      # dimension sorted in increasing order of log-probability.
      # `scores` is [batch, beam/decode_size]
      # We take the highest scoring sequence (-1) and its score
      decoded_sequences = decoded_sequences[:, -1, :]
      # Beam search returns []
      aux = {'scores': scores[:, -1]}
    else:
      # We return all samples and scores, rather than just the top ones.
      aux = {'scores': scores}

    return models.remove_prefix(decoded_sequences, initial_index), aux

  def score_batch(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      return_intermediates: bool = False,
  ) -> jnp.ndarray:
    """Compute log likelihood score on a batch.

    Perceiver AR will return only num_latents outputs for a given forward pass,
    but we want scores for all inputs positions. This method does multiple
    forward passes, striding over the input as determined by
    decoding_latent_reset_fill. The results are combined into a single logits
    array and summed for the final score.

    Args:
      params: Model params.
      batch: Batch to score.
      return_intermediates: Whether to return model intermediates. Not currently
        supported for Perceiver AR.

    Returns:
      Sequence scores with shape [batch].
    """
    if return_intermediates:
      raise NotImplementedError('return_intermediates is not yet supported.')

    decoder_target_tokens = batch['decoder_target_tokens']
    weights = batch['decoder_loss_weights']
    input_length = decoder_target_tokens.shape[-1]
    sequence_lengths = slicing.get_sequence_lengths(
        decoder_target_tokens=decoder_target_tokens)

    def get_token_scores(logits):
      return -losses.cross_entropy_with_logits(
          logits,
          common_utils.onehot(
              decoder_target_tokens, logits.shape[-1], on_value=1, off_value=0),
          z_loss=0.0)[0] * weights

    # Calculate stride given decoding_latent_reset_fill.
    # For example, if decoding_latent_reset_fill=num_latents, then in decoding
    # we would use only the final latent position to predict the next token.
    # The equivalent behavior here is a stride of 1.
    decoding_latent_reset_fill = self.get_decoding_latent_reset_fill(
        input_length)
    stride = self._num_latents - decoding_latent_reset_fill + 1
    logging.info(
        'decoding_latent_reset_fill is %d and num_latents is %d, so using a '
        'stride of %d for scoring.', decoding_latent_reset_fill,
        self._num_latents, stride)

    # Loop forward using strides.
    def body(state):
      slice_end = jnp.maximum(state['slice_end'] + stride, self._num_latents)
      slice_end = jnp.minimum(slice_end, sequence_lengths)

      loop_batch = jax.tree.map(
          functools.partial(_crop_sequences, lengths=slice_end), batch)
      loop_logits = self._compute_logits(
          params=params, batch=loop_batch, dropout_rng=None)
      loop_logits = jnp.pad(loop_logits, [(0, 0),
                                          (0, input_length - self._num_latents),
                                          (0, 0)])
      loop_logits_shift = jnp.maximum(slice_end - self._num_latents, 0)
      loop_logits = jax.vmap(functools.partial(jnp.roll,
                                               axis=0))(loop_logits,
                                                        loop_logits_shift)

      if 'logits' not in state:
        # Should happen only during the initialization pass so we can get the
        # dtype and vocabulary dimension from the actual model outputs.
        # During the lax.while_loop, we can't modify this.
        state['logits'] = jnp.zeros_like(loop_logits)

      logits = jnp.where(
          jnp.arange(input_length)[jnp.newaxis, :, jnp.newaxis] >=
          state['slice_end'][:, jnp.newaxis, jnp.newaxis], loop_logits,
          state['logits'])

      new_state = {
          'logits': logits,
          'slice_end': slice_end,
      }
      return new_state

    def cond(state):
      done = state['slice_end'] >= sequence_lengths
      return jnp.any(~done)

    # Start where loss starts to be calculated.
    slice_end = jnp.argmax(weights > 0, axis=1)
    init_state = {
        'slice_end': slice_end,
    }

    # Run the first iteration outside the while_loop to initialize state dict
    # with logits that match the shape/dtype of the actual model outputs.
    init_state = body(init_state)

    final_state = lax.while_loop(
        cond_fun=cond, body_fun=body, init_val=init_state)
    logits = final_state['logits']

    token_scores = get_token_scores(logits)
    sequence_scores = token_scores.sum(-1)

    return sequence_scores
