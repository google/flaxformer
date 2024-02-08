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

r"""SPMD model partitioning utilities for h-transformer."""
import abc
import dataclasses
import functools
from typing import Any, Dict, Tuple, Union

from flax.linen import spmd
from flaxformer.types import Array


# This adopts the standard_logical_axis_rules in t5x.partitioning.
# If a different set of rules is used, the names should be changed
# accordingly. A few new axis names specific to h-attention have
# been appended. Since no partitioning is expected along these
# axes, there is no need to add them to the logical_axis_rules. They
# will be assigned 'None' by default instead of 'data' or 'model'.
class AxisName:
  """All axis names supported in h-transformer."""

  # Standard Transformer axis names. These are supported by
  # T5x standard_logical_axis_rules and hence should work in
  # param_with_axes().
  BATCH: str = 'batch'
  VOCAB: str = 'vocab'
  EMBED: str = 'embed'
  MLP: str = 'mlp'
  HEADS: str = 'heads'
  KV: str = 'kv'
  JOINED_KV: str = 'joined_kv'
  LENGTH: str = 'length'
  RELPOS_BUCKETS: str = 'relpos_buckets'

  # For 2d images or video frames.
  HEIGHT: str = 'height'
  WIDTH: str = 'width'

  # The h-attention specific axis annotation.
  # These are NOT supported by T5x standard_logical_axis_rules.
  # So do not use them with param_with_axes() if T5x train loop is used.
  # The excerption is 1) Add them to T5x standard_logical_axis_rules;
  # 2) Use PAX train loop.
  PACKED_DIM: str = 'packed_dim'
  BLOCK: str = 'block'
  CLUSTER: str = 'cluster'
  ROW_CLUSTER: str = 'row_cluster'
  COL_CLUSTER: str = 'col_cluster'
  NEIGHBOR: str = 'neighbor'
  REL_POSITION: str = 'rel_position'
  UNMODELED: str = 'unmodeled'
  UNMODELED_HEADS: str = 'unmodeled_heads'
  UNMODELED_KV: str = 'unmodeled_kv'


@dataclasses.dataclass()
class PartitionerBase(metaclass=abc.ABCMeta):
  """Base class for partitioner."""

  layer_output_axis_names: Tuple[str, ...] = ()
  qkv_axis_names: Tuple[str, ...] = ()
  coarse_qkv_axis_names: Tuple[str, ...] = (
      AxisName.BATCH,
      AxisName.PACKED_DIM,
      AxisName.CLUSTER,
      AxisName.HEADS,
      AxisName.KV,
  )
  attention_axis_names: Tuple[str, ...] = (
      AxisName.BATCH,
      AxisName.PACKED_DIM,
      AxisName.ROW_CLUSTER,
      AxisName.COL_CLUSTER,
      AxisName.HEADS,
  )
  correction_mask_axis_names: Tuple[str, ...] = (
      AxisName.PACKED_DIM,
      AxisName.ROW_CLUSTER,
      AxisName.COL_CLUSTER,
      AxisName.HEADS,
  )
  causal_mask_axis_names: Tuple[str, ...] = (
      AxisName.ROW_CLUSTER,
      AxisName.COL_CLUSTER,
      AxisName.HEADS,
  )
  padding_mask_axis_names: Tuple[str, ...] = (
      AxisName.BATCH,
      AxisName.LENGTH,
      # Last dim is added embedding_dim=1. So no need to partition.
      AxisName.UNMODELED,
  )
  coarse_padding_mask_axis_names: Tuple[str, ...] = (
      AxisName.BATCH,
      AxisName.PACKED_DIM,
      AxisName.CLUSTER,
      # This is added heads=1. So no need to partition.
      AxisName.UNMODELED_HEADS,
      # This is added kv=1. So no need to partition.
      AxisName.UNMODELED_KV,
  )
  singlehead_rpb_axis_names: Tuple[str, ...] = (
      AxisName.BATCH,
      AxisName.PACKED_DIM,
      AxisName.ROW_CLUSTER,
      AxisName.COL_CLUSTER,
      # This is for heads=1. So no need to partition.
      AxisName.UNMODELED_HEADS,
  )

  def _generic_annotation_fn(
      self,
      y: Union[Array, Dict[Any, Array]],
      axis_names: Tuple[str, ...],
  ) -> Union[Array, Dict[Any, Array]]:
    """Generic axis annotation function."""
    annotation_fn = functools.partial(
        spmd.with_logical_constraint,
        logical_axis_resources=axis_names,
    )
    axis_count = len(axis_names)
    if isinstance(y, dict):
      annotation = {}
      for key, value in y.items():
        assert (
            axis_count == value.ndim
        ), f'Axis count {axis_count} does match {key} array ndim {value.ndim}.'
        annotation[key] = annotation_fn(value)
    else:
      assert (
          axis_count == y.ndim
      ), f'Axis count {axis_count} does match array ndim {y.ndim}.'
      annotation = annotation_fn(y)
    return annotation

  def annotate_layer_activation(
      self,
      y: Union[Array, Dict[Any, Array]],
  ) -> Union[Array, Dict[Any, Array]]:
    return self._generic_annotation_fn(y, self.layer_output_axis_names)

  def annotate_multihead_qkv(
      self,
      qkv: Union[Array, Dict[Any, Array]],
  ) -> Union[Array, Dict[Any, Array]]:
    return self._generic_annotation_fn(qkv, self.qkv_axis_names)

  def annotate_softmax_partition(
      self,
      partition: Union[Array, Dict[Any, Array]],
  ) -> Union[Array, Dict[Any, Array]]:
    return self.annotate_layer_activation(partition)

  def annotate_coarse_qkv(
      self,
      qkv: Union[Array, Dict[Any, Array]],
  ) -> Union[Array, Dict[Any, Array]]:
    return self._generic_annotation_fn(qkv, self.coarse_qkv_axis_names)

  def annotate_attention(
      self,
      attention: Union[Array, Dict[Any, Array]],
  ) -> Union[Array, Dict[Any, Array]]:
    return self._generic_annotation_fn(attention, self.attention_axis_names)

  # The similarity and attention have the same shape and axis names.
  # The same holds true for rpb and zero_block_mask.
  def annotate_similarity(
      self,
      similarity: Union[Array, Dict[Any, Array]],
  ) -> Union[Array, Dict[Any, Array]]:
    return self.annotate_attention(similarity)


@dataclasses.dataclass()
class Partitioner1D(PartitionerBase):
  """Partitoner for h-transformer-1d."""

  layer_output_axis_names: Tuple[str, ...] = (
      AxisName.BATCH,
      AxisName.LENGTH,
      AxisName.EMBED,
  )

  qkv_axis_names: Tuple[str, ...] = (
      AxisName.BATCH,
      AxisName.LENGTH,
      AxisName.HEADS,
      AxisName.KV,
  )


@dataclasses.dataclass()
class Partitioner2D(PartitionerBase):
  """Partitoner for h-transformer-2d."""

  layer_output_axis_names: Tuple[str, ...] = (
      AxisName.BATCH,
      AxisName.HEIGHT,
      AxisName.WIDTH,
      AxisName.EMBED,
  )

  qkv_axis_names: Tuple[str, ...] = (
      AxisName.BATCH,
      AxisName.HEIGHT,
      AxisName.WIDTH,
      AxisName.HEADS,
      AxisName.KV,
  )
