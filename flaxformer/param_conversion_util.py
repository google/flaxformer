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

"""Common parameter conversion utilities."""

import collections
import re
from typing import Any, Dict, List, Mapping, Tuple

import jax.numpy as jnp
import tensorflow as tf


def load_tf_params(checkpoint_path: str) -> Dict[str, Any]:
  """Loads TF parameters from the checkpoint at the given path."""
  ckpt_reader = tf.train.load_checkpoint(checkpoint_path)
  return {
      tf_name: ckpt_reader.get_tensor(tf_name)
      for tf_name in ckpt_reader.get_variable_to_dtype_map()
  }


def convert_tf_params(
    tf_params: Dict[str, Any],
    mapping: Mapping[Tuple[str, ...], str],
    tf_prefix: str = '',
) -> Dict[str, Any]:
  """Given a mapping from TF to Flax names returns a Flax parameters dict.

  A utility method that given a TF parameters dictionary, a mapping between the
  Flax parameter of depth one (i.e only two keys in the returned dictionary) and
  a tf_prefix returns a partial dictionary of the Flax parameters. The method is
  used in the rest of the functions to create a final full dictionary of Flax
  parameters.

  Args:
    tf_params: TF BERT model parameters extracted from the checkpoint.
    mapping: A parameters mapping between a slice of Flax BERT model dictionary
      of parameters and the TF parameters.
    tf_prefix: A shared prefix of the TF parameter keys. Will be prepended to
      each of TF parameter key in the mapping.

  Returns:
    Partial parameters for the Flax BERT model.
  """
  params = collections.defaultdict(dict)
  for jax_key, tf_var in mapping.items():
    tf_name = f'{tf_prefix}{tf_var}'
    if tf_name in tf_params:
      tf_value = tf_params[tf_name]
    elif f'{tf_name}:0' in tf_params:
      tf_value = tf_params[f'{tf_name}:0']
    else:
      raise ValueError(f'tf_params does not contain {tf_name!r}')

    param_dict = params
    for jax_key_part in jax_key[:-1]:
      param_dict = param_dict[jax_key_part]
    param_dict[jax_key[-1]] = jnp.asarray(tf_value)
  return params


def get_int_regex_matches(pattern: str, state: Mapping[str, Any]) -> List[int]:
  """Matches a pattern with an integer capture group against state keys."""
  matches = [re.match(pattern, key) for key in state]
  return sorted(set(int(m.group(1)) for m in matches if m is not None))
