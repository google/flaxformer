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

"""Logit functions for dual encoder models.

We define a variety of logit  functions for dual encoder models. These
functions are similar to the similarity_functions, but are in this file because
they do not necessarily create logits for "similarity between 2 towers". Eg:
Creation of logits by applying fully connected layer on a single tower.


Pointwise logit_creation functions are computed just a single encoding and
return a vector of the same length as the batch size

  LC[i] = logit_creation(encodings1[i]).
"""

from typing import Optional
from flaxformer.architectures.dual_encoder import similarity_functions
from flaxformer.types import Array


# TODO: Factor out the FFNN bits into a new class and use that
# from `PointwiseFFNN` and `SingleTowerPointwiseFFNN`
class SingleTowerPointwiseFFNN(similarity_functions.PointwiseFFNN):
  """Single Tower Pointwise feed-forward NN logit creation function.

  Attributes:
  """
  tower_name: str = 'left'

  def __call__(self,
               encodings1: Optional[Array] = None,
               encodings2: Optional[Array] = None,
               *,
               enable_dropout: bool = True) -> Array:
    """Apply fully connected layer on either tower to change dim to num_classes.

    Use this class if instead of using a similarity function between the logits
    of 2 towers, you want to apply a Fully Connected NN on top of an individual
    tower to create logits. Eg: Classification task on an individual tower.

    Args:
      encodings1: A 2-D tensor of (left) encodings with shape [batch size,
        encoding dim].
      encodings2: A 2-D tensor of (right) encodings with shape [batch size,
        encoding dim].
      enable_dropout: Whether to enable dropout layers.

    Returns:
      A 1-D tensor of logits with shape [B,num_classes].
    """
    if self.tower_name == 'left':
      return super().__call__(encodings1, None, enable_dropout=enable_dropout)
    else:
      return super().__call__(encodings2, None, enable_dropout=enable_dropout)
