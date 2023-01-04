# Copyright 2023 Google LLC.
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

"""Sets of utilities used across the core components of LightweightMMM."""

import sys
from typing import Any, Mapping, Tuple, Union

import jax.numpy as jnp

from numpyro import distributions as dist

#  pylint: disable=g-import-not-at-top
if sys.version_info >= (3, 8):
  from typing import Protocol
else:
  from typing_extensions import Protocol


class TransformFunction(Protocol):

  def __call__(
      self,
      data: jnp.ndarray,
      custom_priors: Mapping[str, dist.Distribution],
      prefix: str,
      **kwargs: Any,
  ) -> jnp.ndarray:
    ...


class Module(Protocol):

  def __call__(
      self,
      *args: Any,
      **kwargs: Any,
  ) -> jnp.ndarray:
    ...


def get_number_geos(data: jnp.ndarray) -> int:
  return data.shape[2] if data.ndim == 3 else 1


def get_geo_shape(data: jnp.ndarray) -> Union[Tuple[int], Tuple[()]]:
  return (data.shape[2],) if data.ndim == 3 else ()


def apply_exponent_safe(data: jnp.ndarray,
                        exponent: jnp.ndarray) -> jnp.ndarray:
  """Applies an exponent to given data in a gradient safe way.

  More info on the double jnp.where can be found:
  https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf

  Args:
    data: Input data to use.
    exponent: Exponent required for the operations.

  Returns:
    The result of the exponent operation with the inputs provided.
  """
  exponent_safe = jnp.where(condition=(data == 0), x=1, y=data)**exponent
  return jnp.where(condition=(data == 0), x=0, y=exponent_safe)
