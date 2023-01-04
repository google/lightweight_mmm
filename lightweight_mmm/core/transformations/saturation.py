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

"""Set of core and modelling saturation functions."""

from typing import Mapping
import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

from lightweight_mmm.core import core_utils
from lightweight_mmm.core import priors


@jax.jit
def _hill(
    data: jnp.ndarray,
    half_max_effective_concentration: jnp.ndarray,
    slope: jnp.ndarray,
) -> jnp.ndarray:
  """Calculates the hill function for a given array of values.

  Refer to the following link for detailed information on this equation:
    https://en.wikipedia.org/wiki/Hill_equation_(biochemistry)

  Args:
    data: Input data.
    half_max_effective_concentration: ec50 value for the hill function.
    slope: Slope of the hill function.

  Returns:
    The hill values for the respective input data.
  """
  save_transform = core_utils.apply_exponent_safe(
      data=data / half_max_effective_concentration, exponent=-slope)
  return jnp.where(save_transform == 0, x=0, y=1. / (1 + save_transform))


def hill(
    data: jnp.ndarray,
    custom_priors: Mapping[str, dist.Distribution],
    *,
    prefix: str = "",
) -> jnp.ndarray:
  """Transforms the input data with the adstock and hill functions.

  Args:
    data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
    custom_priors: The custom priors we want the model to take instead of the
      default ones. The possible names of parameters for hill_adstock and
      exponent are "lag_weight", "half_max_effective_concentration" and "slope".
    prefix: Prefix to use in the variable name for Numpyro.

  Returns:
    The transformed media data.
  """
  default_priors = priors.get_default_priors()

  with numpyro.plate(
      name=f"{prefix}{priors.HALF_MAX_EFFECTIVE_CONCENTRATION}_plate",
      size=data.shape[1]):
    half_max_effective_concentration = numpyro.sample(
        name=f"{prefix}{priors.HALF_MAX_EFFECTIVE_CONCENTRATION}",
        fn=custom_priors.get(
            priors.HALF_MAX_EFFECTIVE_CONCENTRATION,
            default_priors[priors.HALF_MAX_EFFECTIVE_CONCENTRATION]))

  with numpyro.plate(name=f"{prefix}{priors.SLOPE}_plate", size=data.shape[1]):
    slope = numpyro.sample(
        name=f"{prefix}{priors.SLOPE}",
        fn=custom_priors.get(priors.SLOPE, default_priors[priors.SLOPE]))

  if data.ndim == 3:
    half_max_effective_concentration = jnp.expand_dims(
        half_max_effective_concentration, axis=-1)
    slope = jnp.expand_dims(slope, axis=-1)

  return _hill(
      data=data,
      half_max_effective_concentration=half_max_effective_concentration,
      slope=slope)


def _exponent(data: jnp.ndarray, exponent_values: jnp.ndarray) -> jnp.ndarray:
  """Applies exponent to the given data."""
  return core_utils.apply_exponent_safe(data=data, exponent=exponent_values)


def exponent(
    data: jnp.ndarray,
    custom_priors: Mapping[str, dist.Distribution],
    *,
    prefix: str = "",
) -> jnp.ndarray:
  """Transforms the input data with the carryover function and exponent.

  Args:
    data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
    custom_priors: The custom priors we want the model to take instead of the
      default ones.
    prefix: Prefix to use in the variable name for Numpyro.

  Returns:
    The transformed media data.
  """
  default_priors = priors.get_default_priors()

  with numpyro.plate(
      name=f"{prefix}{priors.EXPONENT}_plate", size=data.shape[1]):
    exponent_values = numpyro.sample(
        name=f"{prefix}{priors.EXPONENT}",
        fn=custom_priors.get(priors.EXPONENT, default_priors[priors.EXPONENT]))

  if data.ndim == 3:
    exponent_values = jnp.expand_dims(exponent_values, axis=-1)
  return _exponent(data=data, exponent_values=exponent_values)
