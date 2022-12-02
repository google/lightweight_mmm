# Copyright 2022 Google LLC.
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

"""Core and modelling functions for trend."""

from typing import Mapping

import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

from lightweight_mmm.core import core_utils
from lightweight_mmm.core import priors


@jax.jit
def _trend_with_exponent(coef_trend: jnp.ndarray, trend: jnp.ndarray,
                         expo_trend: jnp.ndarray) -> jnp.ndarray:
  """Applies the coefficient and exponent to the trend to obtain trend values.

  Args:
    coef_trend: Coefficient to be multiplied by the trend.
    trend: Initial trend values.
    expo_trend: Exponent to be applied to the trend.

  Returns:
    The trend values generated.
  """
  return coef_trend * trend**expo_trend


def trend_with_exponent(
    data: jnp.ndarray,
    custom_priors: Mapping[str, dist.Distribution],
) -> jnp.ndarray:
  """Trend with exponent for curvature.

  Args:
    data: Data for which trend will be created.
    custom_priors: The custom priors we want the model to take instead of the
      default ones. See our custom_priors documentation for details about the
      API and possible options.

  Returns:
    The values of the trend.
  """
  default_priors = priors.get_default_priors()
  n_geos = core_utils.get_number_geos(data=data)
  # TODO(): Force all geos to have the same trend sign.
  with numpyro.plate(name=f"{priors.COEF_TREND}_plate", size=n_geos):
    coef_trend = numpyro.sample(
        name=priors.COEF_TREND,
        fn=custom_priors.get(priors.COEF_TREND,
                             default_priors[priors.COEF_TREND]))

  expo_trend = numpyro.sample(
      name=priors.EXPO_TREND,
      fn=custom_priors.get(priors.EXPO_TREND,
                           default_priors[priors.EXPO_TREND]))
  linear_trend = jnp.arange(data.shape[0])
  if n_geos > 1:  # For geo model's case
    linear_trend = jnp.expand_dims(linear_trend, axis=-1)
  return _trend_with_exponent(
      coef_trend=coef_trend, trend=linear_trend, expo_trend=expo_trend)
