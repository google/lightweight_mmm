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

"""Core and modelling functions for seasonality."""

from typing import Mapping

import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

from lightweight_mmm.core import priors
from lightweight_mmm.core import core_utils


@jax.jit
def _sinusoidal_seasonality(
    seasonality_arange: jnp.ndarray,
    degrees_arange: jnp.ndarray,
    gamma_seasonality: jnp.ndarray,
    frequency: int,
) -> jnp.ndarray:
  """Core calculation of cyclic variation seasonality.

  Args:
    seasonality_arange: Array with range [0, N - 1] where N is the size of the
      data for which the seasonality is modelled.
    degrees_arange: Array with range [0, D - 1] where D is the number of degrees
      to use. Must be greater or equal than 1.
    gamma_seasonality: Factor to multiply to each degree calculation. Shape must
      be aligned with the number of degrees.
    frequency: Frecuency of the seasonality be in computed.

  Returns:
    An array with the seasonality values.
  """
  inner_value = seasonality_arange * 2 * jnp.pi * degrees_arange / frequency
  season_matrix_sin = jnp.sin(inner_value)
  season_matrix_cos = jnp.cos(inner_value)
  season_matrix = jnp.concatenate([
      jnp.expand_dims(a=season_matrix_sin, axis=-1),
      jnp.expand_dims(a=season_matrix_cos, axis=-1)
  ],
                                  axis=-1)
  return jnp.einsum("tds, ds -> t", season_matrix, gamma_seasonality)


def sinusoidal_seasonality(
    data: jnp.ndarray,
    custom_priors: Mapping[str, dist.Distribution],
    *,
    degrees_seasonality: int = 2,
    frequency: int = 52,
) -> jnp.ndarray:
  """Calculates cyclic variation seasonality.

  For detailed info check:
    https://en.wikipedia.org/wiki/Seasonality#Modeling

  Args:
    data: Data for which the seasonality will be modelled for. It is used to
      obtain the length of the time dimension, axis 0.
    custom_priors: The custom priors we want the model to take instead of
      default ones.
    degrees_seasonality: Number of degrees to use. Must be greater or equal than
      1.
    frequency: Frecuency of the seasonality be in computed. By default is 52 for
      weekly data (52 weeks in a year).

  Returns:
    An array with the seasonality values.
  """
  number_periods = data.shape[0]
  default_priors = priors.get_default_priors()
  n_geos = core_utils.get_number_geos(data=data)
  with numpyro.plate(name=f"{priors.GAMMA_SEASONALITY}_sin_cos_plate", size=2):
    with numpyro.plate(
        name=f"{priors.GAMMA_SEASONALITY}_plate", size=degrees_seasonality):
      gamma_seasonality = numpyro.sample(
          name=priors.GAMMA_SEASONALITY,
          fn=custom_priors.get(priors.GAMMA_SEASONALITY,
                               default_priors[priors.GAMMA_SEASONALITY]))
  seasonality_arange = jnp.expand_dims(a=jnp.arange(number_periods), axis=-1)
  degrees_arange = jnp.arange(degrees_seasonality)
  seasonality_values = _sinusoidal_seasonality(
      seasonality_arange=seasonality_arange,
      degrees_arange=degrees_arange,
      frequency=frequency,
      gamma_seasonality=gamma_seasonality,
  )
  if n_geos > 1:
    seasonality_values = jnp.expand_dims(seasonality_values, axis=-1)
  return seasonality_values


def _intra_week_seasonality(
    data: jnp.ndarray,
    weekday: jnp.ndarray,
) -> jnp.ndarray:
  data_size = data.shape[0]
  return weekday[jnp.arange(data_size) % 7]


def intra_week_seasonality(
    data: jnp.ndarray,
    custom_priors: Mapping[str, dist.Distribution],
) -> jnp.ndarray:
  """Models intra week seasonality.

  Args:
    data: Data for which the seasonality will be modelled for. It is used to
      obtain the length of the time dimension, axis 0.
    custom_priors: The custom priors we want the model to take instead of
      default ones.

  Returns:
    The contribution of the weekday seasonality.
  """
  default_priors = priors.get_default_priors()
  with numpyro.plate(name=f"{priors.WEEKDAY}_plate", size=7):
    weekday = numpyro.sample(
        name=priors.WEEKDAY,
        fn=custom_priors.get(priors.WEEKDAY, default_priors[priors.WEEKDAY]))

  weekday_series = _intra_week_seasonality(data=data, weekday=weekday)

  if data.ndim == 3:  # For geo model's case
    weekday_series = jnp.expand_dims(weekday_series, axis=-1)

  return weekday_series
