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

"""Core and modelling functions for trend."""

import functools
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


@functools.partial(jax.jit, static_argnames=("number_periods",))
def _dynamic_trend(
    number_periods: int,
    random_walk_level: jnp.ndarray,
    random_walk_slope: jnp.ndarray,
    initial_level: jnp.ndarray,
    initial_slope: jnp.ndarray,
    variance_level: jnp.ndarray,
    variance_slope: jnp.ndarray,
) -> jnp.ndarray:
  """Calculates dynamic trend using local linear trend method.

  More details about this function can be found in:
  https://storage.googleapis.com/pub-tools-public-publication-data/pdf/41854.pdf

  Args:
    number_periods: Number of time periods in the data.
    random_walk_level: Random walk of level from sample.
    random_walk_slope: Random walk of slope from sample.
    initial_level: The initial value for level in local linear trend model.
    initial_slope: The initial value for slope in local linear trend model.
    variance_level: The variance of the expected increase in level between time.
    variance_slope: The variance of the expected increase in slope between time.

  Returns:
    The dynamic trend values for the given data with the given parameters.
  """
  # Simulate gaussian random walk of level with initial level.
  random_level = variance_level * random_walk_level
  random_level_with_initial_level = jnp.concatenate(
      [jnp.array([random_level[0] + initial_level]), random_level[1:]])
  level_trend_t = jnp.cumsum(random_level_with_initial_level, axis=0)
  # Simulate gaussian random walk of slope with initial slope.
  random_slope = variance_slope * random_walk_slope
  random_slope_with_initial_slope = jnp.concatenate(
      [jnp.array([random_slope[0] + initial_slope]), random_slope[1:]])
  slope_trend_t = jnp.cumsum(random_slope_with_initial_slope, axis=0)
  # Accumulate sum of slope series to address latent variable slope in function
  # level_t = level_t-1 + slope_t-1.
  initial_zero_shape = [(1, 0)] if slope_trend_t.ndim == 1 else [(1, 0), (0, 0)]
  slope_trend_cumsum = jnp.pad(
      jnp.cumsum(slope_trend_t, axis=0)[:number_periods - 1],
      initial_zero_shape, mode="constant", constant_values=0)
  return level_trend_t + slope_trend_cumsum


def dynamic_trend(
    geo_size: int,
    data_size: int,
    is_trend_prediction: bool,
    custom_priors: Mapping[str, dist.Distribution],
) -> jnp.ndarray:
  """Generates the dynamic trend to capture the baseline of kpi.

  Args:
    geo_size: Number of geos in the model.
    data_size: Number of time samples in the model.
    is_trend_prediction: Whether it is used for prediction or fitting.
    custom_priors: The custom priors we want the model to take instead of the
      default ones. See our custom_priors documentation for details about the
      API and possible options.

  Returns:
    Jax array with trend for each time t.
  """
  default_priors = priors.get_default_priors()
  if not is_trend_prediction:
    random_walk_level = numpyro.sample("random_walk_level",
                                       fn=dist.Normal(),
                                       sample_shape=(data_size, 1))
    random_walk_slope = numpyro.sample("random_walk_slope",
                                       fn=dist.Normal(),
                                       sample_shape=(data_size, 1))
  else:
    random_walk_level = numpyro.sample("random_walk_level_prediction",
                                       fn=dist.Normal(),
                                       sample_shape=(data_size, 1))

    random_walk_slope = numpyro.sample("random_walk_slope_prediction",
                                       fn=dist.Normal(),
                                       sample_shape=(data_size, 1))

  with numpyro.plate(
      name=f"{priors.DYNAMIC_TREND_INITIAL_LEVEL}_plate", size=geo_size):
    trend_initial_level = numpyro.sample(
        name=priors.DYNAMIC_TREND_INITIAL_LEVEL,
        fn=custom_priors.get(
            priors.DYNAMIC_TREND_INITIAL_LEVEL,
            default_priors[priors.DYNAMIC_TREND_INITIAL_LEVEL]))

  with numpyro.plate(
      name=f"{priors.DYNAMIC_TREND_INITIAL_SLOPE}_plate", size=geo_size):
    trend_initial_slope = numpyro.sample(
        name=priors.DYNAMIC_TREND_INITIAL_SLOPE,
        fn=custom_priors.get(
            priors.DYNAMIC_TREND_INITIAL_SLOPE,
            default_priors[priors.DYNAMIC_TREND_INITIAL_SLOPE]))

  with numpyro.plate(
      name=f"{priors.DYNAMIC_TREND_LEVEL_VARIANCE}_plate", size=geo_size):
    trend_level_variance = numpyro.sample(
        name=priors.DYNAMIC_TREND_LEVEL_VARIANCE,
        fn=custom_priors.get(
            priors.DYNAMIC_TREND_LEVEL_VARIANCE,
            default_priors[priors.DYNAMIC_TREND_LEVEL_VARIANCE]))

  with numpyro.plate(
      name=f"{priors.DYNAMIC_TREND_SLOPE_VARIANCE}_plate", size=geo_size):
    trend_slope_variance = numpyro.sample(
        name=priors.DYNAMIC_TREND_SLOPE_VARIANCE,
        fn=custom_priors.get(
            priors.DYNAMIC_TREND_SLOPE_VARIANCE,
            default_priors[priors.DYNAMIC_TREND_SLOPE_VARIANCE]))

  if geo_size == 1:  # National level model case.
    random_walk_level = jnp.squeeze(random_walk_level)
    random_walk_slope = jnp.squeeze(random_walk_slope)
    trend_initial_level = jnp.squeeze(trend_initial_level)
    trend_initial_slope = jnp.squeeze(trend_initial_slope)
    trend_level_variance = jnp.squeeze(trend_level_variance)
    trend_slope_variance = jnp.squeeze(trend_slope_variance)

  return _dynamic_trend(
      number_periods=data_size,
      random_walk_level=random_walk_level,
      random_walk_slope=random_walk_slope,
      initial_level=trend_initial_level,
      initial_slope=trend_initial_slope,
      variance_level=trend_level_variance,
      variance_slope=trend_slope_variance)
