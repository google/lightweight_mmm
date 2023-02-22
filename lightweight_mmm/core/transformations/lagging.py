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

"""Set of core and modelling lagging functions."""

import functools
from typing import Mapping, Union

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from lightweight_mmm.core import priors


@functools.partial(jax.vmap, in_axes=(1, 1, None), out_axes=1)
def _carryover_convolve(data: jnp.ndarray, weights: jnp.ndarray,
                        number_lags: int) -> jnp.ndarray:
  """Applies the convolution between the data and the weights for the carryover.

  Args:
    data: Input data.
    weights: Window weights for the carryover.
    number_lags: Number of lags the window has.

  Returns:
    The result values from convolving the data and the weights with padding.
  """
  window = jnp.concatenate([jnp.zeros(number_lags - 1), weights])
  return jax.scipy.signal.convolve(data, window, mode="same") / weights.sum()


@functools.partial(jax.jit, static_argnames=("number_lags",))
def _carryover(
    data: jnp.ndarray,
    ad_effect_retention_rate: jnp.ndarray,
    peak_effect_delay: jnp.ndarray,
    number_lags: int,
) -> jnp.ndarray:
  """Calculates media carryover.

  More details about this function can be found in:
  https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46001.pdf

  Args:
    data: Input data. It is expected that data has either 2 dimensions for
      national models and 3 for geo models.
    ad_effect_retention_rate: Retention rate of the advertisement effect.
      Default is 0.5.
    peak_effect_delay: Delay of the peak effect in the carryover function.
      Default is 1.
    number_lags: Number of lags to include in the carryover calculation. Default
      is 13.

  Returns:
    The carryover values for the given data with the given parameters.
  """
  lags_arange = jnp.expand_dims(
      jnp.arange(number_lags, dtype=jnp.float32), axis=-1)
  convolve_func = _carryover_convolve
  if data.ndim == 3:
    # Since _carryover_convolve is already vmaped in the decorator we only need
    # to vmap it once here to handle the geo level data. We keep the windows bi
    # dimensional also for three dims data and vmap over only the extra data
    # dimension.
    convolve_func = jax.vmap(
        fun=_carryover_convolve, in_axes=(2, None, None), out_axes=2)
  weights = ad_effect_retention_rate**((lags_arange - peak_effect_delay)**2)
  return convolve_func(data, weights, number_lags)


def carryover(
    data: jnp.ndarray,
    custom_priors: Mapping[str, dist.Distribution],
    *,
    number_lags: int = 13,
    prefix: str = "",
) -> jnp.ndarray:
  """Transforms the input data with the carryover function.

  Args:
    data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
    custom_priors: The custom priors we want the model to take instead of the
      default ones.
    number_lags: Number of lags for the carryover function.
    prefix: Prefix to use in the variable name for Numpyro.

  Returns:
    The transformed media data.
  """
  default_priors = priors.get_default_priors()
  with numpyro.plate(
      name=f"{prefix}{priors.AD_EFFECT_RETENTION_RATE}_plate",
      size=data.shape[1]):
    ad_effect_retention_rate = numpyro.sample(
        name=f"{prefix}{priors.AD_EFFECT_RETENTION_RATE}",
        fn=custom_priors.get(priors.AD_EFFECT_RETENTION_RATE,
                             default_priors[priors.AD_EFFECT_RETENTION_RATE]))

  with numpyro.plate(
      name=f"{prefix}{priors.PEAK_EFFECT_DELAY}_plate", size=data.shape[1]):
    peak_effect_delay = numpyro.sample(
        name=f"{prefix}{priors.PEAK_EFFECT_DELAY}",
        fn=custom_priors.get(priors.PEAK_EFFECT_DELAY,
                             default_priors[priors.PEAK_EFFECT_DELAY]))

  return _carryover(
      data=data,
      ad_effect_retention_rate=ad_effect_retention_rate,
      peak_effect_delay=peak_effect_delay,
      number_lags=number_lags)


@jax.jit
def _adstock(
    data: jnp.ndarray,
    lag_weight: Union[float, jnp.ndarray] = .9,
    normalise: bool = True,
) -> jnp.ndarray:
  """Calculates the adstock value of a given array.

  To learn more about advertising lag:
  https://en.wikipedia.org/wiki/Advertising_adstock

  Args:
    data: Input array.
    lag_weight: lag_weight effect of the adstock function. Default is 0.9.
    normalise: Whether to normalise the output value. This normalization will
      divide the output values by (1 / (1 - lag_weight)).

  Returns:
    The adstock output of the input array.
  """

  def adstock_internal(
      prev_adstock: jnp.ndarray,
      data: jnp.ndarray,
      lag_weight: Union[float, jnp.ndarray] = lag_weight,
  ) -> jnp.ndarray:
    adstock_value = prev_adstock * lag_weight + data
    return adstock_value, adstock_value# jax-ndarray

  _, adstock_values = jax.lax.scan(
      f=adstock_internal, init=data[0, ...], xs=data[1:, ...])
  adstock_values = jnp.concatenate([jnp.array([data[0, ...]]), adstock_values])
  return jax.lax.cond(
      normalise,
      lambda adstock_values: adstock_values / (1. / (1 - lag_weight)),
      lambda adstock_values: adstock_values,
      operand=adstock_values)


def adstock(
    data: jnp.ndarray,
    custom_priors: Mapping[str, dist.Distribution],
    *,
    normalise: bool = True,
    prefix: str = "",
) -> jnp.ndarray:
  """Transforms the input data with the adstock function and exponent.

  Args:
    data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
    custom_priors: The custom priors we want the model to take instead of the
      default ones. The possible names of parameters for adstock and exponent
      are "lag_weight" and "exponent".
    normalise: Whether to normalise the output values.
    prefix: Prefix to use in the variable name for Numpyro.

  Returns:
    The transformed media data.
  """
  default_priors = priors.get_default_priors()
  with numpyro.plate(
      name=f"{prefix}{priors.LAG_WEIGHT}_plate", size=data.shape[1]):
    lag_weight = numpyro.sample(
        name=f"{prefix}{priors.LAG_WEIGHT}",
        fn=custom_priors.get(priors.LAG_WEIGHT,
                             default_priors[priors.LAG_WEIGHT]))

  if data.ndim == 3:
    lag_weight = jnp.expand_dims(lag_weight, axis=-1)

  return _adstock(data=data, lag_weight=lag_weight, normalise=normalise)
