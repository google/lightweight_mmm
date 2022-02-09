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

"""Module containing the different models available in the lightweightMMM lib.

Currently this file contains a main model with three possible options for
processing the media data. Which essentially grants the possibility of building
three different models.
  - Adstock
  - Hill-Adstock
  - Carryover
"""

from typing import Any, Callable, Mapping, Optional

import frozendict
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

from lightweight_mmm.lightweight_mmm import media_transforms


def transform_adstock(media_data: jnp.ndarray,
                      normalise: bool = True) -> jnp.ndarray:
  """Transforms the input data with the adstock function and exponent.

  Args:
    media_data: Media data to be transformed.
    normalise: Whether to normalise the output values.

  Returns:
    The transformed media data.
  """
  with numpyro.plate("lag_weight_plate", media_data.shape[1]):
    lag_weight = numpyro.sample("lag_weight",
                                dist.Beta(concentration1=2., concentration0=1.))
  with numpyro.plate("exponent_plate", media_data.shape[1]):
    exponent = numpyro.sample("exponent",
                              dist.Beta(concentration1=9., concentration0=1.))
  adstock = media_transforms.adstock(
      data=media_data, lag_weight=lag_weight, normalise=normalise)

  return media_transforms.apply_exponent_safe(data=adstock, exponent=exponent)


def transform_hill_adstock(media_data: jnp.ndarray,
                           normalise: bool = True) -> jnp.ndarray:
  """Transforms the input data with the adstock and hill functions.

  Args:
    media_data: Media data to be transformed.
    normalise: Whether to normalise the output values.

  Returns:
    The transformed media data.
  """
  with numpyro.plate("lag_weight_plate", media_data.shape[1]):
    lag_weight = numpyro.sample("lag_weight",
                                dist.Beta(concentration1=2., concentration0=1.))

  with numpyro.plate("half_max_effective_concentration_plate",
                     media_data.shape[1]):
    half_max_effective_concentration = numpyro.sample(
        "half_max_effective_concentration",
        dist.Gamma(concentration=1., rate=1.))

  with numpyro.plate("slope_plate", media_data.shape[1]):
    slope = numpyro.sample("slope", dist.Gamma(concentration=1., rate=1.))

  return media_transforms.hill(
      data=media_transforms.adstock(
          data=media_data, lag_weight=lag_weight, normalise=normalise),
      half_max_effective_concentration=half_max_effective_concentration,
      slope=slope)


def transform_carryover(media_data: jnp.ndarray,
                        number_lags: int = 13) -> jnp.ndarray:
  """Transforms the input data with the carryover function and exponent.

  Args:
    media_data: Media data to be transformed.
    number_lags: Number of lags for the carryover function.

  Returns:
    The transformed media data.
  """
  with numpyro.plate("ad_effect_retention_rate_plate", media_data.shape[1]):
    ad_effect_retention_rate = numpyro.sample(
        "ad_effect_retention_rate",
        dist.Beta(concentration1=1., concentration0=1.))

  with numpyro.plate("peak_effect_delay_plate", media_data.shape[1]):
    peak_effect_delay = numpyro.sample("peak_effect_delay",
                                       dist.HalfNormal(scale=2.))
  with numpyro.plate("exponent_plate", media_data.shape[1]):
    exponent = numpyro.sample("exponent",
                              dist.Beta(concentration1=9., concentration0=1.))

  carryover = media_transforms.carryover(
      data=media_data,
      ad_effect_retention_rate=ad_effect_retention_rate,
      peak_effect_delay=peak_effect_delay,
      number_lags=number_lags)

  return media_transforms.apply_exponent_safe(data=carryover, exponent=exponent)


def media_mix_model(media_data: jnp.ndarray,
                    target_data: jnp.ndarray,
                    cost_prior: jnp.ndarray,
                    degrees_seasonality: int,
                    frequency: int,
                    transform_function: Callable[[jnp.array], jnp.array],
                    transform_kwargs: Mapping[str,
                                              Any] = frozendict.frozendict(),
                    extra_features: Optional[jnp.array] = None) -> None:
  """Media mix model.

  Args:
    media_data: Media data to be be used in the model.
    target_data: Target data for the model.
    cost_prior: Cost prior for each of the media channels.
    degrees_seasonality: Number of degrees of seasonality to use.
    frequency: Frequency of the time spam which was used to aggregate the data.
      Eg. if weekly data then frequency is 52.
    transform_function: Function to use to transform the media data in the
      model. Currently the following are supported: 'transform_adstock',
        'transform_carryover' and 'transform_hill_adstock'.
    transform_kwargs: Any extra keyword arguments to pass to the transform
      function. For example the adstock function can take a boolean to noramlise
      output or not.
    extra_features: Extra features data to include in the model.
  """
  data_size = media_data.shape[0]
  intercept = numpyro.sample("intercept", dist.Normal(loc=0., scale=2.))
  sigma = numpyro.sample("sigma", dist.Gamma(concentration=1., rate=1.))
  beta_trend = numpyro.sample("beta_trend", dist.Normal(loc=0., scale=1.))
  expo_trend = numpyro.sample("expo_trend",
                              dist.Beta(concentration1=1., concentration0=1.))

  with numpyro.plate("media_plate", media_data.shape[1]) as i:
    beta_media = numpyro.sample("beta_media",
                                dist.HalfNormal(scale=cost_prior[i]))

  with numpyro.plate("gamma_seasonality_plate", 2):
    with numpyro.plate("seasonality_plate", degrees_seasonality):
      gamma_seasonality = numpyro.sample("gamma_seasonality",
                                         dist.Normal(loc=0., scale=1.))

  media_transformed = numpyro.deterministic(
      name="media_transformed",
      value=transform_function(media_data, **transform_kwargs))
  seasonality = media_transforms.calculate_seasonality(
      number_periods=data_size,
      degrees=degrees_seasonality,
      frequency=frequency,
      gamma_seasonality=gamma_seasonality)

  prediction = (
      intercept + beta_trend * jnp.arange(data_size) ** (expo_trend + 0.5) +
      seasonality + media_transformed.dot(beta_media))
  if extra_features is not None:
    with numpyro.plate("extra_features_plate", extra_features.shape[1]):
      beta_extra_features = numpyro.sample("beta_extra_features",
                                           dist.Normal(loc=0., scale=1.))
    prediction += extra_features.dot(beta_extra_features)
  mu = numpyro.deterministic(name="mu", value=prediction)

  numpyro.sample(
      name="target", fn=dist.Normal(loc=mu, scale=sigma), obs=target_data)
