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

from lightweight_mmm import media_transforms


def transform_adstock(media_data: jnp.ndarray,
                      normalise: bool = True) -> jnp.ndarray:
  """Transforms the input data with the adstock function and exponent.

  Args:
    media_data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
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
  if media_data.ndim == 3:
    lag_weight = jnp.expand_dims(lag_weight, axis=-1)
    exponent = jnp.expand_dims(exponent, axis=-1)

  adstock = media_transforms.adstock(
      data=media_data, lag_weight=lag_weight, normalise=normalise)

  return media_transforms.apply_exponent_safe(data=adstock, exponent=exponent)


def transform_hill_adstock(media_data: jnp.ndarray,
                           normalise: bool = True) -> jnp.ndarray:
  """Transforms the input data with the adstock and hill functions.

  Args:
    media_data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
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

  if media_data.ndim == 3:
    lag_weight = jnp.expand_dims(lag_weight, axis=-1)
    half_max_effective_concentration = jnp.expand_dims(
        half_max_effective_concentration, axis=-1)
    slope = jnp.expand_dims(slope, axis=-1)

  return media_transforms.hill(
      data=media_transforms.adstock(
          data=media_data, lag_weight=lag_weight, normalise=normalise),
      half_max_effective_concentration=half_max_effective_concentration,
      slope=slope)


def transform_carryover(media_data: jnp.ndarray,
                        number_lags: int = 13) -> jnp.ndarray:
  """Transforms the input data with the carryover function and exponent.

  Args:
    media_data: Media data to be transformed. It is expected to have 2 dims for
      national models and 3 for geo models.
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

  if media_data.ndim == 3:
    exponent = jnp.expand_dims(exponent, axis=-1)
  return media_transforms.apply_exponent_safe(data=carryover, exponent=exponent)


def media_mix_model(
    media_data: jnp.ndarray,
    target_data: jnp.ndarray,
    cost_prior: jnp.ndarray,
    degrees_seasonality: int,
    frequency: int,
    transform_function: Callable[[jnp.array], jnp.array],
    transform_kwargs: Mapping[str, Any] = frozendict.frozendict(),
    weekday_seasonality: bool = False,
    extra_features: Optional[jnp.array] = None,
    ) -> None:
  """Media mix model.

  Args:
    media_data: Media data to be be used in the model.
    target_data: Target data for the model.
    cost_prior: Cost prior for each of the media channels.
    degrees_seasonality: Number of degrees of seasonality to use.
    frequency: Frequency of the time span which was used to aggregate the data.
      Eg. if weekly data then frequency is 52.
    transform_function: Function to use to transform the media data in the
      model. Currently the following are supported: 'transform_adstock',
        'transform_carryover' and 'transform_hill_adstock'.
    transform_kwargs: Any extra keyword arguments to pass to the transform
      function. For example the adstock function can take a boolean to noramlise
      output or not.
    weekday_seasonality: In case of daily data you can estimate a weekday (7)
      parameter.
    extra_features: Extra features data to include in the model.
  """
  data_size = media_data.shape[0]
  n_channels = media_data.shape[1]
  geo_shape = (media_data.shape[2],) if media_data.ndim == 3 else ()
  n_geos = media_data.shape[2] if media_data.ndim == 3 else 1

  with numpyro.plate(name="intercept_plate", size=n_geos):
    intercept = numpyro.sample(
        name="intercept",
        fn=dist.Normal(loc=0., scale=2.))

  with numpyro.plate(name="sigma_plate", size=n_geos):
    sigma = numpyro.sample(
        name="sigma",
        fn=dist.Gamma(concentration=1., rate=1.))

  # TODO(): Force all geos to have the same trend sign.
  with numpyro.plate(name="beta_trend_plate", size=n_geos):
    beta_trend = numpyro.sample(
        name="beta_trend",
        fn=dist.Normal(loc=0., scale=1.))

  expo_trend = numpyro.sample(
      name="expo_trend",
      fn=dist.Beta(concentration1=1., concentration0=1.))

  with numpyro.plate(
      name="channel_media_plate",
      size=n_channels,
      dim=-2 if media_data.ndim == 3 else -1):
    beta_media = numpyro.sample(
        name="channel_beta_media" if media_data.ndim == 3 else "beta_media",
        fn=dist.HalfNormal(scale=cost_prior))
    if media_data.ndim == 3:
      with numpyro.plate(
          name="geo_media_plate",
          size=n_geos,
          dim=-1):
        beta_media = numpyro.sample(
            name="beta_media", fn=dist.HalfNormal(scale=beta_media))
  with numpyro.plate(name="gamma_seasonality_sin_cos_plate", size=2):
    with numpyro.plate(name="gamma_seasonality_plate",
                       size=degrees_seasonality):
      gamma_seasonality = numpyro.sample(
          name="gamma_seasonality",
          fn=dist.Normal(loc=0., scale=1.))

  if weekday_seasonality:
    with numpyro.plate(name="weekday_plate", size=7):
      weekday = numpyro.sample(
          name="weekday",
          fn=dist.Normal(loc=0., scale=.5))
    weekday_series = weekday[jnp.arange(data_size) % 7]

  media_transformed = numpyro.deterministic(
      name="media_transformed",
      value=transform_function(media_data,
                               **transform_kwargs))
  seasonality = media_transforms.calculate_seasonality(
      number_periods=data_size,
      degrees=degrees_seasonality,
      frequency=frequency,
      gamma_seasonality=gamma_seasonality)

  # For national model's case
  trend = jnp.arange(data_size)
  media_einsum = "tc, c -> t"  # t = time, c = channel
  beta_seasonality = 1

  # TODO(): Add conversion of prior for HalfNormal distribution.
  if media_data.ndim == 3:  # For geo model's case
    trend = jnp.expand_dims(trend, axis=-1)
    seasonality = jnp.expand_dims(seasonality, axis=-1)
    media_einsum = "tcg, cg -> tg"  # t = time, c = channel, g = geo
    if weekday_seasonality:
      weekday_series = jnp.expand_dims(weekday_series, axis=-1)
    with numpyro.plate(name="seasonality_plate", size=n_geos):
      beta_seasonality = numpyro.sample(
          name="beta_seasonality",
          fn=dist.HalfNormal(scale=.5))
  # expo_trend is B(1, 1) so that the exponent on time is in [.5, 1.5].
  prediction = (
      intercept + beta_trend * trend ** (expo_trend + 0.5) +
      seasonality * beta_seasonality +
      jnp.einsum(media_einsum, media_transformed, beta_media))
  if extra_features is not None:
    plate_prefixes = ("extra_feature",)
    extra_features_einsum = "tf, f -> t"  # t = time, f = feature
    extra_features_plates_shape = (extra_features.shape[1],)
    if extra_features.ndim == 3:
      plate_prefixes = ("extra_feature", "geo")
      extra_features_einsum = "tfg, fg -> tg"  # t = time, f = feature, g = geo
      extra_features_plates_shape = (extra_features.shape[1], *geo_shape)
    with numpyro.plate_stack(plate_prefixes,
                             sizes=extra_features_plates_shape):
      beta_extra_features = numpyro.sample(
          name="beta_extra_features",
          fn=dist.Normal(loc=0., scale=1.))
    extra_features_effect = jnp.einsum(extra_features_einsum,
                                       extra_features,
                                       beta_extra_features)
    prediction += extra_features_effect

  if weekday_seasonality:
    prediction += weekday_series
  mu = numpyro.deterministic(name="mu", value=prediction)

  numpyro.sample(
      name="target", fn=dist.Normal(loc=mu, scale=sigma), obs=target_data)
