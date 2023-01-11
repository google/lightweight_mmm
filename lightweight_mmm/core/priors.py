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

"""Sets priors and prior related constants for LMMM."""
from typing import Mapping

import immutabledict
from numpyro import distributions as dist

# Core model priors
INTERCEPT = "intercept"
COEF_TREND = "coef_trend"
EXPO_TREND = "expo_trend"
SIGMA = "sigma"
GAMMA_SEASONALITY = "gamma_seasonality"
WEEKDAY = "weekday"
COEF_EXTRA_FEATURES = "coef_extra_features"
COEF_SEASONALITY = "coef_seasonality"

# Lagging priors
LAG_WEIGHT = "lag_weight"
AD_EFFECT_RETENTION_RATE = "ad_effect_retention_rate"
PEAK_EFFECT_DELAY = "peak_effect_delay"

# Saturation priors
EXPONENT = "exponent"
HALF_MAX_EFFECTIVE_CONCENTRATION = "half_max_effective_concentration"
SLOPE = "slope"

# Dynamic trend priors
DYNAMIC_TREND_INITIAL_LEVEL = "dynamic_trend_initial_level"
DYNAMIC_TREND_INITIAL_SLOPE = "dynamic_trend_initial_slope"
DYNAMIC_TREND_LEVEL_VARIANCE = "dynamic_trend_level_variance"
DYNAMIC_TREND_SLOPE_VARIANCE = "dynamic_trend_slope_variance"

MODEL_PRIORS_NAMES = frozenset((
    INTERCEPT,
    COEF_TREND,
    EXPO_TREND,
    SIGMA,
    GAMMA_SEASONALITY,
    WEEKDAY,
    COEF_EXTRA_FEATURES,
    COEF_SEASONALITY,
    LAG_WEIGHT,
    AD_EFFECT_RETENTION_RATE,
    PEAK_EFFECT_DELAY,
    EXPONENT,
    HALF_MAX_EFFECTIVE_CONCENTRATION,
    SLOPE,
    DYNAMIC_TREND_INITIAL_LEVEL,
    DYNAMIC_TREND_INITIAL_SLOPE,
    DYNAMIC_TREND_LEVEL_VARIANCE,
    DYNAMIC_TREND_SLOPE_VARIANCE,
))

GEO_ONLY_PRIORS = frozenset((COEF_SEASONALITY,))


def get_default_priors() -> Mapping[str, dist.Distribution]:
  # Since JAX cannot be called before absl.app.run in tests we get default
  # priors from a function.
  return immutabledict.immutabledict({
      INTERCEPT: dist.HalfNormal(scale=2.),
      COEF_TREND: dist.Normal(loc=0., scale=1.),
      EXPO_TREND: dist.Uniform(low=0.5, high=1.5),
      SIGMA: dist.Gamma(concentration=1., rate=1.),
      GAMMA_SEASONALITY: dist.Normal(loc=0., scale=1.),
      WEEKDAY: dist.Normal(loc=0., scale=.5),
      COEF_EXTRA_FEATURES: dist.Normal(loc=0., scale=1.),
      COEF_SEASONALITY: dist.HalfNormal(scale=.5),
      AD_EFFECT_RETENTION_RATE: dist.Beta(concentration1=1., concentration0=1.),
      PEAK_EFFECT_DELAY: dist.HalfNormal(scale=2.),
      EXPONENT: dist.Beta(concentration1=9., concentration0=1.),
      LAG_WEIGHT: dist.Beta(concentration1=2., concentration0=1.),
      HALF_MAX_EFFECTIVE_CONCENTRATION: dist.Gamma(concentration=1., rate=1.),
      SLOPE: dist.Gamma(concentration=1., rate=1.),
      DYNAMIC_TREND_INITIAL_LEVEL: dist.Normal(loc=.5, scale=2.5),
      DYNAMIC_TREND_INITIAL_SLOPE: dist.Normal(loc=0., scale=.2),
      DYNAMIC_TREND_LEVEL_VARIANCE: dist.Uniform(low=0., high=.1),
      DYNAMIC_TREND_SLOPE_VARIANCE: dist.Uniform(low=0., high=.01),
  })
