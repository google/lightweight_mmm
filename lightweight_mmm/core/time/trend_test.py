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

"""Tests for trend."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import distributions as dist
from numpyro import handlers

from lightweight_mmm.core import core_utils
from lightweight_mmm.core import priors
from lightweight_mmm.core.time import trend


class TrendTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="national",
          coef_trend_shape=(),
          trend_length=150,
          expo_trend_shape=(),
      ),
      dict(
          testcase_name="geo",
          coef_trend_shape=(5,),
          trend_length=150,
          expo_trend_shape=(),
      ),
  ])
  def test_core_trend_with_exponent_produces_correct_shape(
      self, coef_trend_shape, trend_length, expo_trend_shape):
    coef_trend = jnp.ones(coef_trend_shape)
    linear_trend = jnp.arange(trend_length)
    if coef_trend.ndim == 1:  # For geo model's case
      linear_trend = jnp.expand_dims(linear_trend, axis=-1)
    expo_trend = jnp.ones(expo_trend_shape)

    trend_values = trend._trend_with_exponent(
        coef_trend=coef_trend, trend=linear_trend, expo_trend=expo_trend)

    self.assertEqual(trend_values.shape,
                     (linear_trend.shape[0], *coef_trend_shape))

  @parameterized.named_parameters([
      dict(testcase_name="national", data_shape=(150, 3)),
      dict(testcase_name="geo", data_shape=(150, 3, 5)),
  ])
  def test_trend_with_exponent_produces_correct_shape(self, data_shape):

    def mock_model_function(data):
      numpyro.deterministic(
          "trend", trend.trend_with_exponent(
              data=data,
              custom_priors={},
          ))

    num_samples = 10
    data = jnp.ones(data_shape)
    kernel = numpyro.infer.NUTS(model=mock_model_function)
    mcmc = numpyro.infer.MCMC(
        sampler=kernel, num_warmup=10, num_samples=num_samples, num_chains=1)
    rng_key = jax.random.PRNGKey(0)
    coef_expected_shape = () if data.ndim == 2 else (data.shape[2],)

    mcmc.run(rng_key, data=data)
    trend_values = mcmc.get_samples()["trend"]

    self.assertEqual(trend_values.shape,
                     (num_samples, data.shape[0], *coef_expected_shape))

  @parameterized.named_parameters(
      dict(
          testcase_name=f"model_{priors.COEF_TREND}",
          prior_name=priors.COEF_TREND,
      ),
      dict(
          testcase_name=f"model_{priors.EXPO_TREND}",
          prior_name=priors.EXPO_TREND,
      ),
  )
  def test_trend_with_exponent_custom_priors_are_taken_correctly(
      self, prior_name):
    expected_value1, expected_value2 = 5.2, 7.56
    custom_priors = {
        prior_name:
            dist.Kumaraswamy(
                concentration1=expected_value1, concentration0=expected_value2)
    }
    media = jnp.ones((10, 5, 5))

    trace_handler = handlers.trace(
        handlers.seed(trend.trend_with_exponent, rng_seed=0))
    trace = trace_handler.get_trace(
        data=media,
        custom_priors=custom_priors,
    )
    values_and_dists = {
        name: site["fn"] for name, site in trace.items() if "fn" in site
    }

    used_distribution = values_and_dists[prior_name]
    if isinstance(used_distribution, dist.ExpandedDistribution):
      used_distribution = used_distribution.base_dist
    self.assertIsInstance(used_distribution, dist.Kumaraswamy)
    self.assertEqual(used_distribution.concentration0, expected_value2)
    self.assertEqual(used_distribution.concentration1, expected_value1)

  @parameterized.named_parameters([
      dict(
          testcase_name="dynamic_trend_national_shape",
          number_periods=100,
          initial_level_shape=(),
          initial_slope_shape=(),
          variance_level_shape=(),
          variance_slope_shape=(),
      ),
      dict(
          testcase_name="dynamic_trend_geo_shape",
          number_periods=100,
          initial_level_shape=(2,),
          initial_slope_shape=(2,),
          variance_level_shape=(2,),
          variance_slope_shape=(2,),
      ),
  ])
  def test_core_dynamic_trend_produces_correct_shape(
      self, number_periods, initial_level_shape, initial_slope_shape,
      variance_level_shape, variance_slope_shape):
    initial_level = jnp.ones(initial_level_shape)
    initial_slope = jnp.ones(initial_slope_shape)
    variance_level = jnp.ones(variance_level_shape)
    variance_slope = jnp.ones(variance_slope_shape)
    random_walk_level = jnp.arange(number_periods)
    random_walk_slope = jnp.arange(number_periods)
    if initial_level.ndim == 1:  # For geo model's case
      random_walk_level = jnp.expand_dims(random_walk_level, axis=-1)
      random_walk_slope = jnp.expand_dims(random_walk_slope, axis=-1)

    dynamic_trend_values = trend._dynamic_trend(
        number_periods=number_periods,
        random_walk_level=random_walk_level,
        random_walk_slope=random_walk_slope,
        initial_level=initial_level,
        initial_slope=initial_slope,
        variance_level=variance_level,
        variance_slope=variance_slope,
    )

    self.assertEqual(dynamic_trend_values.shape,
                     (number_periods, *initial_level_shape))

  def test_core_dynamic_trend_produces_correct_value(self):
    number_periods = 5
    initial_level = jnp.ones(())
    initial_slope = jnp.ones(())
    variance_level = jnp.ones(())
    variance_slope = jnp.ones(())
    random_walk_level = jnp.arange(number_periods)
    random_walk_slope = jnp.arange(number_periods)
    dynamic_trend_expected_value = jnp.array([1, 3, 7, 14, 25])

    dynamic_trend_values = trend._dynamic_trend(
        number_periods=number_periods,
        random_walk_level=random_walk_level,
        random_walk_slope=random_walk_slope,
        initial_level=initial_level,
        initial_slope=initial_slope,
        variance_level=variance_level,
        variance_slope=variance_slope,
    )

    np.testing.assert_array_equal(x=dynamic_trend_values,
                                  y=dynamic_trend_expected_value)

  @parameterized.named_parameters([
      dict(
          testcase_name="national_with_prediction_is_true",
          data_shape=(100, 3),
          is_trend_prediction=True),
      dict(
          testcase_name="geo_with_prediction_is_true",
          data_shape=(150, 3, 5),
          is_trend_prediction=True),
      dict(
          testcase_name="national_with_prediction_is_false",
          data_shape=(100, 3),
          is_trend_prediction=False),
      dict(
          testcase_name="geo_with_prediction_is_false",
          data_shape=(150, 3, 5),
          is_trend_prediction=False),
  ])
  def test_dynamic_trend_produces_correct_shape(
      self, data_shape, is_trend_prediction):

    def mock_model_function(geo_size, data_size):
      numpyro.deterministic(
          "trend", trend.dynamic_trend(
              geo_size=geo_size,
              data_size=data_size,
              is_trend_prediction=is_trend_prediction,
              custom_priors={},
          ))
    num_samples = 10
    data = jnp.ones(data_shape)
    geo_size = core_utils.get_number_geos(data)
    kernel = numpyro.infer.NUTS(model=mock_model_function)
    mcmc = numpyro.infer.MCMC(
        sampler=kernel, num_warmup=10, num_samples=num_samples, num_chains=1)
    rng_key = jax.random.PRNGKey(0)
    coef_expected_shape = core_utils.get_geo_shape(data)

    mcmc.run(rng_key, geo_size=geo_size, data_size=data_shape[0])
    trend_values = mcmc.get_samples()["trend"]

    self.assertEqual(trend_values.shape,
                     (num_samples, data.shape[0], *coef_expected_shape))

  @parameterized.named_parameters(
      dict(
          testcase_name=f"model_{priors.DYNAMIC_TREND_INITIAL_LEVEL}",
          prior_name=priors.DYNAMIC_TREND_INITIAL_LEVEL,
      ),
      dict(
          testcase_name=f"model_{priors.DYNAMIC_TREND_INITIAL_SLOPE}",
          prior_name=priors.DYNAMIC_TREND_INITIAL_SLOPE,
      ),
      dict(
          testcase_name=f"model_{priors.DYNAMIC_TREND_LEVEL_VARIANCE}",
          prior_name=priors.DYNAMIC_TREND_LEVEL_VARIANCE,
      ),
      dict(
          testcase_name=f"model_{priors.DYNAMIC_TREND_SLOPE_VARIANCE}",
          prior_name=priors.DYNAMIC_TREND_SLOPE_VARIANCE,
      ),
  )
  def test_core_dynamic_trend_custom_priors_are_taken_correctly(
      self, prior_name):
    expected_value1, expected_value2 = 5.2, 7.56
    custom_priors = {
        prior_name:
            dist.Kumaraswamy(
                concentration1=expected_value1, concentration0=expected_value2)
    }
    geo_size = 1
    data_size = 10
    trace_handler = handlers.trace(
        handlers.seed(trend.dynamic_trend, rng_seed=0))
    trace = trace_handler.get_trace(
        geo_size=geo_size,
        data_size=data_size,
        is_trend_prediction=False,
        custom_priors=custom_priors,
    )
    values_and_dists = {
        name: site["fn"] for name, site in trace.items() if "fn" in site
    }

    used_distribution = values_and_dists[prior_name]
    if isinstance(used_distribution, dist.ExpandedDistribution):
      used_distribution = used_distribution.base_dist

    self.assertIsInstance(used_distribution, dist.Kumaraswamy)
    self.assertEqual(used_distribution.concentration0, expected_value2)
    self.assertEqual(used_distribution.concentration1, expected_value1)

  @parameterized.named_parameters([
      dict(
          testcase_name="trend_prediction_is_true",
          is_trend_prediction=True,
          expected_trend_parameter=[
              "random_walk_level_prediction", "random_walk_slope_prediction"]
          ),
      dict(
          testcase_name="trend_prediction_is_false",
          is_trend_prediction=False,
          expected_trend_parameter=[
              "random_walk_level", "random_walk_slope"]
          ),
  ])
  def test_dynamic_trend_is_trend_prediction_produuce_correct_parameter_names(
      self, is_trend_prediction, expected_trend_parameter):

    def mock_model_function(geo_size, data_size):
      numpyro.deterministic(
          "trend", trend.dynamic_trend(
              geo_size=geo_size,
              data_size=data_size,
              is_trend_prediction=is_trend_prediction,
              custom_priors={},
          ))
    num_samples = 10
    data_shape = (10, 3)
    data = jnp.ones(data_shape)
    geo_size = core_utils.get_number_geos(data)
    kernel = numpyro.infer.NUTS(model=mock_model_function)
    mcmc = numpyro.infer.MCMC(
        sampler=kernel, num_warmup=10, num_samples=num_samples, num_chains=1)
    rng_key = jax.random.PRNGKey(0)

    mcmc.run(rng_key, geo_size=geo_size, data_size=data_shape[0])
    trend_parameter = [
        parameter for parameter, _ in mcmc.get_samples().items()
        if parameter.startswith("random_walk")]

    self.assertEqual(trend_parameter, expected_trend_parameter)

if __name__ == "__main__":
  absltest.main()
