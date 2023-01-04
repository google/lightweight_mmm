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

"""Tests for models."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro import handlers

from lightweight_mmm import models


class ModelsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="one_channel", shape=(10, 1)),
      dict(testcase_name="five_channel", shape=(10, 5)),
      dict(testcase_name="same_channels_as_rows", shape=(10, 10)),
      dict(testcase_name="geo_shape_1", shape=(10, 10, 5)),
      dict(testcase_name="geo_shape_2", shape=(10, 5, 2)),
      dict(testcase_name="one_channel_one_row", shape=(1, 1)))
  def test_transform_adstock_produces_correct_output_shape(self, shape):

    def mock_model_function(media_data):
      numpyro.deterministic(
          "transformed_media",
          models.transform_adstock(media_data, custom_priors={}))

    media = jnp.ones(shape)
    kernel = numpyro.infer.NUTS(model=mock_model_function)
    mcmc = numpyro.infer.MCMC(
        sampler=kernel, num_warmup=10, num_samples=10, num_chains=1)
    rng_key = jax.random.PRNGKey(0)

    mcmc.run(rng_key, media_data=media)
    transformed_media = mcmc.get_samples()["transformed_media"].mean(axis=0)

    self.assertEqual(media.shape, transformed_media.shape)

  @parameterized.named_parameters(
      dict(testcase_name="one_channel", shape=(10, 1)),
      dict(testcase_name="five_channel", shape=(10, 5)),
      dict(testcase_name="same_channels_as_rows", shape=(10, 10)),
      dict(testcase_name="geo_shape_1", shape=(10, 10, 5)),
      dict(testcase_name="geo_shape_2", shape=(10, 5, 2)),
      dict(testcase_name="one_channel_one_row", shape=(1, 1)))
  def test_transform_hill_adstock_produces_correct_output_shape(self, shape):

    def mock_model_function(media_data):
      numpyro.deterministic(
          "transformed_media",
          models.transform_hill_adstock(media_data, custom_priors={}))

    media = jnp.ones(shape)
    kernel = numpyro.infer.NUTS(model=mock_model_function)
    mcmc = numpyro.infer.MCMC(
        sampler=kernel, num_warmup=10, num_samples=10, num_chains=1)
    rng_key = jax.random.PRNGKey(0)

    mcmc.run(rng_key, media_data=media)
    transformed_media = mcmc.get_samples()["transformed_media"].mean(axis=0)

    self.assertEqual(media.shape, transformed_media.shape)

  @parameterized.named_parameters(
      dict(testcase_name="one_channel", shape=(10, 1)),
      dict(testcase_name="five_channel", shape=(10, 5)),
      dict(testcase_name="same_channels_as_rows", shape=(10, 10)),
      dict(testcase_name="geo_shape_1", shape=(10, 10, 5)),
      dict(testcase_name="geo_shape_2", shape=(10, 5, 2)),
      dict(testcase_name="one_channel_one_row", shape=(1, 1)))
  def test_transform_carryover_produces_correct_output_shape(self, shape):

    def mock_model_function(media_data):
      numpyro.deterministic(
          "transformed_media",
          models.transform_carryover(media_data, custom_priors={}))

    media = jnp.ones(shape)
    kernel = numpyro.infer.NUTS(model=mock_model_function)
    mcmc = numpyro.infer.MCMC(
        sampler=kernel, num_warmup=10, num_samples=10, num_chains=1)
    rng_key = jax.random.PRNGKey(0)

    mcmc.run(rng_key, media_data=media)
    transformed_media = mcmc.get_samples()["transformed_media"].mean(axis=0)

    self.assertEqual(media.shape, transformed_media.shape)

  @parameterized.named_parameters(
      dict(
          testcase_name="national_no_extra",
          media_shape=(10, 3),
          extra_features_shape=(),
          target_shape=(10,),
          total_costs_shape=(3,)),
      dict(
          testcase_name="national_extra",
          media_shape=(10, 5),
          extra_features_shape=(10, 2),
          target_shape=(10,),
          total_costs_shape=(5,)),
      dict(
          testcase_name="geo_extra_3d",
          media_shape=(10, 7, 5),
          extra_features_shape=(10, 8, 5),
          target_shape=(10, 5),
          total_costs_shape=(7, 1)),
      dict(
          testcase_name="geo_no_extra",
          media_shape=(10, 7, 5),
          extra_features_shape=(),
          target_shape=(10, 5),
          total_costs_shape=(7, 1)))
  def test_media_mix_model_parameters_have_correct_shapes(
      self, media_shape, extra_features_shape, target_shape, total_costs_shape):
    media = jnp.ones(media_shape)
    extra_features = None if not extra_features_shape else jnp.ones(
        extra_features_shape)
    costs_prior = jnp.ones(total_costs_shape)
    degrees = 2
    target = jnp.ones(target_shape)
    kernel = numpyro.infer.NUTS(model=models.media_mix_model)
    mcmc = numpyro.infer.MCMC(
        sampler=kernel, num_warmup=10, num_samples=10, num_chains=1)
    rng_key = jax.random.PRNGKey(0)

    mcmc.run(
        rng_key,
        media_data=media,
        extra_features=extra_features,
        target_data=target,
        media_prior=costs_prior,
        degrees_seasonality=degrees,
        custom_priors={},
        frequency=52,
        transform_function=models.transform_carryover)
    trace = mcmc.get_samples()

    self.assertEqual(
        jnp.squeeze(trace["intercept"].mean(axis=0)).shape, target_shape[1:])
    self.assertEqual(
        jnp.squeeze(trace["sigma"].mean(axis=0)).shape, target_shape[1:])
    self.assertEqual(
        jnp.squeeze(trace["expo_trend"].mean(axis=0)).shape, ())
    self.assertEqual(
        jnp.squeeze(trace["coef_trend"].mean(axis=0)).shape, target_shape[1:])
    self.assertEqual(
        jnp.squeeze(trace["coef_media"].mean(axis=0)).shape, media_shape[1:])
    if extra_features_shape:
      self.assertEqual(trace["coef_extra_features"].mean(axis=0).shape,
                       extra_features.shape[1:])
    self.assertEqual(trace["gamma_seasonality"].mean(axis=0).shape,
                     (degrees, 2))
    self.assertEqual(trace["mu"].mean(axis=0).shape, target_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name=f"model_{models._INTERCEPT}",
          prior_name=models._INTERCEPT,
          transform_function=models.transform_carryover),
      dict(
          testcase_name=f"model_{models._COEF_TREND}",
          prior_name=models._COEF_TREND,
          transform_function=models.transform_carryover),
      dict(
          testcase_name=f"model_{models._EXPO_TREND}",
          prior_name=models._EXPO_TREND,
          transform_function=models.transform_carryover),
      dict(
          testcase_name=f"model_{models._SIGMA}",
          prior_name=models._SIGMA,
          transform_function=models.transform_carryover),
      dict(
          testcase_name=f"model_{models._GAMMA_SEASONALITY}",
          prior_name=models._GAMMA_SEASONALITY,
          transform_function=models.transform_carryover),
      dict(
          testcase_name=f"model_{models._WEEKDAY}",
          prior_name=models._WEEKDAY,
          transform_function=models.transform_carryover),
      dict(
          testcase_name=f"model_{models._COEF_EXTRA_FEATURES}",
          prior_name=models._COEF_EXTRA_FEATURES,
          transform_function=models.transform_carryover),
      dict(
          testcase_name=f"model_{models._COEF_SEASONALITY}",
          prior_name=models._COEF_SEASONALITY,
          transform_function=models.transform_carryover),
      dict(
          testcase_name=f"carryover_{models._AD_EFFECT_RETENTION_RATE}",
          prior_name=models._AD_EFFECT_RETENTION_RATE,
          transform_function=models.transform_carryover),
      dict(
          testcase_name=f"carryover_{models._PEAK_EFFECT_DELAY}",
          prior_name=models._PEAK_EFFECT_DELAY,
          transform_function=models.transform_carryover),
      dict(
          testcase_name=f"carryover_{models._EXPONENT}",
          prior_name=models._EXPONENT,
          transform_function=models.transform_carryover),
      dict(
          testcase_name=f"adstock_{models._EXPONENT}",
          prior_name=models._EXPONENT,
          transform_function=models.transform_adstock),
      dict(
          testcase_name=f"adstock_{models._LAG_WEIGHT}",
          prior_name=models._LAG_WEIGHT,
          transform_function=models.transform_adstock),
      dict(
          testcase_name=f"hilladstock_{models._LAG_WEIGHT}",
          prior_name=models._LAG_WEIGHT,
          transform_function=models.transform_hill_adstock),
      dict(
          testcase_name=f"hilladstock_{models._HALF_MAX_EFFECTIVE_CONCENTRATION}",
          prior_name=models._HALF_MAX_EFFECTIVE_CONCENTRATION,
          transform_function=models.transform_hill_adstock),
      dict(
          testcase_name=f"hilladstock_{models._SLOPE}",
          prior_name=models._SLOPE,
          transform_function=models.transform_hill_adstock),
  )
  def test_media_mix_model_custom_priors_are_taken_correctly(
      self, prior_name, transform_function):
    expected_value1, expected_value2 = 5.2, 7.56
    custom_priors = {
        prior_name: dist.Kumaraswamy(
            concentration1=expected_value1, concentration0=expected_value2)}
    media = jnp.ones((10, 5, 5))
    extra_features = jnp.ones((10, 3, 5))
    costs_prior = jnp.ones((5, 1))
    target = jnp.ones((10, 5))

    trace_handler = handlers.trace(handlers.seed(
        models.media_mix_model, rng_seed=0))
    trace = trace_handler.get_trace(
        media_data=media,
        extra_features=extra_features,
        target_data=target,
        media_prior=costs_prior,
        custom_priors=custom_priors,
        degrees_seasonality=2,
        frequency=52,
        transform_function=transform_function,
        weekday_seasonality=True
    )
    values_and_dists = {
        name: site["fn"]
        for name, site in trace.items() if "fn" in site
    }

    used_distribution = values_and_dists[prior_name]
    if isinstance(used_distribution, dist.ExpandedDistribution):
      used_distribution = used_distribution.base_dist
    self.assertIsInstance(used_distribution, dist.Kumaraswamy)
    self.assertEqual(used_distribution.concentration0, expected_value2)
    self.assertEqual(used_distribution.concentration1, expected_value1)


if __name__ == "__main__":
  absltest.main()
