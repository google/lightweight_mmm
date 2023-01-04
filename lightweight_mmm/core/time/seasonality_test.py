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

"""Tests for seasonality."""

import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro import handlers

from absl.testing import absltest
from absl.testing import parameterized
from lightweight_mmm.core import priors
from lightweight_mmm.core.time import seasonality


class SeasonalityTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="2_degrees",
          seasonality_arange_value=150,
          degrees_arange_shape=5,
          gamma_seasonality_shape=(5, 2),
      ),
      dict(
          testcase_name="10_degree",
          seasonality_arange_value=150,
          degrees_arange_shape=10,
          gamma_seasonality_shape=(10, 2),
      ),
      dict(
          testcase_name="1_degree",
          seasonality_arange_value=200,
          degrees_arange_shape=1,
          gamma_seasonality_shape=(1, 2),
      ),
  ])
  def test_core_sinusoidal_seasonality_produces_correct_shape(
      self, seasonality_arange_value, degrees_arange_shape,
      gamma_seasonality_shape):
    seasonality_arange = jnp.expand_dims(
        jnp.arange(seasonality_arange_value), axis=-1)
    degrees_arange = jnp.arange(degrees_arange_shape)
    gamma_seasonality = jnp.ones(gamma_seasonality_shape)

    seasonality_values = seasonality._sinusoidal_seasonality(
        seasonality_arange=seasonality_arange,
        degrees_arange=degrees_arange,
        gamma_seasonality=gamma_seasonality,
        frequency=52,
    )
    self.assertEqual(seasonality_values.shape, (seasonality_arange_value,))

  @parameterized.named_parameters(
      dict(
          testcase_name="ten_degrees_national",
          data_shape=(500, 5),
          degrees_seasonality=10,
          expected_shape=(10, 500),
      ),
      dict(
          testcase_name="ten_degrees_geo",
          data_shape=(500, 5, 5),
          degrees_seasonality=10,
          expected_shape=(10, 500, 1),
      ),
      dict(
          testcase_name="one_degrees_national",
          data_shape=(500, 5),
          degrees_seasonality=1,
          expected_shape=(10, 500),
      ),
      dict(
          testcase_name="one_degrees_geo",
          data_shape=(500, 5, 5),
          degrees_seasonality=1,
          expected_shape=(10, 500, 1),
      ),
  )
  def test_model_sinusoidal_seasonality_produces_correct_shape(
      self, data_shape, degrees_seasonality, expected_shape):

    def mock_model_function(data, degrees_seasonality, frequency):
      numpyro.deterministic(
          "seasonality",
          seasonality.sinusoidal_seasonality(
              data=data,
              degrees_seasonality=degrees_seasonality,
              custom_priors={},
              frequency=frequency))

    num_samples = 10
    data = jnp.ones(data_shape)
    kernel = numpyro.infer.NUTS(model=mock_model_function)
    mcmc = numpyro.infer.MCMC(
        sampler=kernel, num_warmup=10, num_samples=num_samples, num_chains=1)
    rng_key = jax.random.PRNGKey(0)

    mcmc.run(
        rng_key,
        data=data,
        degrees_seasonality=degrees_seasonality,
        frequency=52,
    )
    seasonality_values = mcmc.get_samples()["seasonality"]

    self.assertEqual(seasonality_values.shape, expected_shape)

  def test_sinusoidal_seasonality_custom_priors_are_taken_correctly(self):
    prior_name = priors.GAMMA_SEASONALITY
    expected_value1, expected_value2 = 5.2, 7.56
    custom_priors = {
        prior_name:
            dist.Kumaraswamy(
                concentration1=expected_value1, concentration0=expected_value2)
    }
    media = jnp.ones((10, 5, 5))
    degrees_seasonality = 3
    frequency = 365

    trace_handler = handlers.trace(
        handlers.seed(seasonality.sinusoidal_seasonality, rng_seed=0))
    trace = trace_handler.get_trace(
        data=media,
        custom_priors=custom_priors,
        degrees_seasonality=degrees_seasonality,
        frequency=frequency,
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

  @parameterized.named_parameters(
      dict(
          testcase_name="ten_degrees",
          data_shape=(500, 3),
          expected_shape=(10, 500),
      ),
      dict(
          testcase_name="five_degrees",
          data_shape=(500, 3, 5),
          expected_shape=(10, 500, 1),
      ),
  )
  def test_intra_week_seasonality_produces_correct_shape(
      self, data_shape, expected_shape):

    def mock_model_function(data):
      numpyro.deterministic(
          "intra_week",
          seasonality.intra_week_seasonality(
              data=data,
              custom_priors={},
          ))

    num_samples = 10
    data = jnp.ones(data_shape)
    kernel = numpyro.infer.NUTS(model=mock_model_function)
    mcmc = numpyro.infer.MCMC(
        sampler=kernel, num_warmup=10, num_samples=num_samples, num_chains=1)
    rng_key = jax.random.PRNGKey(0)

    mcmc.run(rng_key, data=data)
    seasonality_values = mcmc.get_samples()["intra_week"]

    self.assertEqual(seasonality_values.shape, expected_shape)

  def test_intra_week_seasonality_custom_priors_are_taken_correctly(self):
    prior_name = priors.WEEKDAY
    expected_value1, expected_value2 = 5.2, 7.56
    custom_priors = {
        prior_name:
            dist.Kumaraswamy(
                concentration1=expected_value1, concentration0=expected_value2)
    }
    media = jnp.ones((10, 5, 5))

    trace_handler = handlers.trace(
        handlers.seed(seasonality.intra_week_seasonality, rng_seed=0))
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


if __name__ == "__main__":
  absltest.main()
