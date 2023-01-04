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

"""Tests for lagging."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import handlers
import numpyro.distributions as dist

from lightweight_mmm.core import priors
from lightweight_mmm.core.transformations import lagging


class LaggingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="national",
          data_shape=(150, 3),
          ad_effect_retention_rate_shape=(3,),
          peak_effect_delay_shape=(3,),
          number_lags=13,
      ),
      dict(
          testcase_name="geo",
          data_shape=(150, 3, 5),
          ad_effect_retention_rate_shape=(3,),
          peak_effect_delay_shape=(3,),
          number_lags=13,
      ),
  )
  def test_core_carryover_produces_correct_shape(
      self,
      data_shape,
      ad_effect_retention_rate_shape,
      peak_effect_delay_shape,
      number_lags,
  ):
    data = jnp.ones(data_shape)
    ad_effect_retention_rate = jnp.ones(ad_effect_retention_rate_shape)
    peak_effect_delay = jnp.ones(peak_effect_delay_shape)

    output = lagging._carryover(
        data=data,
        ad_effect_retention_rate=ad_effect_retention_rate,
        peak_effect_delay=peak_effect_delay,
        number_lags=number_lags,
    )

    self.assertEqual(output.shape, data_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name="national",
          data_shape=(150, 3),
      ),
      dict(
          testcase_name="geo",
          data_shape=(150, 3, 5),
      ),
  )
  def test_carryover_produces_correct_shape(self, data_shape):

    def mock_model_function(data, number_lags):
      numpyro.deterministic(
          "carryover",
          lagging.carryover(
              data=data, custom_priors={}, number_lags=number_lags))

    num_samples = 10
    data = jnp.ones(data_shape)
    number_lags = 15
    kernel = numpyro.infer.NUTS(model=mock_model_function)
    mcmc = numpyro.infer.MCMC(
        sampler=kernel, num_warmup=10, num_samples=num_samples, num_chains=1)
    rng_key = jax.random.PRNGKey(0)

    mcmc.run(rng_key, data=data, number_lags=number_lags)
    carryover_values = mcmc.get_samples()["carryover"]

    self.assertEqual(carryover_values.shape, (num_samples, *data.shape))

  @parameterized.named_parameters(
      dict(
          testcase_name="ad_effect_retention_rate",
          prior_name=priors.AD_EFFECT_RETENTION_RATE,
      ),
      dict(
          testcase_name="peak_effect_delay",
          prior_name=priors.PEAK_EFFECT_DELAY,
      ),
  )
  def test_carryover_custom_priors_are_taken_correctly(self, prior_name):
    expected_value1, expected_value2 = 5.2, 7.56
    custom_priors = {
        prior_name:
            dist.Kumaraswamy(
                concentration1=expected_value1, concentration0=expected_value2)
    }
    media = jnp.ones((10, 5, 5))
    number_lags = 13

    trace_handler = handlers.trace(handlers.seed(lagging.carryover, rng_seed=0))
    trace = trace_handler.get_trace(
        data=media,
        custom_priors=custom_priors,
        number_lags=number_lags,
    )
    values_and_dists = {
        name: site["fn"] for name, site in trace.items() if "fn" in site
    }

    used_distribution = values_and_dists[prior_name]
    used_distribution = used_distribution.base_dist
    self.assertIsInstance(used_distribution, dist.Kumaraswamy)
    self.assertEqual(used_distribution.concentration0, expected_value2)
    self.assertEqual(used_distribution.concentration1, expected_value1)

  @parameterized.named_parameters(
      dict(
          testcase_name="national",
          data_shape=(150, 3),
          lag_weight_shape=(3,),
      ),
      dict(
          testcase_name="geo",
          data_shape=(150, 3, 5),
          lag_weight_shape=(3, 1),
      ),
  )
  def test_core_adstock_produces_correct_shape(self, data_shape,
                                               lag_weight_shape):
    data = jnp.ones(data_shape)
    lag_weight = jnp.ones(lag_weight_shape)

    output = lagging._adstock(data=data, lag_weight=lag_weight)

    self.assertEqual(output.shape, data_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name="national",
          data_shape=(150, 3),
      ),
      dict(
          testcase_name="geo",
          data_shape=(150, 3, 5),
      ),
  )
  def test_adstock_produces_correct_shape(self, data_shape):

    def mock_model_function(data, normalise):
      numpyro.deterministic(
          "adstock",
          lagging.adstock(data=data, custom_priors={}, normalise=normalise))

    num_samples = 10
    data = jnp.ones(data_shape)
    normalise = True
    kernel = numpyro.infer.NUTS(model=mock_model_function)
    mcmc = numpyro.infer.MCMC(
        sampler=kernel, num_warmup=10, num_samples=num_samples, num_chains=1)
    rng_key = jax.random.PRNGKey(0)

    mcmc.run(rng_key, data=data, normalise=normalise)
    adstock_values = mcmc.get_samples()["adstock"]

    self.assertEqual(adstock_values.shape, (num_samples, *data.shape))

  def test_adstock_custom_priors_are_taken_correctly(self):
    prior_name = priors.LAG_WEIGHT
    expected_value1, expected_value2 = 5.2, 7.56
    custom_priors = {
        prior_name:
            dist.Kumaraswamy(
                concentration1=expected_value1, concentration0=expected_value2)
    }
    data = jnp.ones((10, 5, 5))

    trace_handler = handlers.trace(handlers.seed(lagging.adstock, rng_seed=0))
    trace = trace_handler.get_trace(
        data=data,
        custom_priors=custom_priors,
        normalise=True,
    )
    values_and_dists = {
        name: site["fn"] for name, site in trace.items() if "fn" in site
    }

    used_distribution = values_and_dists[prior_name]
    used_distribution = used_distribution.base_dist
    self.assertIsInstance(used_distribution, dist.Kumaraswamy)
    self.assertEqual(used_distribution.concentration0, expected_value2)
    self.assertEqual(used_distribution.concentration1, expected_value1)

  def test_adstock_zeros_stay_zeros(self):
    data = jnp.zeros((10, 5))
    lag_weight = jnp.full(5, 0.5)

    generated_output = lagging._adstock(data=data, lag_weight=lag_weight)

    np.testing.assert_array_equal(x=generated_output, y=data)

  def test_carryover_zeros_stay_zeros(self):
    data = jnp.zeros((10, 5))
    ad_effect_retention_rate = jnp.full(5, 0.5)
    peak_effect_delay = jnp.full(5, 0.5)

    generated_output = lagging._carryover(
        data=data,
        ad_effect_retention_rate=ad_effect_retention_rate,
        peak_effect_delay=peak_effect_delay,
        number_lags=7,
    )

    np.testing.assert_array_equal(x=generated_output, y=data)


if __name__ == "__main__":
  absltest.main()
