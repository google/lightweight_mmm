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

"""Tests for saturation."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import handlers
import numpyro.distributions as dist

from lightweight_mmm.core import priors
from lightweight_mmm.core.transformations import saturation


class SaturationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="national",
          data_shape=(150, 3),
          half_max_effective_concentration_shape=(3,),
          slope_shape=(3,),
      ),
      dict(
          testcase_name="geo",
          data_shape=(150, 3, 5),
          half_max_effective_concentration_shape=(3, 1),
          slope_shape=(3, 1),
      ),
  )
  def test_hill_core_produces_correct_shape(
      self, data_shape, half_max_effective_concentration_shape, slope_shape):
    data = jnp.ones(data_shape)
    half_max_effective_concentration = jnp.ones(
        half_max_effective_concentration_shape)
    slope = jnp.ones(slope_shape)

    output = saturation._hill(
        data=data,
        half_max_effective_concentration=half_max_effective_concentration,
        slope=slope,
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
  def test_hill_produces_correct_shape(self, data_shape):

    def mock_model_function(data):
      numpyro.deterministic("hill",
                            saturation.hill(data=data, custom_priors={}))

    num_samples = 10
    data = jnp.ones(data_shape)
    kernel = numpyro.infer.NUTS(model=mock_model_function)
    mcmc = numpyro.infer.MCMC(
        sampler=kernel, num_warmup=10, num_samples=num_samples, num_chains=1)
    rng_key = jax.random.PRNGKey(0)

    mcmc.run(rng_key, data=data)
    output_values = mcmc.get_samples()["hill"]

    self.assertEqual(output_values.shape, (num_samples, *data.shape))

  @parameterized.named_parameters(
      dict(
          testcase_name="half_max_effective_concentration",
          prior_name=priors.HALF_MAX_EFFECTIVE_CONCENTRATION,
      ),
      dict(
          testcase_name="slope",
          prior_name=priors.SLOPE,
      ),
  )
  def test_hill_custom_priors_are_taken_correctly(self, prior_name):
    expected_value1, expected_value2 = 5.2, 7.56
    custom_priors = {
        prior_name:
            dist.Kumaraswamy(
                concentration1=expected_value1, concentration0=expected_value2)
    }
    media = jnp.ones((10, 5, 5))

    trace_handler = handlers.trace(handlers.seed(saturation.hill, rng_seed=0))
    trace = trace_handler.get_trace(
        data=media,
        custom_priors=custom_priors,
    )
    values_and_dists = {
        name: site["fn"] for name, site in trace.items() if "fn" in site
    }

    used_distribution = values_and_dists[prior_name]
    used_distribution = used_distribution.base_dist
    self.assertIsInstance(used_distribution, dist.Kumaraswamy)
    self.assertEqual(used_distribution.concentration0, expected_value2)
    self.assertEqual(used_distribution.concentration1, expected_value1)

  def test_exponent_core_produces_correct_shape(self):
    pass

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
  def test_exponent_produces_correct_shape(self, data_shape):

    def mock_model_function(data):
      numpyro.deterministic("outer_exponent",
                            saturation.exponent(data=data, custom_priors={}))

    num_samples = 10
    data = jnp.ones(data_shape)
    kernel = numpyro.infer.NUTS(model=mock_model_function)
    mcmc = numpyro.infer.MCMC(
        sampler=kernel, num_warmup=10, num_samples=num_samples, num_chains=1)
    rng_key = jax.random.PRNGKey(0)

    mcmc.run(rng_key, data=data)
    output_values = mcmc.get_samples()["outer_exponent"]

    self.assertEqual(output_values.shape, (num_samples, *data.shape))

  def test_exponent_custom_priors_are_taken_correctly(self):
    prior_name = priors.EXPONENT
    expected_value1, expected_value2 = 5.2, 7.56
    custom_priors = {
        prior_name:
            dist.Kumaraswamy(
                concentration1=expected_value1, concentration0=expected_value2)
    }
    media = jnp.ones((10, 5, 5))

    trace_handler = handlers.trace(
        handlers.seed(saturation.exponent, rng_seed=0))
    trace = trace_handler.get_trace(
        data=media,
        custom_priors=custom_priors,
    )
    values_and_dists = {
        name: site["fn"] for name, site in trace.items() if "fn" in site
    }

    used_distribution = values_and_dists[prior_name]
    used_distribution = used_distribution.base_dist
    self.assertIsInstance(used_distribution, dist.Kumaraswamy)
    self.assertEqual(used_distribution.concentration0, expected_value2)
    self.assertEqual(used_distribution.concentration1, expected_value1)

  def test_hill_zeros_stay_zeros(self):
    data = jnp.zeros((10, 5))
    half_max_effective_concentration = jnp.full(5, 0.5)
    slope = jnp.full(5, 0.5)

    generated_output = saturation._hill(
        data=data,
        half_max_effective_concentration=half_max_effective_concentration,
        slope=slope,
    )

    np.testing.assert_array_equal(x=generated_output, y=data)

  def test_exponent_zeros_stay_zero(self):
    data = jnp.zeros((10, 5))
    exponent_values = jnp.full(5, 0.5)

    generated_output = saturation._exponent(
        data=data,
        exponent_values=exponent_values,
    )

    np.testing.assert_array_equal(x=generated_output, y=data)


if __name__ == "__main__":
  absltest.main()
