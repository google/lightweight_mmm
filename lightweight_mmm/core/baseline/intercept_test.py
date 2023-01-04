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

"""Tests for intercept."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpyro
from numpyro import handlers
import numpyro.distributions as dist

from lightweight_mmm.core import core_utils
from lightweight_mmm.core import priors
from lightweight_mmm.core.baseline import intercept


class InterceptTest(parameterized.TestCase):

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
  def test_simple_intercept_produces_output_correct_shape(self, data_shape):

    def mock_model_function(data):
      numpyro.deterministic(
          "intercept_values",
          intercept.simple_intercept(data=data, custom_priors={}))

    num_samples = 10
    data = jnp.ones(data_shape)
    n_geos = core_utils.get_number_geos(data=data)
    kernel = numpyro.infer.NUTS(model=mock_model_function)
    mcmc = numpyro.infer.MCMC(
        sampler=kernel, num_warmup=10, num_samples=num_samples, num_chains=1)
    rng_key = jax.random.PRNGKey(0)

    mcmc.run(rng_key, data=data)
    intercept_values = mcmc.get_samples()["intercept_values"]

    self.assertEqual(intercept_values.shape, (num_samples, n_geos))

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
  def test_simple_intercept_takes_custom_priors_correctly(self, data_shape):
    prior_name = priors.INTERCEPT
    expected_value1, expected_value2 = 5.2, 7.56
    custom_priors = {
        prior_name:
            dist.Kumaraswamy(
                concentration1=expected_value1, concentration0=expected_value2)
    }
    media = jnp.ones(data_shape)

    trace_handler = handlers.trace(
        handlers.seed(intercept.simple_intercept, rng_seed=0))
    trace = trace_handler.get_trace(data=media, custom_priors=custom_priors)
    values_and_dists = {
        name: site["fn"] for name, site in trace.items() if "fn" in site
    }

    used_distribution = values_and_dists[prior_name].base_dist
    self.assertIsInstance(used_distribution, dist.Kumaraswamy)
    self.assertEqual(used_distribution.concentration0, expected_value2)
    self.assertEqual(used_distribution.concentration1, expected_value1)


if __name__ == "__main__":
  absltest.main()
