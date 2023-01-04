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

"""Tests for media_transforms."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from lightweight_mmm import media_transforms


class MediaTransformsTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="2d_four_channels",
          data=np.ones((100, 4)),
          ad_effect_retention_rate=np.array([0.9, 0.8, 0.7, 1]),
          peak_effect_delay=np.array([0.9, 0.8, 0.7, 1]),
          number_lags=5),
      dict(
          testcase_name="2d_one_channel",
          data=np.ones((300, 1)),
          ad_effect_retention_rate=np.array([0.2]),
          peak_effect_delay=np.array([1]),
          number_lags=10),
      dict(
          testcase_name="3d_10channels_10geos",
          data=np.ones((100, 10, 10)),
          ad_effect_retention_rate=np.ones(10),
          peak_effect_delay=np.ones(10),
          number_lags=13),
      dict(
          testcase_name="3d_10channels_8geos",
          data=np.ones((100, 10, 8)),
          ad_effect_retention_rate=np.ones(10),
          peak_effect_delay=np.ones(10),
          number_lags=13),
  ])
  def test_carryover_produces_correct_shape(self, data,
                                            ad_effect_retention_rate,
                                            peak_effect_delay, number_lags):

    generated_output = media_transforms.carryover(data,
                                                  ad_effect_retention_rate,
                                                  peak_effect_delay,
                                                  number_lags)
    self.assertEqual(generated_output.shape, data.shape)

  @parameterized.named_parameters([
      dict(
          testcase_name="2d_three_channels",
          data=np.ones((100, 3)),
          half_max_effective_concentration=np.array([0.9, 0.8, 0.7]),
          slope=np.array([2, 2, 1])),
      dict(
          testcase_name="2d_one_channels",
          data=np.ones((100, 1)),
          half_max_effective_concentration=np.array([0.9]),
          slope=np.array([5])),
      dict(
          testcase_name="3d_10channels_5geos",
          data=np.ones((100, 10, 5)),
          half_max_effective_concentration=np.expand_dims(np.ones(10), axis=-1),
          slope=np.expand_dims(np.ones(10), axis=-1)),
      dict(
          testcase_name="3d_8channels_10geos",
          data=np.ones((100, 8, 10)),
          half_max_effective_concentration=np.expand_dims(np.ones(8), axis=-1),
          slope=np.expand_dims(np.ones(8), axis=-1)),
  ])
  def test_hill_produces_correct_shape(self, data,
                                       half_max_effective_concentration, slope):
    generated_output = media_transforms.hill(
        data=data,
        half_max_effective_concentration=half_max_effective_concentration,
        slope=slope)

    self.assertEqual(generated_output.shape, data.shape)

  @parameterized.named_parameters([
      dict(
          testcase_name="2d_five_channels",
          data=np.ones((100, 5)),
          lag_weight=np.array([0.2, 0.3, 0.8, 0.2, 0.1]),
          normalise=True),
      dict(
          testcase_name="2d_one_channels",
          data=np.ones((100, 1)),
          lag_weight=np.array([0.4]),
          normalise=False),
      dict(
          testcase_name="3d_10channels_5geos",
          data=np.ones((100, 10, 5)),
          lag_weight=np.expand_dims(np.ones(10), axis=-1),
          normalise=True),
      dict(
          testcase_name="3d_8channels_10geos",
          data=np.ones((100, 8, 10)),
          lag_weight=np.expand_dims(np.ones(8), axis=-1),
          normalise=True),
  ])
  def test_adstock_produces_correct_shape(self, data, lag_weight, normalise):
    generated_output = media_transforms.adstock(
        data=data, lag_weight=lag_weight, normalise=normalise)

    self.assertEqual(generated_output.shape, data.shape)

  def test_apply_exponent_safe_produces_correct_shape(self):
    data = jnp.arange(50).reshape((10, 5))
    exponent = jnp.full(5, 0.5)

    output = media_transforms.apply_exponent_safe(data=data, exponent=exponent)

    np.testing.assert_array_equal(x=output, y=data**exponent)

  def test_apply_exponent_safe_produces_same_exponent_results(self):
    data = jnp.ones((10, 5))
    exponent = jnp.full(5, 0.5)

    output = media_transforms.apply_exponent_safe(data=data, exponent=exponent)

    self.assertEqual(output.shape, data.shape)

  def test_apply_exponent_safe_produces_non_nan_or_inf_grads(self):
    def f_safe(data, exponent):
      x = media_transforms.apply_exponent_safe(data=data, exponent=exponent)
      return x.sum()
    data = jnp.ones((10, 5))
    data = data.at[0, 0].set(0.)
    exponent = jnp.full(5, 0.5)

    grads = jax.grad(f_safe)(data, exponent)

    self.assertFalse(np.isnan(grads).any())
    self.assertFalse(np.isinf(grads).any())

  def test_adstock_zeros_stay_zeros(self):
    data = jnp.zeros((10, 5))
    lag_weight = jnp.full(5, 0.5)

    generated_output = media_transforms.adstock(
        data=data, lag_weight=lag_weight)

    np.testing.assert_array_equal(x=generated_output, y=data)

  def test_hill_zeros_stay_zeros(self):
    data = jnp.zeros((10, 5))
    half_max_effective_concentration = jnp.full(5, 0.5)
    slope = jnp.full(5, 0.5)

    generated_output = media_transforms.hill(
        data=data,
        half_max_effective_concentration=half_max_effective_concentration,
        slope=slope)

    np.testing.assert_array_equal(x=generated_output, y=data)

  def test_carryover_zeros_stay_zeros(self):
    data = jnp.zeros((10, 5))
    ad_effect_retention_rate = jnp.full(5, 0.5)
    peak_effect_delay = jnp.full(5, 0.5)

    generated_output = media_transforms.carryover(
        data=data,
        ad_effect_retention_rate=ad_effect_retention_rate,
        peak_effect_delay=peak_effect_delay)

    np.testing.assert_array_equal(x=generated_output, y=data)


@parameterized.parameters(range(1, 5))
def test_calculate_seasonality_produces_correct_standard_deviation(
    self, degrees):
  # It's not very obvious that this is the expected standard deviation, but it
  # seems to be true mathematically and this makes a very convenient unit test.
  expected_standard_deviation = jnp.sqrt(degrees)

  seasonal_curve = media_transforms.calculate_seasonality(
      number_periods=1,
      degrees=degrees,
      gamma_seasonality=1,
      frequency=1200,
  )
  observed_standard_deviation = jnp.std(seasonal_curve)

  self.assertAlmostEqual(
      observed_standard_deviation, expected_standard_deviation, delta=0.01)


if __name__ == "__main__":
  absltest.main()
