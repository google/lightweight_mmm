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

"""Tests for optimize_media."""
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from lightweight_mmm import lightweight_mmm
from lightweight_mmm import optimize_media
from lightweight_mmm import preprocessing


class OptimizeMediaTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(OptimizeMediaTest, cls).setUpClass()
    cls.national_mmm = lightweight_mmm.LightweightMMM()
    cls.national_mmm.fit(
        media=jnp.ones((50, 5)),
        target=jnp.ones(50),
        media_prior=jnp.ones(5) * 50,
        number_warmup=2,
        number_samples=2,
        number_chains=1)
    cls.geo_mmm = lightweight_mmm.LightweightMMM()
    cls.geo_mmm.fit(
        media=jnp.ones((50, 5, 3)),
        target=jnp.ones((50, 3)),
        media_prior=jnp.ones(5) * 50,
        number_warmup=2,
        number_samples=2,
        number_chains=1)

  def setUp(self):
    super().setUp()
    self.mock_minimize = self.enter_context(
        mock.patch.object(optimize_media.optimize, "minimize", autospec=True))

  @parameterized.named_parameters([
      dict(
          testcase_name="national",
          model_name="national_mmm",
          geo_ratio=1),
      dict(
          testcase_name="geo",
          model_name="geo_mmm",
          geo_ratio=np.tile(0.33, reps=(5, 3)))
  ])
  def test_objective_function_generates_correct_value_type_and_sign(
      self, model_name, geo_ratio):

    mmm = getattr(self, model_name)
    extra_features = mmm._extra_features
    time_periods = 10

    kpi_predicted = optimize_media._objective_function(
        extra_features=extra_features,
        media_mix_model=mmm,
        media_input_shape=(time_periods, *mmm.media.shape[1:]),
        media_gap=None,
        target_scaler=None,
        media_scaler=preprocessing.CustomScaler(),
        media_values=jnp.ones(mmm.n_media_channels) * time_periods,
        geo_ratio=geo_ratio,
        seed=10)

    self.assertIsInstance(kpi_predicted, jax.Array)
    self.assertLessEqual(kpi_predicted, 0)
    self.assertEqual(kpi_predicted.shape, ())

  @parameterized.named_parameters([
      dict(
          testcase_name="zero_output",
          media=np.ones(9),
          prices=np.array([1, 2, 3]),
          budget=18,
          expected_value=0),
      dict(
          testcase_name="negative_output",
          media=np.ones(9),
          prices=np.array([1, 2, 3]),
          budget=20,
          expected_value=-2),
      dict(
          testcase_name="positive_output",
          media=np.ones(9),
          prices=np.array([1, 2, 3]),
          budget=16,
          expected_value=2),
      dict(
          testcase_name="bigger_array",
          media=np.ones(18),
          prices=np.array([2, 2, 2]),
          budget=36,
          expected_value=0),
  ])
  def test_budget_constraint(self, media, prices, budget, expected_value):
    generated_value = optimize_media._budget_constraint(
        media=media, prices=prices, budget=budget)

    self.assertEqual(generated_value, expected_value)

  @parameterized.named_parameters([
      dict(
          testcase_name="national_media_scaler",
          model_name="national_mmm"),
      dict(
          testcase_name="geo_media_scaler",
          model_name="geo_mmm")
  ])
  def test_find_optimal_budgets_with_scaler_optimize_called_with_right_params(
      self, model_name):

    mmm = getattr(self, model_name)
    media_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    media_scaler.fit(2 * jnp.ones((10, *mmm.media.shape[1:])))
    optimize_media.find_optimal_budgets(
        n_time_periods=15,
        media_mix_model=mmm,
        budget=30,
        prices=jnp.ones(mmm.n_media_channels),
        target_scaler=None,
        media_scaler=media_scaler)

    _, call_kwargs = self.mock_minimize.call_args_list[0]
    # 15 weeks at 1.2 gives us 12. and 18. bounds times 2 (scaler) 24. and 36.
    np.testing.assert_array_almost_equal(call_kwargs["bounds"].lb,
                                         np.repeat(24., repeats=5) * mmm.n_geos,
                                         decimal=3)
    np.testing.assert_array_almost_equal(call_kwargs["bounds"].ub,
                                         np.repeat(36., repeats=5) * mmm.n_geos,
                                         decimal=3)
    # We only added scaler with divide operation so we only expectec x2 in
    # the divide_by parameter.
    np.testing.assert_array_almost_equal(call_kwargs["fun"].args[5].divide_by,
                                         2 * jnp.ones(mmm.media.shape[1:]),
                                         decimal=3)
    np.testing.assert_array_almost_equal(call_kwargs["fun"].args[5].multiply_by,
                                         jnp.ones(mmm.media.shape[1:]),
                                         decimal=3)

  @parameterized.named_parameters([
      dict(
          testcase_name="national",
          model_name="national_mmm"),
      dict(
          testcase_name="geo",
          model_name="geo_mmm")
  ])
  def test_find_optimal_budgets_without_scaler_optimize_called_with_right_params(
      self, model_name):

    mmm = getattr(self, model_name)
    optimize_media.find_optimal_budgets(
        n_time_periods=15,
        media_mix_model=mmm,
        budget=30,
        prices=jnp.ones(mmm.n_media_channels),
        target_scaler=None,
        media_scaler=None)

    _, call_kwargs = self.mock_minimize.call_args_list[0]
    # 15 weeks at 1.2 gives us 18. bounds
    np.testing.assert_array_almost_equal(
        call_kwargs["bounds"].lb,
        np.repeat(12., repeats=5) * mmm.n_geos,
        decimal=3)
    np.testing.assert_array_almost_equal(
        call_kwargs["bounds"].ub,
        np.repeat(18., repeats=5) * mmm.n_geos,
        decimal=3)

    np.testing.assert_array_almost_equal(
        call_kwargs["fun"].args[5].divide_by,
        jnp.ones(mmm.n_media_channels),
        decimal=3)
    np.testing.assert_array_almost_equal(
        call_kwargs["fun"].args[5].multiply_by,
        jnp.ones(mmm.n_media_channels),
        decimal=3)

  @parameterized.named_parameters([
      dict(
          testcase_name="national",
          model_name="national_mmm"),
      dict(
          testcase_name="geo",
          model_name="geo_mmm")
  ])
  def test_predict_called_with_right_args(self, model_name):
    mmm = getattr(self, model_name)

    optimize_media.find_optimal_budgets(
        n_time_periods=15,
        media_mix_model=mmm,
        budget=30,
        prices=jnp.ones(mmm.n_media_channels),
        target_scaler=None,
        media_scaler=None)

  @parameterized.named_parameters([
      dict(
          testcase_name="national",
          model_name="national_mmm"),
      dict(
          testcase_name="geo",
          model_name="geo_mmm")
  ])
  def test_budget_lower_than_constraints_warns_user(self, model_name):
    mmm = getattr(self, model_name)
    expected_warning = (
        "Budget given is smaller than the lower bounds of the constraints for "
        "optimization. This will lead to faulty optimization. Please either "
        "increase the budget or change the lower bound by increasing the "
        "percentage decrease with the `bounds_lower_pct` parameter.")

    with self.assertLogs(level="WARNING") as context_manager:
      optimize_media.find_optimal_budgets(
          n_time_periods=5,
          media_mix_model=mmm,
          budget=1,
          prices=jnp.ones(mmm.n_media_channels))
    self.assertEqual(f"WARNING:absl:{expected_warning}",
                     context_manager.output[0])

  @parameterized.named_parameters([
      dict(
          testcase_name="national",
          model_name="national_mmm"),
      dict(
          testcase_name="geo",
          model_name="geo_mmm")
  ])
  def test_budget_higher_than_constraints_warns_user(self, model_name):
    mmm = getattr(self, model_name)
    expected_warning = (
        "Budget given is larger than the upper bounds of the constraints for "
        "optimization. This will lead to faulty optimization. Please either "
        "reduce the budget or change the upper bound by increasing the "
        "percentage increase with the `bounds_upper_pct` parameter.")

    with self.assertLogs(level="WARNING") as context_manager:
      optimize_media.find_optimal_budgets(
          n_time_periods=5,
          media_mix_model=mmm,
          budget=2000,
          prices=jnp.ones(5))
    self.assertEqual(f"WARNING:absl:{expected_warning}",
                     context_manager.output[0])

  @parameterized.named_parameters([
      dict(
          testcase_name="national",
          model_name="national_mmm",
          expected_len=3),
      dict(
          testcase_name="geo",
          model_name="geo_mmm",
          expected_len=3)
  ])
  def test_find_optimal_budgets_has_right_output_length_datatype(
      self, model_name, expected_len):

    mmm = getattr(self, model_name)
    media_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    media_scaler.fit(2 * jnp.ones((10, *mmm.media.shape[1:])))
    results = optimize_media.find_optimal_budgets(
        n_time_periods=15,
        media_mix_model=mmm,
        budget=30,
        prices=jnp.ones(mmm.n_media_channels),
        target_scaler=None,
        media_scaler=media_scaler)
    self.assertLen(results, expected_len)
    self.assertIsInstance(results[1], jax.Array)
    self.assertIsInstance(results[2], jax.Array)

  @parameterized.named_parameters([
      dict(
          testcase_name="national_prices",
          model_name="national_mmm",
          prices=np.array([1., 0.8, 1.2, 1.5, 0.5]),
      ),
      dict(
          testcase_name="national_ones",
          model_name="national_mmm",
          prices=np.ones(5),
      ),
      dict(
          testcase_name="geo_prices",
          model_name="geo_mmm",
          prices=np.array([1., 0.8, 1.2, 1.5, 0.5]),
      ),
      dict(
          testcase_name="geo_ones",
          model_name="geo_mmm",
          prices=np.ones(5),
      ),
  ])
  def test_generate_starting_values_calculates_correct_values(
      self, model_name, prices):
    mmm = getattr(self, model_name)
    n_time_periods = 10
    budget = mmm.n_media_channels * n_time_periods
    starting_values = optimize_media._generate_starting_values(
        n_time_periods=10,
        media_scaler=None,
        media=mmm.media,
        budget=budget,
        prices=prices,
    )

    # Given that data is all ones, starting values will be equal to prices.
    np.testing.assert_array_almost_equal(
        starting_values, jnp.repeat(n_time_periods, repeats=5))

if __name__ == "__main__":
  absltest.main()
