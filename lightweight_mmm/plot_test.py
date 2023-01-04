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

"""Tests for plot."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import pandas as pd

from lightweight_mmm import lightweight_mmm
from lightweight_mmm import models
from lightweight_mmm import plot
from lightweight_mmm import preprocessing

MOCK_NATIONAL_TRACE = {
    "coef_extra_features": np.ones([10, 2]),
    "coef_media": np.ones([10, 5]),
    "coef_trend": np.ones([10, 1]),
    "expo_trend": np.ones([10, 1]),
    "gamma_seasonality": np.ones([10, 3, 2]),
    "intercept": np.ones([10, 1]),
    "media_transformed": np.ones([10, 50, 5,]),
    "mu": np.ones([10, 50]),
    "sigma": np.ones([10, 1]),
    "ad_effect_retention_rate": np.ones([10, 5]),
    "exponent": np.ones([10, 5]),
    "half_max_effective_concentration": np.ones([10, 5]),
    "lag_weight": np.ones([10, 5]),
    "slope": np.ones([10, 5]),
    "peak_effect_delay": np.ones([10, 5]),
    }

MOCK_GEO_TRACE = {
    "channel_coef_media": np.ones([10, 5, 1]),
    "coef_extra_features": np.ones([10, 2, 3]),
    "coef_media": np.ones([10, 5, 3]),
    "coef_seasonality": np.ones([10, 3]),
    "coef_trend": np.ones([10, 3]),
    "expo_trend": np.ones([10, 1]),
    "gamma_seasonality": np.ones([10, 3, 2]),
    "intercept": np.ones([10, 3]),
    "media_transformed": np.ones([10, 50, 5, 3]),
    "mu": np.ones([10, 50, 3]),
    "sigma": np.ones([10, 3]),
    "ad_effect_retention_rate": np.ones([10, 5]),
    "exponent": np.ones([10, 5]),
    "half_max_effective_concentration": np.ones([10, 5]),
    "lag_weight": np.ones([10, 5]),
    "peak_effect_delay": np.ones([10, 5]),
    "slope": np.ones([10, 5]),
}


def _set_up_mock_mmm(model_name: str,
                     is_geo_model: bool) -> lightweight_mmm.LightweightMMM:
  """Creates a mock LightweightMMM instance that acts like a fitted model.

  These instances are used when we want to run tests on more diverse ranges of
  models than the two standard national_mmm and geo_mmm defined below but don't
  need the unit tests to spend time actually running the model fits.

  Args:
    model_name: One of ["adstock", "carryover", or "hill_adstock"], specifying
      which model type should be used in the mock LightweightMMM.
    is_geo_model: Whether to create a geo-level model (True) or a national-level
      model (False).

  Returns:
    mmm: A LightweightMMM object that can be treated like a fitted model
    for plotting-related unit tests.
  """
  initial_mock_trace = MOCK_GEO_TRACE if is_geo_model else MOCK_NATIONAL_TRACE
  all_model_names = {"adstock", "carryover", "hill_adstock"}
  model_items_to_delete = frozenset.union(*[
      models.TRANSFORM_PRIORS_NAMES[x]
      for x in all_model_names - {model_name}
  ]) - models.TRANSFORM_PRIORS_NAMES[model_name]
  mock_trace = {
      key: initial_mock_trace[key]
      for key in initial_mock_trace
      if key not in model_items_to_delete
  }
  mmm = lightweight_mmm.LightweightMMM(model_name=model_name)
  mmm.n_media_channels = 5
  mmm.n_geos = 3 if is_geo_model else 1
  mmm._media_prior = jnp.ones(5)
  mmm._weekday_seasonality = False
  mmm._degrees_seasonality = 3
  mmm.custom_priors = {}
  mmm._extra_features = None
  mmm.trace = mock_trace
  mmm.media = jnp.ones_like(mock_trace["media_transformed"][0])
  mmm.media_names = [f"channel_{i}" for i in range(5)]
  return mmm


class PlotTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(PlotTest, cls).setUpClass()
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
    cls.not_fitted_mmm = lightweight_mmm.LightweightMMM()

  def setUp(self):
    super().setUp()
    self.mock_ax_scatter = self.enter_context(
        mock.patch.object(plot.plt.Axes, "scatter", autospec=True))
    self.mock_sns_lineplot = self.enter_context(
        mock.patch.object(plot.sns, "lineplot", autospec=True))
    self.mock_plt_plot = self.enter_context(
        mock.patch.object(plot.plt.Axes, "plot", autospec=True))
    self.mock_plt_barplot = self.enter_context(
        mock.patch.object(plot.plt.Axes, "bar", autospec=True))
    self.mock_pd_area_plot = self.enter_context(
        mock.patch.object(plot.pd.DataFrame.plot, "area", autospec=True))
    self.mock_sns_kdeplot = self.enter_context(
        mock.patch.object(plot.sns, "kdeplot", autospec=True))
    self.mock_plt_ax_legend = self.enter_context(
        mock.patch.object(plot.plt.Axes, "legend", autospec=True))

  @parameterized.named_parameters([
      dict(
          testcase_name="national",
          media_mix_model="national_mmm",
          expected_calls=1),
      dict(testcase_name="geo", media_mix_model="geo_mmm", expected_calls=2)
  ])

  def test_plot_model_fit_plot_called_with_scaler(self, media_mix_model,
                                                  expected_calls):
    target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    target_scaler.fit(jnp.ones(1))
    mmm = getattr(self, media_mix_model)

    plot.plot_model_fit(media_mix_model=mmm, target_scaler=target_scaler)

    self.assertTrue(self.mock_plt_plot.call_count, expected_calls)

  @parameterized.named_parameters([
      dict(
          testcase_name="national",
          media_mix_model="national_mmm",
          expected_calls=1),
      dict(
          testcase_name="geo",
          media_mix_model="geo_mmm",
          expected_calls=2)
  ])
  def test_plot_model_fit_plot_called_without_scaler(
      self, media_mix_model, expected_calls):
    mmm = getattr(self, media_mix_model)

    plot.plot_model_fit(media_mix_model=mmm)

    self.assertTrue(self.mock_plt_plot.call_count, expected_calls)

  @parameterized.named_parameters([
      dict(
          testcase_name="national",
          media_mix_model="national_mmm"),
      dict(
          testcase_name="geo",
          media_mix_model="geo_mmm")
  ])
  def test_plot_response_curves_plots_n_times_with_correct_params(
      self, media_mix_model):
    mmm = getattr(self, media_mix_model)
    n_geos = (mmm.media.shape[-1] if mmm.media.ndim == 3
              else 1)
    plot.plot_response_curves(media_mix_model=mmm)

    _, call_kwargs = self.mock_sns_lineplot.call_args_list[0]
    # n channels times 2 charts.
    self.assertEqual(self.mock_sns_lineplot.call_count,
                     2 * mmm.n_media_channels)
    self.assertEqual(jnp.round(a=call_kwargs["x"].max(), decimals=4),
                     jnp.round(a=1.2 * n_geos, decimals=4))

  @parameterized.named_parameters([
      dict(
          testcase_name="national",
          media_mix_model="national_mmm"),
      dict(
          testcase_name="geo",
          media_mix_model="geo_mmm")
  ])
  def test_plot_response_curves_with_prices_plots_n_times_with_correct_params(
      self, media_mix_model="geo_mmm"):
    mmm = getattr(self, media_mix_model)
    n_channels = mmm.n_media_channels
    n_geos = mmm.media.shape[-1] if mmm.media.ndim == 3 else 1
    prices = jnp.array([1., 0.8, 2., 3., 1.])
    expected_maxes = jnp.repeat(jnp.array([1.2, 0.96, 2.4, 3.6, 1.2]), 2)

    plot.plot_response_curves(media_mix_model=mmm, prices=prices)

    calls_list = self.mock_sns_lineplot.call_args_list
    self.assertEqual(self.mock_sns_lineplot.call_count, n_channels * 2)
    for (_, call_kwargs), expected_max in zip(calls_list, expected_maxes):
      self.assertAlmostEqual(
          jnp.round(a=call_kwargs["x"].max().item(), decimals=4).item(),
          jnp.round(a=expected_max, decimals=4).item() * n_geos,
          places=4)

  @parameterized.named_parameters([
      dict(
          testcase_name="national",
          media_mix_model="national_mmm"),
      dict(
          testcase_name="geo",
          media_mix_model="geo_mmm")
  ])
  def test_plot_response_curves_produces_y_axis_starting_at_zero(
      self, media_mix_model):
    mmm = getattr(self, media_mix_model)

    plot.plot_response_curves(media_mix_model=mmm)

    calls_list = self.mock_sns_lineplot.call_args_list
    for _, call_kwargs in calls_list[:3]:
      self.assertLessEqual(call_kwargs["y"].min().item(), 0.1)
      self.assertGreaterEqual(call_kwargs["y"].min().item(), -0.1)

  @parameterized.named_parameters([
      dict(
          testcase_name="national",
          media_mix_model="national_mmm"),
      dict(
          testcase_name="geo",
          media_mix_model="geo_mmm")
  ])
  def test_plot_response_curves_scales_with_media_scaler(
      self, media_mix_model):
    media_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    factor = 5
    media_scaler.fit(jnp.ones(1) * factor)
    expected_maxes = jnp.repeat(
        jnp.repeat(jnp.array([1.2]), repeats=5),
        repeats=2)
    mmm = getattr(self, media_mix_model)

    plot.plot_response_curves(media_mix_model=mmm,
                              media_scaler=media_scaler)

    calls_list = self.mock_plt_plot.call_args_list
    for (_, call_kwargs), expected_max in zip(calls_list, expected_maxes):
      self.assertAlmostEqual(call_kwargs["x"].max().item(),
                             expected_max * factor,
                             places=4)

  @parameterized.named_parameters([
      dict(
          testcase_name="national",
          media_mix_model="national_mmm"),
      dict(
          testcase_name="geo",
          media_mix_model="geo_mmm")
  ])
  def test_plot_response_curves_scales_with_target_scaler(
      self, media_mix_model):
    target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    factor = 5
    target_scaler.fit(jnp.ones(1) * factor)
    mmm = getattr(self, media_mix_model)

    plot.plot_response_curves(media_mix_model=mmm,
                              target_scaler=target_scaler)

    calls_list = self.mock_plt_plot.call_args_list
    for _, call_kwargs in calls_list:
      self.assertAlmostEqual(call_kwargs["y"].max().item(),
                             1 * factor,
                             places=4)

  def test_perfect_correlation_returns_correct_output(self):
    x = jnp.arange(100)
    y = jnp.arange(100, 200)

    idx, maxcorr = plot.plot_cross_correlate(x, y)

    self.assertEqual(idx, 0)
    self.assertEqual(maxcorr, 1)

  def test_var_cost_plot_called_with_correct_kwargs(self):
    media = jnp.arange(10).reshape((5, 2))
    costs = [1, 2]
    names = ["a", "b"]
    std = jnp.repeat(2.82842712, 2)
    means = jnp.array([4, 5])
    expected_coef_of_variation = std / means

    _ = plot.plot_var_cost(media, costs, names)
    _, call_kwargs = self.mock_ax_scatter.call_args_list[0]

    np.testing.assert_array_almost_equal(call_kwargs["x"], costs)
    np.testing.assert_array_almost_equal(call_kwargs["y"],
                                         expected_coef_of_variation)

  @parameterized.named_parameters([
      dict(
          testcase_name="national",
          media_mix_model="national_mmm"),
      dict(
          testcase_name="geo",
          media_mix_model="geo_mmm")
  ])
  def test_plot_media_channel_posteriors_plots_right_number_subplots(
      self, media_mix_model):
    mmm = getattr(self, media_mix_model)
    n_channels = mmm.n_media_channels
    n_geos = mmm.media.shape[-1] if mmm.media.ndim == 3 else 1

    fig = plot.plot_media_channel_posteriors(media_mix_model=mmm)

    self.assertLen(fig.get_axes(), n_channels * n_geos)

  def test_unequal_length_ground_truth_and_predictions_raises_error(self):
    prediction = jnp.arange(10).reshape((5, 2))
    ground_truth = jnp.array([1, 2, 3])
    with self.assertRaises(ValueError):
      plot.plot_out_of_sample_model_fit(prediction, ground_truth)

  @parameterized.named_parameters([
      dict(
          testcase_name="national",
          media_mix_model="national_mmm",
          expected_calls=3),
      dict(testcase_name="geo", media_mix_model="geo_mmm", expected_calls=3)
  ])
  def test_plot_pre_post_budget_allocation_comparison_n_times_with_correct_params(
      self, media_mix_model, expected_calls):
    mmm = getattr(self, media_mix_model)
    kpi_with_optim = -503
    kpi_without_optim = -479
    optimal_buget_allocation = jnp.array([118, 278, 100, 100, 100])
    previous_budget_allocation = jnp.array([199, 197, 100, 100, 100])

    plot.plot_pre_post_budget_allocation_comparison(
        media_mix_model=mmm,
        kpi_with_optim=kpi_with_optim,
        kpi_without_optim=kpi_without_optim,
        optimal_buget_allocation=optimal_buget_allocation,
        previous_budget_allocation=previous_budget_allocation)

    self.assertEqual(self.mock_plt_barplot.call_count, expected_calls)
    call_list = self.mock_plt_barplot.call_args_list

    np.testing.assert_array_almost_equal(call_list[0][0][-1],
                                         previous_budget_allocation)
    np.testing.assert_array_almost_equal(call_list[1][0][-1],
                                         optimal_buget_allocation)
    np.testing.assert_array_almost_equal(
        call_list[2][0][-1], [kpi_without_optim * -1, kpi_with_optim * -1])

  @parameterized.named_parameters([
      dict(testcase_name="national and geo", media_mix_model="not_fitted_mmm")
  ])
  def test_plot_pre_post_budget_allocation_comparison_raise_notfittedmodelerror(
      self, media_mix_model):
    mmm = getattr(self, media_mix_model)
    kpi_with_optim = -503
    kpi_without_optim = -479
    optimal_buget_allocation = jnp.array([118, 278, 100, 100, 100])
    previous_budget_allocation = jnp.array([199, 197, 100, 100, 100])
    with self.assertRaises(lightweight_mmm.NotFittedModelError):
      plot.plot_pre_post_budget_allocation_comparison(
          media_mix_model=mmm,
          kpi_with_optim=kpi_with_optim,
          kpi_without_optim=kpi_without_optim,
          optimal_buget_allocation=optimal_buget_allocation,
          previous_budget_allocation=previous_budget_allocation)

  @parameterized.named_parameters(
      dict(
          testcase_name="no channel_name input",
          media_mix_model="not_fitted_mmm",
          media_spend=np.ones((50, 5)),
          channel_names=None),
      dict(
          testcase_name="channel_name input",
          media_mix_model="not_fitted_mmm",
          media_spend=np.ones((50, 5)),
          channel_names=[f"channel_{x}" for x in range(5)]))
  def test_create_attribution_over_spend_fractions_raise_notfittedmodelerror(
      self, media_mix_model, media_spend, channel_names):
    with self.assertRaises(lightweight_mmm.NotFittedModelError):
      plot.create_attribution_over_spend_fractions(
          media_mix_model=getattr(self, media_mix_model),
          media_spend=media_spend,
          channel_names=channel_names)

  @parameterized.named_parameters([
      dict(
          testcase_name="negative_number",
          media_mix_model="national_mmm",
          media_spend=-np.ones((50, 5))),
      dict(
          testcase_name="aggeregated_zero",
          media_mix_model="national_mmm",
          media_spend=np.zeros((50, 5)))
  ])
  def test_create_attribution_over_spend_fractions_raise_error_on_invalid_values(
      self, media_mix_model, media_spend):
    expected_message = ("Values in media must all be non-negative or values in "
                        "aggregated media must be possitive.")
    with self.assertRaisesRegex(ValueError, expected_message):
      plot.create_attribution_over_spend_fractions(
          getattr(self, media_mix_model), media_spend)

  @parameterized.product(
      (dict(is_geo_model=False, media_spend=np.array([1, 2, 3, 4, 5])),
       dict(
           is_geo_model=False,
           media_spend=np.resize(np.array([1, 2, 3, 4, 5]), 250).reshape(50,
                                                                         5)),),
      (dict(channel_names=None),
       dict(channel_names=[f"channel_{x}" for x in range(5)])),
      (dict(time_index=None), dict(time_index=(1, 10))),
      (dict(model_name="adstock"), dict(model_name="carryover"),
       dict(model_name="hill_adstock")))
  def test_create_attribution_over_spend_fractions_results_are_correct(
      self, model_name, is_geo_model, media_spend, channel_names, time_index):
    mmm = _set_up_mock_mmm(model_name, is_geo_model)

    expected_results = pd.DataFrame(
        np.transpose([np.ones(5)/np.ones(5).sum(),
                      np.arange(1, 6)/np.arange(1, 6).sum(),
                      3/np.arange(1, 6)]),
        index=[f"channel_{x}" for x in range(5)],
        columns=["media attribution", "media spend", "attribution over spend"])

    aos_fractions_df = plot.create_attribution_over_spend_fractions(
        mmm, media_spend, channel_names, time_index)

    pd.testing.assert_frame_equal(aos_fractions_df, expected_results, atol=1e-3)

  def test_create_media_baseline_contribution_df_raise_notfittedmodelerror(
      self):
    with self.assertRaises(lightweight_mmm.NotFittedModelError):
      plot.create_media_baseline_contribution_df(
          media_mix_model=getattr(self, "not_fitted_mmm"))

  @parameterized.named_parameters([
      dict(testcase_name="national", media_mix_model="national_mmm"),
      dict(testcase_name="geo", media_mix_model="geo_mmm")
  ])
  def test_create_media_baseline_contribution_df_contributions_add_up_avg_prediction(
      self, media_mix_model):
    mmm = getattr(self, media_mix_model)
    target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    target_scaler.fit(jnp.ones(1))
    media_channels_baseline_contribution_df = plot.create_media_baseline_contribution_df(
        media_mix_model=mmm, target_scaler=target_scaler)
    contribution_pct_cols = [
        col for col in media_channels_baseline_contribution_df.columns
        if "percentage" in col
    ]
    contribution_cols = [
        col for col in media_channels_baseline_contribution_df.columns
        if "contribution" in col
    ]
    # test whether contribution percentage sums up to 1 for each period
    np.testing.assert_array_almost_equal(
        media_channels_baseline_contribution_df[contribution_pct_cols].sum(
            axis=1),
        jnp.repeat(1, media_channels_baseline_contribution_df.shape[0]))
    # test whether contribution volume sums up to avg predition for each period
    np.testing.assert_array_almost_equal(
        np.round(
            media_channels_baseline_contribution_df[contribution_cols].sum(
                axis=1), 0),
        np.round(media_channels_baseline_contribution_df["avg_prediction"], 0))

  def test_create_media_baseline_contribution_df_returns_accurate_contribution_pct(
      self):
    mmm_object = lightweight_mmm.LightweightMMM()
    mmm_object.media = jnp.ones((1, 3))
    mmm_object._total_costs = jnp.array([2., 1., 3.]) * 15
    mmm_object._target = jnp.ones((1, 1)) * 5
    mmm_object.media_names = ["channel_0", "channel_1", "channel_2"]
    mmm_object.trace = {
        "media_transformed": jnp.ones((500, 1, 3)) * jnp.arange(1, 4),
        "mu": jnp.ones((500, 1)) * 10,
        "coef_media": jnp.ones((500, 3)) * 0.5
    }
    expected_contribution_pct = jnp.array([0.05, 0.1, 0.15])

    contribution_df = plot.create_media_baseline_contribution_df(
        media_mix_model=mmm_object)
    contribution_percentage_cols = [
        "{}_percentage".format(col) for col in mmm_object.media_names
    ]
    np.testing.assert_array_almost_equal(
        expected_contribution_pct,
        contribution_df[contribution_percentage_cols].values.flatten().tolist())

  def test_create_media_baseline_contribution_df_returns_non_nan_value_for_media_contribution(
      self):
    mmm_object = lightweight_mmm.LightweightMMM()
    mmm_object.media = jnp.concatenate([jnp.ones((1, 3)), jnp.zeros((1, 3))])
    mmm_object._total_costs = jnp.array([2., 1., 3.]) * 15
    mmm_object._target = jnp.ones((2, 1)) * 5
    mmm_object.media_names = ["channel_0", "channel_1", "channel_2"]
    mmm_object.trace = {
        "media_transformed": jnp.concatenate(
            [jnp.ones((20, 1, 3)), jnp.zeros((20, 1, 3))], axis=1
            ) * jnp.arange(1, 4),
        "mu": jnp.concatenate(
            [jnp.ones((20, 1)), jnp.zeros((20, 1))-1], axis=1)  * 10,
        "coef_media": jnp.ones((20, 3))
    }
    expected_contribution_pct = jnp.array([0.1, 0.2, 0.3, 0, 0, 0])
    # Columns want to be tested whether they will return nan value.
    contribution_percentage_cols = [
        "{}_percentage".format(col) for col in mmm_object.media_names
    ]

    contribution_df = plot.create_media_baseline_contribution_df(
        media_mix_model=mmm_object)

    np.testing.assert_array_almost_equal(
        expected_contribution_pct,
        contribution_df[contribution_percentage_cols].values.flatten().tolist())

  @parameterized.named_parameters([
      dict(
          testcase_name="national",
          media_mix_model="national_mmm",
          expected_calls=1),
      dict(testcase_name="geo", media_mix_model="geo_mmm", expected_calls=1)
  ])
  def test_plot_media_baseline_contribution_area_plot(self, media_mix_model,
                                                      expected_calls):
    mmm = getattr(self, media_mix_model)
    target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    target_scaler.fit(jnp.ones(1))
    _ = plot.plot_media_baseline_contribution_area_plot(
        media_mix_model=mmm, target_scaler=target_scaler, legend_outside=True)
    self.assertEqual(self.mock_pd_area_plot.call_count, expected_calls)

  def test_legend_plot_media_baseline_contribution_area_plot(self):
    mmm = getattr(self, "national_mmm")
    target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    target_scaler.fit(jnp.ones(1))
    _ = plot.plot_media_baseline_contribution_area_plot(
        media_mix_model=mmm, target_scaler=target_scaler, legend_outside=True)
    call_args, call_kwargs = self.mock_plt_ax_legend.call_args_list[0]
    self.assertEqual(call_kwargs["loc"], "center left")
    self.assertEqual(call_kwargs["bbox_to_anchor"], (1, 0.5))

    mmm_object = lightweight_mmm.LightweightMMM()
    mmm_object.fit(
        media=np.ones((50, 5)),
        target=np.ones(50),
        media_prior=np.ones(5) * 50,
        number_warmup=2,
        number_samples=2,
        number_chains=1)
    _ = plot.plot_response_curves(media_mix_model=mmm_object)
    fig = plot.plot_response_curves(media_mix_model=mmm_object)
    self.assertIsInstance(fig, matplotlib.figure.Figure)

  @parameterized.named_parameters([
      dict(testcase_name="trace_missing", missing_attribute="trace"),
      dict(
          testcase_name="weekday_seasonality_missing",
          missing_attribute="_weekday_seasonality"),
      dict(
          testcase_name="custom_priors_missing",
          missing_attribute="custom_priors"),
      dict(testcase_name="n_geos_missing", missing_attribute="n_geos"),
      dict(
          testcase_name="n_media_channels_missing",
          missing_attribute="n_media_channels"),
      dict(
          testcase_name="media_prior_missing",
          missing_attribute="_media_prior"),
  ])
  def test_prior_posterior_plot_raises_notfittedmodelerror(
      self, missing_attribute):
    mmm = getattr(self, "not_fitted_mmm")
    mmm.trace = jnp.ones((50, 5))
    mmm._weekday_seasonality = True
    mmm.custom_priors = None
    mmm.n_geos = 1
    mmm.n_media_channels = 3
    mmm._media_prior = jnp.ones(5) * 50

    with self.assertRaises(lightweight_mmm.NotFittedModelError):
      delattr(mmm, missing_attribute)
      plot.plot_prior_and_posterior(
          media_mix_model=mmm, number_of_samples_for_prior=100)

  @parameterized.named_parameters([
      dict(
          testcase_name="half_normal_prior",
          prior_distribution=dist.HalfNormal(3),
          expected_clipping_bounds=[0, None]),
      dict(
          testcase_name="normal_prior",
          prior_distribution=dist.Normal(0, 1),
          expected_clipping_bounds=None),
      dict(
          testcase_name="beta_prior",
          prior_distribution="beta distribution placeholder",
          expected_clipping_bounds=[0, 1]),
      dict(
          testcase_name="gamma_prior",
          prior_distribution=dist.Gamma(0.5),
          expected_clipping_bounds=[0, None]),
  ])
  def test_prior_posterior_plot_clipping_bounds_for_kdeplots(
      self, prior_distribution, expected_clipping_bounds):
    fig = plt.figure()
    gridspec_fig = matplotlib.gridspec.GridSpec(
        nrows=1, ncols=1, figure=fig, hspace=10)

    # dist.Beta() calls jnp.broadcast_to() upon instantiation, so this
    # distribution can't be left in the decorator or it raises a
    # RuntimeError: "Attempted call to JAX before absl.app.run() is called".
    # Thus we have to instantiate it inside the unit test instead.
    if prior_distribution == "beta distribution placeholder":
      prior_distribution = dist.Beta(0.5, 0.5)

    plot._make_prior_and_posterior_subplot_for_one_feature(
        prior_distribution=prior_distribution,
        posterior_samples=jnp.ones(50),
        subplot_title="title",
        fig=fig,
        gridspec_fig=gridspec_fig,
        i_ax=0, number_of_samples_for_prior=100)
    call_details = self.mock_sns_kdeplot.call_args_list
    called_clipping_bounds = call_details[0][1]["clip"]

    self.assertEqual(called_clipping_bounds, expected_clipping_bounds)

  @parameterized.product(
      (dict(
          is_geo_model=False,
          has_extra_features=False,
          extra_expected_number_of_subplots=0),
       dict(
           is_geo_model=False,
           has_extra_features=True,
           extra_expected_number_of_subplots=2),
       dict(
           is_geo_model=True,
           has_extra_features=False,
           extra_expected_number_of_subplots=14),
       dict(
           is_geo_model=True,
           has_extra_features=True,
           extra_expected_number_of_subplots=20)),
      (dict(model_name="adstock", base_expected_number_of_subplots=25),
       dict(model_name="carryover", base_expected_number_of_subplots=30),
       dict(model_name="hill_adstock", base_expected_number_of_subplots=30)))
  def test_prior_posterior_plot_makes_correct_number_of_subplots(
      self, model_name, is_geo_model, has_extra_features,
      base_expected_number_of_subplots, extra_expected_number_of_subplots):
    expected_number_of_subplots = (
        base_expected_number_of_subplots + extra_expected_number_of_subplots)
    mmm = _set_up_mock_mmm(model_name=model_name, is_geo_model=is_geo_model)
    if not has_extra_features:
      del mmm.trace["coef_extra_features"]
    mmm._extra_features = jnp.ones_like(
        mmm.trace["coef_extra_features"][0]) if has_extra_features else None

    plot.plot_prior_and_posterior(
        media_mix_model=mmm, number_of_samples_for_prior=10, seed=0)

    # each subplot gets two calls, one for the prior and one for the posterior
    number_of_subplots_created = self.mock_sns_kdeplot.call_count / 2
    self.assertEqual(number_of_subplots_created, expected_number_of_subplots)

  @parameterized.product(
      (dict(is_geo_model=True, expected_number_of_subplots=11),
       dict(is_geo_model=False, expected_number_of_subplots=7)),
      (dict(model_name="adstock",
            selected_features=["sigma", "intercept", "exponent"]),
       dict(model_name="carryover",
            selected_features=["sigma", "intercept", "exponent"]),
       dict(model_name="hill_adstock",
            selected_features=["sigma", "intercept", "slope"]))
      )
  def test_selected_features_for_prior_posterior_plot_makes_correct_number_of_subplots(
      self, model_name, selected_features, is_geo_model,
      expected_number_of_subplots):
    mmm = _set_up_mock_mmm(model_name=model_name, is_geo_model=is_geo_model)

    plot.plot_prior_and_posterior(
        media_mix_model=mmm,
        number_of_samples_for_prior=10,
        seed=0,
        selected_features=selected_features)

    # each subplot gets two calls, one for the prior and one for the posterior
    number_of_subplots_created = self.mock_sns_kdeplot.call_count / 2
    self.assertEqual(number_of_subplots_created, expected_number_of_subplots)

  @parameterized.named_parameters([
      dict(
          testcase_name="national_model",
          model_name="national_mmm"),
      dict(
          testcase_name="geo_model",
          model_name="geo_mmm"),
  ])
  def test_selected_features_raises_value_error(self, model_name):
    mmm = getattr(self, model_name)

    # The expected error message here is an f-string whose value is filled in by
    # a set that contains a single element. The message should look exactly as
    # it appears below; no special regex parsing should be happening here.
    expected_error = ("Selected_features {'misspelled_feature'}"
                      " not in media_mix_model.")

    with self.assertRaisesRegex(ValueError, expected_regex=expected_error):
      plot.plot_prior_and_posterior(
          media_mix_model=mmm,
          number_of_samples_for_prior=10,
          selected_features=[
              "sigma", "intercept", "misspelled_feature"
          ])

if __name__ == "__main__":
  absltest.main()
