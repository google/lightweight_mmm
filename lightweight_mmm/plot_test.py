# Copyright 2022 Google LLC.
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
import numpy as np

from lightweight_mmm import lightweight_mmm
from lightweight_mmm import plot
from lightweight_mmm import preprocessing


class PlotTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(PlotTest, cls).setUpClass()
    cls.national_mmm = lightweight_mmm.LightweightMMM()
    cls.national_mmm.fit(
        media=jnp.ones((50, 5)),
        target=jnp.ones(50),
        total_costs=jnp.ones(5) * 50,
        number_warmup=2,
        number_samples=2,
        number_chains=1)
    cls.geo_mmm = lightweight_mmm.LightweightMMM()
    cls.geo_mmm.fit(
        media=jnp.ones((50, 5, 3)),
        target=jnp.ones((50, 3)),
        total_costs=jnp.ones(5) * 50,
        number_warmup=2,
        number_samples=2,
        number_chains=1)

  def setUp(self):
    super().setUp()
    self.mock_ax_scatter = self.enter_context(
        mock.patch.object(plot.plt.Axes, "scatter", autospec=True))
    self.mock_sns_lineplot = self.enter_context(
        mock.patch.object(plot.sns, "lineplot", autospec=True))
    self.mock_plt_plot = self.enter_context(
        mock.patch.object(plot.plt.Axes, "plot", autospec=True))

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
  def test_plot_model_fit_plot_called_with_scaler(
      self, media_mix_model, expected_calls):
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
      self.assertEqual(call_kwargs["y"].min().item(), 0)

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


if __name__ == "__main__":
  absltest.main()
