# Copyright 2021 Google LLC.
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

  def setUp(self):
    super().setUp()
    self.mock_plt_scatter = self.enter_context(
        mock.patch.object(plot.plt, "scatter", autospec=True))
    self.mock_sns_lineplot = self.enter_context(
        mock.patch.object(plot.sns, "lineplot", autospec=True))
    self.mock_plt_plot = self.enter_context(
        mock.patch.object(plot.plt.Axes, "plot", autospec=True))

  def test_plot_model_fit_plot_called_with_scaler(self):
    target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    target = target_scaler.fit_transform(jnp.ones(50))
    mmm = lightweight_mmm.LightweightMMM()
    mmm.fit(
        media=jnp.ones((50, 3)),
        target=target,
        costs=jnp.repeat(50, 3),
        number_warmup=50,
        number_samples=50,
        number_chains=1)

    plot.plot_model_fit(media_mix_model=mmm, target_scaler=target_scaler)

    self.assertTrue(self.mock_plt_plot.called)

  def test_plot_model_fit_plot_called_without_scaler(self):
    mmm = lightweight_mmm.LightweightMMM()
    mmm.fit(
        media=jnp.ones((50, 3)),
        target=jnp.ones(50),
        costs=jnp.repeat(50, 3),
        number_warmup=50,
        number_samples=50,
        number_chains=1)

    plot.plot_model_fit(media_mix_model=mmm)

    self.assertTrue(self.mock_plt_plot.called)

  @parameterized.named_parameters([
      dict(testcase_name="five_channels", n_channels=5),
      dict(testcase_name="three_channel", n_channels=3),
      dict(testcase_name="ten_channels", n_channels=10),
  ])
  def test_plot_curve_response_plots_n_times_with_correct_params(
      self, n_channels):
    mmm = lightweight_mmm.LightweightMMM()
    mmm.fit(
        media=jnp.ones((50, n_channels)),
        target=jnp.ones(50),
        costs=jnp.repeat(50, n_channels),
        number_warmup=50,
        number_samples=50,
        number_chains=1)

    plot.plot_curve_response(media_mix_model=mmm)

    _, call_kwargs = self.mock_sns_lineplot.call_args_list[0]
    self.assertEqual(self.mock_sns_lineplot.call_count, n_channels)
    self.assertEqual(call_kwargs["x"].max(), 1.1)

  def test_perfect_correlation_returns_correct_output(self):
    x = np.arange(100)
    y = np.arange(100, 200)
    idx, maxcorr = plot.plot_cross_correlate(x, y)
    self.assertEqual(idx, 0)
    self.assertEqual(maxcorr, 1)

  def test_var_cost_plot_called_with_correct_kwargs(self):
    media = np.arange(10).reshape((5, 2))
    costs = [1, 2]
    names = ["a", "b"]
    std = np.repeat(2.82842712, 2)
    means = np.array([4, 5])
    expected_coef_of_variation = std / means

    plot.plot_var_cost(media, costs, names)
    _, call_kwargs = self.mock_plt_scatter.call_args_list[0]

    np.testing.assert_array_almost_equal(call_kwargs["x"], costs)
    np.testing.assert_array_almost_equal(call_kwargs["y"],
                                         expected_coef_of_variation)


if __name__ == "__main__":
  absltest.main()
