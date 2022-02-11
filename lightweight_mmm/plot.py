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

"""Plotting functions pre and post model fitting."""

import functools
from typing import Any, List, Optional, Sequence, Tuple

import arviz
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics

from lightweight_mmm.lightweight_mmm import lightweight_mmm
from lightweight_mmm.lightweight_mmm import preprocessing


@functools.partial(jax.jit, static_argnames=("media_mix_model"))
def _make_single_prediction(media_mix_model: lightweight_mmm.LightweightMMM,
                            mock_media: jnp.ndarray,
                            extra_features: Optional[jnp.ndarray]
                            ) -> jnp.ndarray:
  """Makes a prediction of a single row.

  Serves as a helper function for making predictions individually for each media
  channel and one row at a time. It is meant to be used vmaped otherwise it can
  be slow as it's meant to be used for plotting curve responses only. Use
  lightweight_mmm.LightweightMMM for regular predict functionality.

  Args:
    media_mix_model: Media mix model to use for getting the predictions.
    mock_media: Mock media for this iteration of predictions.
    extra_features: Extra features to use for predictions.

  Returns:
    A point estimate for the given data.
  """
  return media_mix_model.predict(
      jnp.expand_dims(mock_media, axis=0), extra_features).mean(axis=0)


@functools.partial(
    jax.jit,
    static_argnames=("media_mix_model", "extra_features", "target_scaler",
                     "apply_log_scale"))
def _generate_diagonal_predictions(
    media_mix_model: lightweight_mmm.LightweightMMM,
    media_values: jnp.ndarray,
    extra_features: Optional[jnp.ndarray],
    target_scaler: Optional[preprocessing.CustomScaler],
    prediction_offset: jnp.ndarray):
  """Generates predictions for one value per channel leaving the rest to zero.

  This function does the following steps:
    - Vmaps the single prediction function on axis=0 of the media arg.
    - Diagonalizes the media input values so that each value is represented
      along side zeros on for the rest of the channels.
    - Generate predictions.
    - Unscale prediction if target_scaler is given.

  Args:
    media_mix_model: Media mix model to use for plotting the response curves.
    media_values: Media values.
    extra_features: Extra features values.
    target_scaler: Scaler used for scaling the target, to unscaled values and
      plot in the original scale.
    prediction_offset: The value of a prediction of an all zero media input.

  Returns:
    The predictions for the given data.
  """
  make_predictions = jax.vmap(_make_single_prediction, in_axes=(None, 0, None))
  diag_media_values = jnp.eye(media_values.shape[0]) * media_values
  predictions = make_predictions(
      media_mix_model,
      diag_media_values,
      extra_features) - prediction_offset
  if target_scaler:
    predictions = target_scaler.inverse_transform(predictions)
  return predictions


def _calculate_number_rows_plot(n_media_channels: int, n_columns: int):
  """Calculates the number of rows of plots needed to fit n + 1 plots in n_cols.

  Args:
    n_media_channels: Number of media channels. The total of plots needed is
      n_media_channels + 1.
    n_columns: Number of columns in the plot grid.

  Returns:
    The number of rows of plots needed to fit n + 1 plots in n cols
  """
  if n_media_channels % n_columns == 0:
    return n_media_channels // n_columns + 1
  return n_media_channels // n_columns + 2


def plot_response_curves(
    media_mix_model: lightweight_mmm.LightweightMMM,
    media_scaler: Optional[preprocessing.CustomScaler] = None,
    target_scaler: Optional[preprocessing.CustomScaler] = None,
    prices: jnp.ndarray = None,
    optimal_allocation_per_timeunit: jnp.ndarray = None,
    steps: int = 50,
    percentage_add: float = 0.2,
    apply_log_scale: bool = False,
    figure_size: Tuple[int, int] = (10, 14),
    n_columns: int = 3,
    marker_size: int = 8,
    legend_fontsize: int = 8) -> matplotlib.figure.Figure:
  """Plots the response curves of each media channel based on the model.

  It plots an individual subplot for each media channel. If '
  optimal_allocation_per_timeunit is given it uses it to add markers based on
  historic average spend and the given optimal one on each of the individual
  subplots.

  It then plots a combined plot with all the response curves which can be
  changed to log scale if apply_log_scale is True.

  Args:
    media_mix_model: Media mix model to use for plotting the response curves.
    media_scaler: Scaler that was used to scale the media data before training.
    target_scaler: Scaler used for scaling the target, to unscaled values and
      plot in the original scale.
    prices: Prices to translate the media units to spend. If all your data is
      already in spend numbers you can leave this as None. If some of your data
      is media spend and others is media unit, leave the media spend with price
      1 and add the price to the media unit channels.
    optimal_allocation_per_timeunit: Optimal allocation per time unit per media
      channel. This can be obtained by running the optimization provided by
      LightweightMMM.
    steps: Number of steps to simulate.
    percentage_add: Percentage too exceed the maximum historic spend for the
      simulation of the response curve.
    apply_log_scale: Whether to apply the log scale to the predictions (Y axis).
      When some media channels have very large scale compare to others it might
      be useful to use apply_log_scale=True. Default is False.
    figure_size: Size of the plot figure.
    n_columns: Number of columns to display in the subplots grid. Modifying this
      parameter might require to adjust figure_size accordingly for the plot
      to still have reasonable structure.
    marker_size: Size of the marker for the optimization annotations. Only
      useful if optimal_allocation_per_timeunit is not None. Default is 8.
    legend_fontsize: Legend font size for individual subplots.

  Returns:
    Plots of response curves.
  """
  if not hasattr(media_mix_model, "trace"):
    raise lightweight_mmm.NotFittedModelError(
        "Model needs to be fit first before attempting to plot its response "
        "curves.")
  media = media_mix_model.media
  media_maxes = media.max(axis=0) * (1 + percentage_add)
  if media_mix_model._extra_features is not None:
    extra_features = jnp.expand_dims(
        media_mix_model._extra_features.mean(axis=0), axis=0)
  else:
    extra_features = None
  media_ranges = jnp.expand_dims(
      jnp.linspace(start=0, stop=media_maxes, num=steps), axis=0)

  make_predictions = jax.vmap(
      jax.vmap(_make_single_prediction, in_axes=(None, 0, None)),
      in_axes=(None, 0, None))
  diagonal = jnp.repeat(
      jnp.eye(media_mix_model.n_media_channels), steps,
      axis=0).reshape(media_mix_model.n_media_channels, steps,
                      media_mix_model.n_media_channels)

  prediction_offset = media_mix_model.predict(
      media=jnp.zeros((1, *media.shape[1:])),
      extra_features=extra_features).mean(axis=0)
  mock_media = media_ranges * diagonal
  predictions = jnp.squeeze(a=make_predictions(media_mix_model,
                                               mock_media,
                                               extra_features))
  predictions = jnp.transpose(predictions) - prediction_offset
  if target_scaler:
    predictions = target_scaler.inverse_transform(predictions)

  if media_scaler:
    media_ranges = media_scaler.inverse_transform(media_ranges)

  if prices is not None:
    media_ranges *= prices

  if optimal_allocation_per_timeunit is not None:
    average_allocation = media_mix_model.media.mean(axis=0)
    average_allocation_predictions = _generate_diagonal_predictions(
        media_mix_model=media_mix_model,
        media_values=average_allocation,
        extra_features=extra_features,
        target_scaler=target_scaler,
        prediction_offset=prediction_offset)
    optimal_allocation_predictions = _generate_diagonal_predictions(
        media_mix_model=media_mix_model,
        media_values=optimal_allocation_per_timeunit,
        extra_features=extra_features,
        target_scaler=target_scaler,
        prediction_offset=prediction_offset)

    if media_scaler:
      average_allocation = media_scaler.inverse_transform(average_allocation)
      optimal_allocation_per_timeunit = media_scaler.inverse_transform(
          optimal_allocation_per_timeunit)
    if prices is not None:
      optimal_allocation_per_timeunit *= prices
      average_allocation *= prices

  fig = plt.figure(media_mix_model.n_media_channels + 1,
                   figsize=figure_size,
                   tight_layout=True)
  n_rows = _calculate_number_rows_plot(
      n_media_channels=media_mix_model.n_media_channels, n_columns=n_columns)
  last_ax = fig.add_subplot(n_rows, 1, n_rows)
  for i in range(media_mix_model.n_media_channels):
    ax = fig.add_subplot(n_rows, n_columns, i + 1)
    sns.lineplot(
        x=jnp.squeeze(media_ranges)[:, i],
        y=predictions[:, i],
        label=media_mix_model.media_names[i],
        color=sns.color_palette()[i],
        ax=ax)
    sns.lineplot(
        x=jnp.squeeze(media_ranges)[:, i],
        y=jnp.log(predictions[:, i]) if apply_log_scale else predictions[:, i],
        label=media_mix_model.media_names[i],
        color=sns.color_palette()[i],
        ax=last_ax)
    if optimal_allocation_per_timeunit is not None:
      ax.plot(
          average_allocation[i],
          average_allocation_predictions[i],
          marker="o",
          markersize=marker_size,
          label="avg_spend",
          color=sns.color_palette()[i])
      ax.plot(
          optimal_allocation_per_timeunit[i],
          optimal_allocation_predictions[i],
          marker="x",
          markersize=marker_size + 2,
          label="optimal_spend",
          color=sns.color_palette()[i])
    ax.set_ylabel("KPI")
    ax.set_xlabel("Normalized Spend" if not media_scaler else "Spend")
    ax.legend(fontsize=legend_fontsize)

  fig.suptitle("Response curves", fontsize=20)
  last_ax.set_ylabel("KPI" if not apply_log_scale else "log(KPI)")
  last_ax.set_xlabel("Normalized spend per channel"
                     if not media_scaler else "Spend per channel")
  plt.close()
  return fig


def plot_cross_correlate(feature: jnp.ndarray,
                         target: jnp.ndarray,
                         maxlags: int = 10) -> Tuple[int, float]:
  """Plots the cross correlation coefficients between 2 vectors.

  In the chart look for positive peaks, this shows how the lags of the feature
  lead the target.

  Args:
    feature: Vector, the lags of which predict target.
    target: Vector, what is predicted.
    maxlags: Maximum number of lags.

  Returns:
    Lag index and corresponding correlation of the peak correlation.

  Raises:
    ValueError: If inputs don't have same length.
  """
  if len(feature) != len(target):
    raise ValueError("feature and target need to have the same length.")
  maxlags = jnp.minimum(len(feature) - 1, maxlags)
  mean_feature, mean_target = feature.mean(), target.mean()
  plot = plt.xcorr(
      x=feature - mean_feature, y=target - mean_target, maxlags=maxlags)
  plt.show()
  maxidx = plot[1][plot[0] <= 0].argmax()
  return plot[0][maxidx], plot[1][maxidx]


def plot_var_cost(media: jnp.ndarray, costs: jnp.ndarray,
                  names: List[str]) -> matplotlib.figure.Figure:
  """Plots a a chart between the coefficient of variation and cost.

  Args:
    media: Media matrix.
    costs: Cost vector.
    names: List of variable names.

  Returns:
    Plot of coefficient of variation and cost.

  Raises:
    ValueError if inputs don't conform to same length.
  """
  if media.shape[1] != len(costs):
    raise ValueError("media columns and costs needs to have same length.")
  if media.shape[1] != len(names):
    raise ValueError("media columns and names needs to have same length.")
  coef_of_variation = media.std(axis=0) / media.mean(axis=0)

  fig, ax = plt.subplots(1, 1)
  ax.scatter(x=costs, y=coef_of_variation)
  # https://queirozf.com/entries/add-labels-and-text-to-matplotlib-plots-annotation-examples.
  for i in range(len(costs)):
    x, y, label = costs[i], coef_of_variation[i], names[i]
    ax.annotate(text=label, xy=(x, y))
  ax.set_xlabel("Cost")
  ax.set_ylabel("Coef of Variation")
  plt.close()
  return fig


def _create_shaded_line_plot(prediction: jnp.ndarray,
                             true_values: jnp.ndarray,
                             interval_mid_range: float = .9,
                             digits: int = 3) -> matplotlib.figure.Figure:
  """Creates a plot of ground truth, predicted value and credibility interval.

  Args:
    prediction: 2d array of predicted values.
    true_values: Array of true values. Must be same length as prediction.
    interval_mid_range: Mid range interval to take for plotting. Eg. .9 will use
      .05 and .95 as the lower and upper quantiles. Must be a float number
      between 0 and 1.
    digits: Number of decimals to display on metrics in the plot.

  Returns:
      Plot of model fit.
  """
  if prediction.shape[1] != len(true_values):
    raise ValueError(
        "Predicted data and ground-truth data must have same length.")
  upper_quantile = 1 - (1 - interval_mid_range) / 2
  lower_quantile = (1 - interval_mid_range) / 2
  upper_bound = jnp.quantile(a=prediction, q=upper_quantile, axis=0)
  lower_bound = jnp.quantile(a=prediction, q=lower_quantile, axis=0)

  r2, _ = arviz.r2_score(y_true=true_values, y_pred=prediction)
  mape = 100 * metrics.mean_absolute_percentage_error(
      y_true=true_values, y_pred=prediction.mean(axis=0))
  fig, ax = plt.subplots(1, 1)
  ax.plot(jnp.arange(true_values.shape[0]), true_values, c="grey", alpha=.9)
  ax.plot(
      jnp.arange(true_values.shape[0]),
      prediction.mean(axis=0),
      c="green",
      alpha=.9)
  ax.fill_between(
      x=jnp.arange(true_values.shape[0]),
      y1=lower_bound,
      y2=upper_bound,
      alpha=.35,
      color="green")
  ax.legend(["True KPI", "Predicted KPI"])
  ax.yaxis.grid(color="gray", linestyle="dashed", alpha=0.3)
  ax.xaxis.grid(color="gray", linestyle="dashed", alpha=0.3)
  title = "\n".join([
      "True and predicted KPI.",
      "R2 = {r2:.{digits}f}".format(r2=r2, digits=digits),
      "MAPE = {mape:.{digits}f}%".format(mape=mape, digits=digits)
  ])
  ax.title.set_text(title)
  plt.close()
  return fig


def plot_model_fit(media_mix_model: lightweight_mmm.LightweightMMM,
                   target_scaler: Optional[preprocessing.CustomScaler] = None,
                   interval_mid_range: float = .9,
                   digits: int = 3) -> matplotlib.figure.Figure:
  """Plots the ground truth, predicted value and interval for the training data.

  Model needs to be fit before calling this function to plot.

  Args:
    media_mix_model: Media mix model.
    target_scaler: Scaler used for scaling the target, to unscaled values and
      plot in the original scale.
    interval_mid_range: Mid range interval to take for plotting. Eg. .9 will use
      .05 and .95 as the lower and upper quantiles. Must be a float number.
      between 0 and 1.
    digits: Number of decimals to display on metrics in the plot.

  Returns:
    Plot of model fit.
  """
  if not hasattr(media_mix_model, "trace"):
    raise lightweight_mmm.NotFittedModelError(
        "Model needs to be fit first before attempting to plot its fit.")
  target_train = media_mix_model._target
  posterior_pred = media_mix_model.trace["mu"]
  if target_scaler:
    posterior_pred = target_scaler.inverse_transform(posterior_pred)
    target_train = target_scaler.inverse_transform(target_train)
  return _create_shaded_line_plot(posterior_pred, target_train,
                                  interval_mid_range, digits)


def plot_out_of_sample_model_fit(out_of_sample_predictions: jnp.ndarray,
                                 out_of_sample_target: jnp.ndarray,
                                 interval_mid_range: float = .9,
                                 digits: int = 3) -> matplotlib.figure.Figure:
  """Plots the ground truth, predicted value and interval for the test data.

  Args:
    out_of_sample_predictions: Predictions for the out-of-sample period, as
      derived from mmm.predict.
    out_of_sample_target: Target for the out-of-sample period. Needs to be on
      the same scale as out_of_sample_predictions.
    interval_mid_range: Mid range interval to take for plotting. Eg. .9 will use
      .05 and .95 as the lower and upper quantiles. Must be a float number.
      between 0 and 1.
    digits: Number of decimals to display on metrics in the plot.

  Returns:
    Plot of model fit.
  """
  return _create_shaded_line_plot(out_of_sample_predictions,
                                  out_of_sample_target, interval_mid_range,
                                  digits)


def plot_media_channel_posteriors(
    media_mix_model: lightweight_mmm.LightweightMMM,
    channel_names: Optional[Sequence[Any]] = None,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    n_columns: int = 3) -> matplotlib.figure.Figure:
  """Plots the posterior distributions of estimated media channel effects.

  Model needs to be fit before calling this function to plot.

  Args:
    media_mix_model: Media mix model.
    channel_names: Names of media channels to be added to plot.
    quantiles: Quantiles to draw on the distribution.
    n_columns: Number of columns of generated subplot.

  Returns:
    Plot of posterior distributions.
  """
  if not hasattr(media_mix_model, "trace"):
    raise lightweight_mmm.NotFittedModelError(
        "Model needs to be fit first before attempting to plot its fit.")
  n_media_channels = np.shape(media_mix_model.trace["beta_media"])[1]
  n_rows = (n_media_channels + n_columns - 1) // n_columns

  media_channel_posteriors = media_mix_model.trace["beta_media"]
  if channel_names is None:
    channel_names = np.arange(np.shape(media_channel_posteriors)[1])
  fig, axes = plt.subplots(n_rows, n_columns, figsize=(10, 10))
  for index, ax in enumerate(axes.flatten()[:n_media_channels]):
    ax = arviz.plot_kde(
        media_channel_posteriors[:, index], quantiles=quantiles, ax=ax)
    ax.set_xlabel(f"media channel {channel_names[index]}")

  for index, ax in enumerate(axes.flatten()[n_media_channels:]):
    fig.delaxes(ax)
  fig.tight_layout()
  plt.close()
  return fig


def plot_bars_media_metrics(
    metric: jnp.ndarray,
    metric_name: str = "metric",
    channel_names: Optional[Tuple[Any]] = None,
    interval_mid_range: float = .9) -> matplotlib.figure.Figure:
  """Plots a barchart of estimated media effects with their percentile interval.

  The lower and upper percentile need to be between 0-1.

  Args:
    metric: Estimated media metric as returned by
      lightweight_mmm.get_posterior_metrics(). Can be either effects or ROI.
    metric_name: Name of the media metric, e.g. effect or ROI.
    channel_names: Names of media channels to be added to plot.
    interval_mid_range: Mid range interval to take for plotting. Eg. .9 will use
      .05 and .95 as the lower and upper quantiles. Must be a float number.

  Returns:
    Barplot of estimated media effects with defined percentile-bars.
  """
  if channel_names is None:
    channel_names = np.arange(np.shape(metric)[1])
  upper_quantile = 1 - (1 - interval_mid_range) / 2
  lower_quantile = (1 - interval_mid_range) / 2

  fig, ax = plt.subplots(1, 1)
  sns.barplot(data=metric, ci=None, ax=ax)
  quantile_bounds = np.quantile(
      metric, q=[lower_quantile, upper_quantile], axis=0)
  quantile_bounds[0] = metric.mean(axis=0) - quantile_bounds[0]
  quantile_bounds[1] = quantile_bounds[1] - metric.mean(axis=0)

  ax.errorbar(
      x=np.arange(np.shape(metric)[1]),
      y=metric.mean(axis=0),
      yerr=quantile_bounds,
      fmt="none",
      c="black")
  ax.set_xticks(range(len(channel_names)))
  ax.set_xticklabels(channel_names, rotation=45)
  fig.suptitle(
      f"Estimated media channel {metric_name}, error bars show {np.round(lower_quantile, 2)} - {np.round(upper_quantile, 2)} credibility interval"
  )
  plt.close()
  return fig
