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

"""Plotting functions pre and post model fitting."""

import functools
import logging

# Using these types from typing instead of their generic types in the type hints
# in order to be compatible with Python 3.7 and 3.8.
from typing import Any, List, Optional, Sequence, Tuple

import arviz
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd
import seaborn as sns
from sklearn import metrics

from lightweight_mmm import lightweight_mmm
from lightweight_mmm import models
from lightweight_mmm import preprocessing
from lightweight_mmm import utils

plt.style.use("default")

_PALETTE = sns.color_palette(n_colors=100)


@functools.partial(jax.jit, static_argnames=("media_mix_model"))
def _make_single_prediction(media_mix_model: lightweight_mmm.LightweightMMM,
                            mock_media: jnp.ndarray,
                            extra_features: Optional[jnp.ndarray],
                            seed: Optional[int]
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
    seed: Seed to use for PRNGKey during sampling. For replicability run
      this function and any other function that gets predictions with the same
      seed.

  Returns:
    A point estimate for the given data.
  """
  return media_mix_model.predict(
      media=jnp.expand_dims(mock_media, axis=0),
      extra_features=extra_features,
      seed=seed).mean(axis=0)


@functools.partial(
    jax.jit,
    static_argnames=("media_mix_model", "target_scaler"))
def _generate_diagonal_predictions(
    media_mix_model: lightweight_mmm.LightweightMMM,
    media_values: jnp.ndarray,
    extra_features: Optional[jnp.ndarray],
    target_scaler: Optional[preprocessing.CustomScaler],
    prediction_offset: jnp.ndarray,
    seed: Optional[int]):
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
    seed: Seed to use for PRNGKey during sampling. For replicability run
      this function and any other function that gets predictions with the same
      seed.

  Returns:
    The predictions for the given data.
  """
  make_predictions = jax.vmap(fun=_make_single_prediction,
                              in_axes=(None, 0, None, None))
  diagonal = jnp.eye(media_values.shape[0])
  if media_values.ndim == 2:  # Only two since we only provide one row
    diagonal = jnp.expand_dims(diagonal, axis=-1)
    media_values = jnp.expand_dims(media_values, axis=0)
  diag_media_values = diagonal * media_values
  predictions = make_predictions(
      media_mix_model,
      diag_media_values,
      extra_features,
      seed) - prediction_offset
  predictions = jnp.squeeze(predictions)
  if target_scaler:
    predictions = target_scaler.inverse_transform(predictions)
  if predictions.ndim == 2:
    predictions = jnp.sum(predictions, axis=-1)
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


def _calculate_media_contribution(
    media_mix_model: lightweight_mmm.LightweightMMM) -> jnp.ndarray:
  """Computes contribution for each sample, time, channel.

  Serves as a helper function for making predictions for each channel, time
  and estimate sample. It is meant to be used in creating media baseline
  contribution dataframe and visualize media attribution over spend proportion
  plot.

  Args:
    media_mix_model: Media mix model.

  Returns:
    Estimation of contribution for each sample, time, channel.

  Raises:
    NotFittedModelError: if the model is not fitted before computation
  """
  if not hasattr(media_mix_model, "trace"):
    raise lightweight_mmm.NotFittedModelError(
        "Model needs to be fit first before attempting to plot its fit.")

  if media_mix_model.trace["media_transformed"].ndim > 3:
    # s for samples, t for time, c for media channels, g for geo
    einsum_str = "stcg, scg->stcg"
  elif media_mix_model.trace["media_transformed"].ndim == 3:
    # s for samples, t for time, c for media channels
    einsum_str = "stc, sc->stc"

  media_contribution = jnp.einsum(einsum_str,
                                  media_mix_model.trace["media_transformed"],
                                  media_mix_model.trace["coef_media"])
  if media_mix_model.trace["media_transformed"].ndim > 3:
    # Aggregate media channel contribution across geos.
    media_contribution = media_contribution.sum(axis=-1)
  return media_contribution


def create_attribution_over_spend_fractions(
    media_mix_model: lightweight_mmm.LightweightMMM,
    media_spend: jnp.ndarray,
    channel_names: Optional[Sequence[str]] = None,
    time_index: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
  """Creates a dataframe for media attribution over spend.

  The output dataframe will be used to create media attribution and spend
  barplot; and attribution over spend lineplot

  Args:
    media_mix_model: Media mix model.
    media_spend: Media spend per channel. If 1D, it needs to be pre-processed to
      align with time index range. 2D represents media spend per time and
      channel. Media spends need to be aggregated over geo before input.
    channel_names: Names of media channels to be added to the output dataframe.
    time_index: Time range index used for calculation.

  Returns:
    DataFrame containing fractions of the contribution and spend and
    attribution over spend for each channel.

  Rasies:
    ValueError: if any of the media values are negative or any aggregated media
    spends are zero or negative.
    NotFittedModelError: if the model is not fitted.
  """
  if (media_spend < 0).any():
    raise ValueError("Values in media must all be non-negative or values "
                     "in aggregated media must be possitive.")
  if channel_names is None:
    try:
      channel_names = media_mix_model.media_names
    except AttributeError as att_error:
      raise lightweight_mmm.NotFittedModelError(
          "Model needs to be fit first before attempting to plot its fit."
      ) from att_error

  media_contribution = _calculate_media_contribution(media_mix_model)
  if time_index is None:
    time_index = (0, media_contribution.shape[1])

  # Select time span and aggregate 2D media spend.
  if media_spend.ndim == 1:
    logging.warning("1D media spend has to align with time index range.")
  elif media_spend.ndim == 2:
    media_spend = media_spend[time_index[0]:time_index[1], :]
    media_spend = media_spend.sum(axis=0)

  # Select time span and aggregate media contribution.
  media_contribution = media_contribution[:, time_index[0]:time_index[1],]
  media_contribution = media_contribution.sum(axis=(0, 1))

  # Create media contribution and spend dataframe
  media_df = pd.DataFrame(
      np.transpose([media_contribution, media_spend]),
      index=channel_names,
      columns=["media attribution", "media spend"])

  if (media_df["media spend"] == 0).any():
    raise ValueError("Values in media must all be non-negative or values "
                     "in aggregated media must be possitive.")

  normalized_media_df = media_df.div(media_df.sum(axis=0), axis=1)
  normalized_media_df["attribution over spend"] = normalized_media_df[
      "media attribution"] / normalized_media_df["media spend"]
  return normalized_media_df


def create_media_baseline_contribution_df(
    media_mix_model: lightweight_mmm.LightweightMMM,
    target_scaler: Optional[preprocessing.CustomScaler] = None,
    channel_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
  """Creates a dataframe for weekly media channels & basline contribution.

  The output dataframe will be used to create a stacked area plot to visualize
  the contribution of each media channels & baseline.

  Args:
    media_mix_model: Media mix model.
    target_scaler: Scaler used for scaling the target.
    channel_names: Names of media channels.

  Returns:
    contribution_df: DataFrame of weekly channels & baseline contribution
    percentage & volume.
  """
  # Create media contribution matrix.
  scaled_media_contribution = _calculate_media_contribution(media_mix_model)

  # Aggregate media channel contribution across samples.
  sum_scaled_media_contribution_across_samples = scaled_media_contribution.sum(
      axis=0)
  # Aggregate media channel contribution across channels.
  sum_scaled_media_contribution_across_channels = scaled_media_contribution.sum(
      axis=2)

  # Calculate the baseline contribution.
  # Scaled prediction - sum of scaled contribution across channels.
  scaled_prediction = media_mix_model.trace["mu"]
  if media_mix_model.trace["media_transformed"].ndim > 3:
    # Sum up the scaled prediction across all the geos.
    scaled_prediction = scaled_prediction.sum(axis=-1)
  baseline_contribution = scaled_prediction - sum_scaled_media_contribution_across_channels

  # Sum up the scaled media, baseline contribution and predictio across samples.
  sum_scaled_media_contribution_across_channels_samples = sum_scaled_media_contribution_across_channels.sum(
      axis=0)
  sum_scaled_baseline_contribution_across_samples = baseline_contribution.sum(
      axis=0)

  # Adjust baseline contribution and prediction when there's any negative value.
  adjusted_sum_scaled_baseline_contribution_across_samples = np.where(
      sum_scaled_baseline_contribution_across_samples < 0, 0,
      sum_scaled_baseline_contribution_across_samples)
  adjusted_sum_scaled_prediction_across_samples = adjusted_sum_scaled_baseline_contribution_across_samples + sum_scaled_media_contribution_across_channels_samples

  # Calculate the media and baseline pct.
  # Media/baseline contribution across samples/total prediction across samples.
  media_contribution_pct_by_channel = (
      sum_scaled_media_contribution_across_samples /
      adjusted_sum_scaled_prediction_across_samples.reshape(-1, 1))
  # Adjust media pct contribution if the value is nan
  media_contribution_pct_by_channel = np.nan_to_num(
      media_contribution_pct_by_channel)

  baseline_contribution_pct = adjusted_sum_scaled_baseline_contribution_across_samples / adjusted_sum_scaled_prediction_across_samples
  # Adjust baseline pct contribution if the value is nan
  baseline_contribution_pct = np.nan_to_num(
      baseline_contribution_pct)

  # If the channel_names is none, then create naming covention for the channels.
  if channel_names is None:
    channel_names = media_mix_model.media_names

  # Create media/baseline contribution pct as dataframes.
  media_contribution_pct_by_channel_df = pd.DataFrame(
      media_contribution_pct_by_channel, columns=channel_names)
  baseline_contribution_pct_df = pd.DataFrame(
      baseline_contribution_pct, columns=["baseline"])
  contribution_pct_df = pd.merge(
      media_contribution_pct_by_channel_df,
      baseline_contribution_pct_df,
      left_index=True,
      right_index=True)

  # If there's target scaler then inverse transform the posterior prediction.
  posterior_pred = media_mix_model.trace["mu"]
  if target_scaler:
    posterior_pred = target_scaler.inverse_transform(posterior_pred)

  # Take the sum of posterior predictions across geos.
  if media_mix_model.trace["media_transformed"].ndim > 3:
    posterior_pred = posterior_pred.sum(axis=-1)

  # Take the average of the inverse transformed prediction across samples.
  posterior_pred_df = pd.DataFrame(
      posterior_pred.mean(axis=0), columns=["avg_prediction"])

  # Adjust prediction value when prediction is less than 0.
  posterior_pred_df["avg_prediction"] = np.where(
      posterior_pred_df["avg_prediction"] < 0, 0,
      posterior_pred_df["avg_prediction"])

  contribution_pct_df.columns = [
      "{}_percentage".format(col) for col in contribution_pct_df.columns
  ]
  contribution_df = pd.merge(
      contribution_pct_df, posterior_pred_df, left_index=True, right_index=True)

  # Create contribution by multiplying average prediction by media/baseline pct.
  for channel in channel_names:
    channel_contribution_col_name = "{} contribution".format(channel)
    channel_pct_col = "{}_percentage".format(channel)
    contribution_df.loc[:, channel_contribution_col_name] = contribution_df[
        channel_pct_col] * contribution_df["avg_prediction"]
    contribution_df.loc[:, channel_contribution_col_name] = contribution_df[
        channel_contribution_col_name].astype("float")
  contribution_df.loc[:, "baseline contribution"] = contribution_df[
      "baseline_percentage"] * contribution_df["avg_prediction"]

  period = np.arange(1, contribution_df.shape[0] + 1)
  contribution_df.loc[:, "period"] = period
  return contribution_df


def plot_response_curves(# jax-ndarray
    media_mix_model: lightweight_mmm.LightweightMMM,
    media_scaler: Optional[preprocessing.CustomScaler] = None,
    target_scaler: Optional[preprocessing.CustomScaler] = None,
    prices: jnp.ndarray = None,
    optimal_allocation_per_timeunit: Optional[jnp.ndarray] = None,
    steps: int = 50,
    percentage_add: float = 0.2,
    apply_log_scale: bool = False,
    figure_size: Tuple[int, int] = (8, 10),
    n_columns: int = 3,
    marker_size: int = 8,
    legend_fontsize: int = 8,
    seed: Optional[int] = None) -> matplotlib.figure.Figure:
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
    seed: Seed to use for PRNGKey during sampling. For replicability run
      this function and any other function that gets predictions with the same
      seed.

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
      jax.vmap(_make_single_prediction,
               in_axes=(None, 0, None, None),
               out_axes=0),
      in_axes=(None, 0, None, None), out_axes=1)
  diagonal = jnp.repeat(
      jnp.eye(media_mix_model.n_media_channels), steps,
      axis=0).reshape(media_mix_model.n_media_channels, steps,
                      media_mix_model.n_media_channels)

  prediction_offset = media_mix_model.predict(
      media=jnp.zeros((1, *media.shape[1:])),
      extra_features=extra_features).mean(axis=0)

  if media.ndim == 3:
    diagonal = jnp.expand_dims(diagonal, axis=-1)
    prediction_offset = jnp.expand_dims(prediction_offset, axis=0)
  mock_media = media_ranges * diagonal
  predictions = jnp.squeeze(a=make_predictions(media_mix_model,
                                               mock_media,
                                               extra_features,
                                               seed))
  predictions = predictions - prediction_offset
  media_ranges = jnp.squeeze(media_ranges)
  if target_scaler:
    predictions = target_scaler.inverse_transform(predictions)

  if media_scaler:
    media_ranges = media_scaler.inverse_transform(media_ranges)

  if prices is not None:
    if media.ndim == 3:
      prices = jnp.expand_dims(prices, axis=-1)
    media_ranges *= prices

  if predictions.ndim == 3:
    media_ranges = jnp.sum(media_ranges, axis=-1)
    predictions = jnp.sum(predictions, axis=-1)

  if optimal_allocation_per_timeunit is not None:
    average_allocation = media_mix_model.media.mean(axis=0)
    average_allocation_predictions = _generate_diagonal_predictions(
        media_mix_model=media_mix_model,
        media_values=average_allocation,
        extra_features=extra_features,
        target_scaler=target_scaler,
        prediction_offset=prediction_offset,
        seed=seed)
    optimal_allocation_predictions = _generate_diagonal_predictions(
        media_mix_model=media_mix_model,
        media_values=optimal_allocation_per_timeunit,
        extra_features=extra_features,
        target_scaler=target_scaler,
        prediction_offset=prediction_offset,
        seed=seed)
    if media_scaler:
      average_allocation = media_scaler.inverse_transform(average_allocation)
      optimal_allocation_per_timeunit = media_scaler.inverse_transform(
          optimal_allocation_per_timeunit)
    if prices is not None:
      optimal_allocation_per_timeunit *= prices
      average_allocation *= prices
    if media.ndim == 3:
      average_allocation = jnp.sum(average_allocation, axis=-1)
      optimal_allocation_per_timeunit = jnp.sum(
          optimal_allocation_per_timeunit, axis=-1)

  kpi_label = "KPI" if target_scaler else "Normalized KPI"
  fig = plt.figure(media_mix_model.n_media_channels + 1,
                   figsize=figure_size,
                   tight_layout=True)
  n_rows = _calculate_number_rows_plot(
      n_media_channels=media_mix_model.n_media_channels, n_columns=n_columns)
  last_ax = fig.add_subplot(n_rows, 1, n_rows)
  for i in range(media_mix_model.n_media_channels):
    ax = fig.add_subplot(n_rows, n_columns, i + 1)
    sns.lineplot(
        x=media_ranges[:, i],
        y=predictions[:, i],
        label=media_mix_model.media_names[i],
        color=_PALETTE[i],
        ax=ax)
    sns.lineplot(
        x=media_ranges[:, i],
        y=jnp.log(predictions[:, i]) if apply_log_scale else predictions[:, i],
        label=media_mix_model.media_names[i],
        color=_PALETTE[i],
        ax=last_ax)
    if optimal_allocation_per_timeunit is not None:
      ax.plot(
          average_allocation[i],
          average_allocation_predictions[i],
          marker="o",
          markersize=marker_size,
          label="avg_spend",
          color=_PALETTE[i])
      ax.plot(
          optimal_allocation_per_timeunit[i],
          optimal_allocation_predictions[i],
          marker="x",
          markersize=marker_size + 2,
          label="optimal_spend",
          color=_PALETTE[i])
    ax.set_ylabel(kpi_label)
    ax.set_xlabel("Normalized Spend" if not media_scaler else "Spend")
    ax.legend(fontsize=legend_fontsize)

  fig.suptitle("Response curves", fontsize=20)
  last_ax.set_ylabel(kpi_label if not apply_log_scale else f"log({kpi_label})")
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


def _create_shaded_line_plot(predictions: jnp.ndarray,
                             target: jnp.ndarray,
                             axis: matplotlib.axes.Axes,
                             title_prefix: str = "",
                             interval_mid_range: float = .9,
                             digits: int = 3) -> None:
  """Creates a plot of ground truth, predicted value and credibility interval.

  Args:
    predictions: 2d array of predicted values.
    target: Array of true values. Must be same length as predictions.
    axis: Matplotlib axis in which to plot the data.
    title_prefix: Prefix to add as the label of the plot.
    interval_mid_range: Mid range interval to take for plotting. Eg. .9 will use
      .05 and .95 as the lower and upper quantiles. Must be a float number
      between 0 and 1.
    digits: Number of decimals to display on metrics in the plot.
  """
  if predictions.shape[1] != len(target):
    raise ValueError(
        "Predicted data and ground-truth data must have same length.")
  upper_quantile = 1 - (1 - interval_mid_range) / 2
  lower_quantile = (1 - interval_mid_range) / 2
  upper_bound = jnp.quantile(a=predictions, q=upper_quantile, axis=0)
  lower_bound = jnp.quantile(a=predictions, q=lower_quantile, axis=0)

  r2, _ = arviz.r2_score(y_true=target, y_pred=predictions)
  mape = 100 * metrics.mean_absolute_percentage_error(
      y_true=target, y_pred=predictions.mean(axis=0))
  axis.plot(jnp.arange(target.shape[0]), target, c="grey", alpha=.9)
  axis.plot(
      jnp.arange(target.shape[0]),
      predictions.mean(axis=0),
      c="green",
      alpha=.9)
  axis.fill_between(
      x=jnp.arange(target.shape[0]),
      y1=lower_bound,
      y2=upper_bound,
      alpha=.35,
      color="green")
  axis.legend(["True KPI", "Predicted KPI"])
  axis.yaxis.grid(color="gray", linestyle="dashed", alpha=0.3)
  axis.xaxis.grid(color="gray", linestyle="dashed", alpha=0.3)
  title = " ".join([
      title_prefix,
      "True and predicted KPI.",
      "R2 = {r2:.{digits}f}".format(r2=r2, digits=digits),
      "MAPE = {mape:.{digits}f}%".format(mape=mape, digits=digits)
  ])
  axis.title.set_text(title)
  plt.close()


def _call_fit_plotter(
    predictions: jnp.array,
    target: jnp.array,
    interval_mid_range: float,
    digits: int) -> matplotlib.figure.Figure:
  """Calls the shaded line plot once for national and N times for geo models.

  Args:
    predictions: 2d array of predicted values.
    target: Array of true values. Must be same length as prediction.
    interval_mid_range: Mid range interval to take for plotting. Eg. .9 will use
      .05 and .95 as the lower and upper quantiles. Must be a float number
      between 0 and 1.
    digits: Number of decimals to display on metrics in the plot.

  Returns:
    Figure of the plot.
  """
  # TODO(): Allow to pass geo names for fit plots
  if predictions.ndim == 3:  # Multiple plots for geo model
    figure, axes = plt.subplots(predictions.shape[-1],
                                figsize=(10, 5 * predictions.shape[-1]))
    for i, ax in enumerate(axes):
      _create_shaded_line_plot(predictions=predictions[..., i],
                               target=target[..., i],
                               axis=ax,
                               title_prefix=f"Geo {i}:",
                               interval_mid_range=interval_mid_range,
                               digits=digits)
  else:  # Single plot for national model
    figure, ax = plt.subplots(1, 1)
    _create_shaded_line_plot(predictions=predictions,
                             target=target,
                             axis=ax,
                             interval_mid_range=interval_mid_range,
                             digits=digits)
  return figure


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

  return _call_fit_plotter(
      predictions=posterior_pred,
      target=target_train,
      interval_mid_range=interval_mid_range,
      digits=digits)


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
  return _call_fit_plotter(
      predictions=out_of_sample_predictions,
      target=out_of_sample_target,
      interval_mid_range=interval_mid_range,
      digits=digits)


def plot_media_channel_posteriors(
    media_mix_model: lightweight_mmm.LightweightMMM,
    channel_names: Optional[Sequence[Any]] = None,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    fig_size: Optional[Tuple[int, int]] = None) -> matplotlib.figure.Figure:
  """Plots the posterior distributions of estimated media channel effect.

  Model needs to be fit before calling this function to plot.

  Args:
    media_mix_model: Media mix model.
    channel_names: Names of media channels to be added to plot.
    quantiles: Quantiles to draw on the distribution.
    fig_size: Size of the figure to plot as used by matplotlib. If not specified
      it will be determined dynamically based on the number of media channels
      and geos the model was trained on.

  Returns:
    Plot of posterior distributions.
  """
  if not hasattr(media_mix_model, "trace"):
    raise lightweight_mmm.NotFittedModelError(
        "Model needs to be fit first before attempting to plot its fit.")

  n_media_channels = np.shape(media_mix_model.trace["coef_media"])[1]
  n_geos = (
      media_mix_model.media.shape[2] if media_mix_model.media.ndim == 3 else 1)

  if not fig_size:
    fig_size = (5 * n_geos, 3 * n_media_channels)

  media_channel_posteriors = media_mix_model.trace["coef_media"]
  if channel_names is None:
    channel_names = np.arange(np.shape(media_channel_posteriors)[1])
  fig, axes = plt.subplots(
      nrows=n_media_channels, ncols=n_geos, figsize=fig_size)
  for channel_i, channel_axis in enumerate(axes):
    if isinstance(channel_axis, np.ndarray):
      for geo_i, geo_axis in enumerate(channel_axis):
        geo_axis = arviz.plot_kde(
            media_channel_posteriors[:, channel_i, geo_i],
            quantiles=quantiles,
            ax=geo_axis)
        axis_label = f"media channel {channel_names[channel_i]} geo {geo_i}"
        geo_axis.set_xlabel(axis_label)
    else:
      channel_axis = arviz.plot_kde(
          media_channel_posteriors[:, channel_i],
          quantiles=quantiles,
          ax=channel_axis)
      axis_label = f"media channel {channel_names[channel_i]}"
      channel_axis.set_xlabel(axis_label)

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
      lightweight_mmm.get_posterior_metrics(). Can be either contribution
      percentage or ROI.
    metric_name: Name of the media metric, e.g. contribution percentage or ROI.
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

  if metric.ndim == 3:
    metric = jnp.mean(metric, axis=-1)

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
      f"Estimated media channel {metric_name}. \n Error bars show "
      f"{np.round(lower_quantile, 2)} - {np.round(upper_quantile, 2)} "
      "credibility interval."
  )
  plt.close()
  return fig


def plot_pre_post_budget_allocation_comparison(
    media_mix_model: lightweight_mmm.LightweightMMM,
    kpi_with_optim: jnp.ndarray,
    kpi_without_optim: jnp.ndarray,
    optimal_buget_allocation: jnp.ndarray,
    previous_budget_allocation: jnp.ndarray,
    channel_names: Optional[Sequence[Any]] = None,
    figure_size: Tuple[int, int] = (20, 10)
) -> matplotlib.figure.Figure:
  """Plots a barcharts to compare pre & post budget allocation.

  Args:
    media_mix_model: Media mix model to use for the optimization.
    kpi_with_optim: Negative predicted target variable with optimized budget
      allocation.
    kpi_without_optim: negative predicted target variable with original budget
      allocation proportion base on the historical data.
    optimal_buget_allocation: Optmized budget allocation.
    previous_budget_allocation: Starting budget allocation based on original
      budget allocation proportion.
    channel_names: Names of media channels to be added to plot.
    figure_size: size of the plot.

  Returns:
    Barplots of budget allocation across media channels pre & post optimization.
  """

  if not hasattr(media_mix_model, "trace"):
    raise lightweight_mmm.NotFittedModelError(
        "Model needs to be fit first before attempting to plot its fit.")

  previous_budget_allocation_pct = previous_budget_allocation / jnp.sum(
      previous_budget_allocation)
  optimized_budget_allocation_pct = optimal_buget_allocation / jnp.sum(
      optimal_buget_allocation)

  if channel_names is None:
    channel_names = media_mix_model.media_names
  x_axis = np.arange(len(channel_names))

  pre_optimizaiton_predicted_target = kpi_without_optim * -1
  post_optimization_predictiond_target = kpi_with_optim * -1
  predictions = [
      pre_optimizaiton_predicted_target, post_optimization_predictiond_target
  ]

  # Create bar chart.
  fig, axes = plt.subplots(2, 1, figsize=figure_size)

  plots1 = axes[0].bar(
      x_axis - 0.2,
      previous_budget_allocation,
      width=0.4,
      label="previous budget allocation")
  plots2 = axes[0].bar(
      x_axis + 0.2,
      optimal_buget_allocation,
      width=0.4,
      label="optimized budget allocation")
  axes[0].set_ylabel("Budget Allocation", fontsize="x-large")
  axes[0].set_title(
      "Before and After Optimization Budget Allocation Comparison",
      fontsize="x-large")
  # Iterrating over the bars one-by-one.
  for bar_i in range(len(plots1.patches)):
    bar = plots1.patches[bar_i]
    axes[0].annotate(
        "{:.0%}".format(previous_budget_allocation_pct[bar_i]),
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        ha="center",
        va="center",
        size=10,
        xytext=(0, 8),
        textcoords="offset points")

  # Iterrating over the bars one-by-one.
  for bar_i in range(len(plots2.patches)):
    bar = plots2.patches[bar_i]
    axes[0].annotate(
        "{:.0%}".format(optimized_budget_allocation_pct[bar_i]),
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        ha="center",
        va="center",
        size=10,
        xytext=(0, 8),
        textcoords="offset points")

  axes[0].set_xticks(x_axis)
  axes[0].set_xticklabels(channel_names, fontsize="medium")
  axes[0].legend(fontsize="medium")

  plots3 = axes[1].bar([
      "pre optimization predicted target",
      "post optimization predicted target"
  ], predictions)
  axes[1].set_ylim(
      min(predictions) - min(predictions) * 0.1,
      max(predictions) + min(predictions) * 0.1)
  axes[1].set_ylabel("Predicted Target Variable", fontsize="x-large")
  axes[1].set_title(
      "Pre Post Optimization Target Variable Comparison", fontsize="x-large")
  axes[1].set_xticks(range(2))
  axes[1].set_xticklabels([
      "pre optimization predicted target",
      "post optimization predicted target"
  ],
                          fontsize="x-large")

  # Iterrating over the bars one-by-one.
  for bar_i in range(len(plots3.patches)):
    bar = plots3.patches[bar_i]
    axes[1].annotate(
        "{:,.1f}".format(predictions[bar_i]),
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        ha="center",
        va="center",
        size=10,
        xytext=(0, 8),
        textcoords="offset points")

  plt.tight_layout()
  plt.close()
  return fig


def plot_media_baseline_contribution_area_plot(
    media_mix_model: lightweight_mmm.LightweightMMM,
    target_scaler: Optional[preprocessing.CustomScaler] = None,
    channel_names: Optional[Sequence[Any]] = None,
    fig_size: Optional[Tuple[int, int]] = (20, 7),
    legend_outside: Optional[bool] = False,
) -> matplotlib.figure.Figure:
  """Plots an area chart to visualize weekly media & baseline contribution.

  Args:
    media_mix_model: Media mix model.
    target_scaler: Scaler used for scaling the target.
    channel_names: Names of media channels.
    fig_size: Size of the figure to plot as used by matplotlib.
    legend_outside: Put the legend outside of the chart, center-right.

  Returns:
    Stacked area chart of weekly baseline & media contribution.
  """
  # Create media channels & baseline contribution dataframe.
  contribution_df = create_media_baseline_contribution_df(
      media_mix_model=media_mix_model,
      target_scaler=target_scaler,
      channel_names=channel_names)

  # Create contribution dataframe for the plot.
  contribution_columns = [
      col for col in contribution_df.columns if "contribution" in col
  ]
  contribution_df_for_plot = contribution_df.loc[:, contribution_columns]
  contribution_df_for_plot = contribution_df_for_plot[
      contribution_df_for_plot.columns[::-1]]
  period = np.arange(1, contribution_df_for_plot.shape[0] + 1)
  contribution_df_for_plot.loc[:, "period"] = period

  # Plot the stacked area chart.
  fig, ax = plt.subplots()
  contribution_df_for_plot.plot.area(
      x="period", stacked=True, figsize=fig_size, ax=ax)
  ax.set_title("Attribution Over Time", fontsize="x-large")
  ax.tick_params(axis="y")
  ax.set_ylabel("Baseline & Media Chanels Attribution")
  ax.set_xlabel("Period")
  ax.set_xlim(1, contribution_df_for_plot["period"].max())
  ax.set_xticks(contribution_df_for_plot["period"])
  ax.set_xticklabels(contribution_df_for_plot["period"])
  # Get handles and labels for sorting.
  handles, labels = ax.get_legend_handles_labels()
  # If true, legend_outside reversed the legend and puts the legend center left,
  # outside the chart.
  # If false, legend_outside only reverses the legend order.

  # Channel order is based on the media input and chart logic.
  # Chart logic puts the last column onto the bottom.
  # To be in line with chart order, we reserve the channel order in the legend.
  if legend_outside:
    ax.legend(handles[::-1], labels[::-1],
              loc="center left", bbox_to_anchor=(1, 0.5))
  # Only sort the legend.
  else:
    ax.legend(handles[::-1], labels[::-1])

  for tick in ax.get_xticklabels():
    tick.set_rotation(45)
  plt.close()
  return fig


def _make_prior_and_posterior_subplot_for_one_feature(
    prior_distribution: numpyro.distributions.Distribution,
    posterior_samples: np.ndarray,
    subplot_title: str,
    fig: matplotlib.figure.Figure,
    gridspec_fig: matplotlib.gridspec.GridSpec,
    i_ax: int,
    hyperprior: bool = False,
    number_of_samples_for_prior: int = 5000,
    kde_bandwidth_adjust_for_posterior: float = 1,
    seed: Optional[int] = None,
) -> Tuple[matplotlib.figure.Figure, matplotlib.gridspec.GridSpec, int]:
  """Helper function to make the prior and posterior distribution subplots.

  This function makes some (hard-coded) choices about how to display the prior
  and posterior distributions. First, it uses kernel density estimators to
  smooth the distributions rather than plotting the histograms directly (since
  the histograms look too noisy unless we take unreasonably large numbers of
  samples). Second, we found that the default bandwidth works pretty well for
  visualization of these distributions, though we expose the bw_adjust parameter
  if users wish to modify this. Finally, kernel density estimators tend to have
  issues at the edges of distributions, and this issue persists here too. We
  clip some distributions (Half-Normal, Beta, Gamma) to keep the KDE plot
  representations inside the domains where these distributions are defined, and
  we also set cut=0 for the posterior distributions to give better insight into
  their exact ranges. We don't do this for the prior distributions, since in
  general they have wider (or infinite) ranges and the bounds that we show would
  be mostly determined by number_of_samples_for_prior, which should just be an
  implementation detail.

  Args:
    prior_distribution: Numpyro distribution specifying the prior from which we
      will sample to generate the plot.
    posterior_samples: Array of samples from the posterior distribution,
      obtained from the trace of the media_mix_model. Might need to be flattened
      in some cases.
    subplot_title: Title to display for this particular subplot
    fig: The matplotlib Figure object for the overall plot.
    gridspec_fig: The matplotlib GridSpec object for the overall plot.
    i_ax: Index of the subplot within the gridspec_fig.
    hyperprior: Flag which indicates that the prior_distribution is actually a
      hyperprior distribution. LMMM is hierarchical on the channel coefficients
      when run at the geo-level, so this should currently only be set to True
      when working with the media channel coefficients.
    number_of_samples_for_prior: Controls the level of smoothing for the plotted
      version of the prior distribution. The default should be fine unless you
      want to decrease it to speed up runtime.
    kde_bandwidth_adjust_for_posterior: Multiplicative factor to adjust the
      bandwidth of the kernel density estimator, to control the level of
      smoothing for the posterior distribution. Passed to seaborn.kdeplot as the
      bw_adjust parameter there.
    seed: Seed to use for PRNGKey during sampling. For replicability run
        this function and any other function that utilises predictions with the
        same seed.

  Returns:
    fig: The matplotlib Figure object for the overall plot.
    gridspec_fig: The matplotlib GridSpec object for the overall plot.
    i_ax: Index of the subplot within the gridspec_fig, iterated by one.
  """

  if seed is None:
    seed = utils.get_time_seed()
  prior_samples = prior_distribution.sample(
      key=jax.random.PRNGKey(seed=seed),
      sample_shape=(number_of_samples_for_prior,))

  # Truncate the KDE plot representation of Half-Normal, Beta, and Gamma
  # distributions, since users might be confused to see a Half-Normal or Gamma
  # going negative, or a Beta distribution outside of [0, 1].
  if isinstance(
      prior_distribution,
      (numpyro.distributions.HalfNormal, numpyro.distributions.Gamma)):
    clipping_bounds = [0, None]
  elif isinstance(prior_distribution, numpyro.distributions.Beta):
    clipping_bounds = [0, 1]
  else:
    clipping_bounds = None

  if hyperprior:
    square_root_of_number_of_samples_for_prior = int(
        np.sqrt(number_of_samples_for_prior))
    prior_distribution = numpyro.distributions.continuous.HalfNormal(
        scale=prior_samples[:square_root_of_number_of_samples_for_prior])
    prior_samples = prior_distribution.sample(
        key=jax.random.PRNGKey(seed=seed),
        sample_shape=(square_root_of_number_of_samples_for_prior,)).flatten()

  ax = fig.add_subplot(gridspec_fig[i_ax, 0])
  sns.kdeplot(
      data=prior_samples,
      lw=4,
      clip=clipping_bounds,
      color="tab:blue", ax=ax, label="prior")
  prior_xlims = ax.get_xlim()

  sns.kdeplot(
      data=posterior_samples.flatten(),
      lw=4,
      clip=clipping_bounds,
      cut=0,
      bw_adjust=kde_bandwidth_adjust_for_posterior,
      color="tab:orange", ax=ax, label="posterior")
  posterior_xlims = ax.get_xlim()

  ax.legend(loc="best")
  ax.set_xlim(
      min(prior_xlims[0], posterior_xlims[0]),
      max(prior_xlims[1], posterior_xlims[1]))
  ax.set_yticks([])
  ax.set_ylabel("")
  ax.set_title(subplot_title, y=0.85, va="top", fontsize=10)

  i_ax += 1

  return fig, gridspec_fig, i_ax


def _collect_features_for_prior_posterior_plot(
    media_mix_model: lightweight_mmm.LightweightMMM,
    selected_features: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
  """Helper function to collect features to include in the prior/posterior plot.

  Args:
    media_mix_model: Fitted media mix model.
    selected_features: Optional list of feature names to select. If not
      specified (the default), all features are selected.

  Returns:
    features: List of all features for the given model type, for which we have
    defined prior distributions.
    geo_level_features: List of all geo-level features for the given model type.
    channel_level_features: List of all channel-level features for the given
    model type.
    seasonal_features: List of all seasonal features for the given model type.
    other_features: List of all other features for the given media_mix_model.

  Raises:
    ValueError: if feature names are passed to selected_features which do not
    appear in media_mix_model.
  """

  media_mix_model_attributes_to_check_for = [
      "trace",
      "_weekday_seasonality",
      "custom_priors",
      "n_geos",
      "n_media_channels",
      "_media_prior",
  ]
  if not all([
      hasattr(media_mix_model, x)
      for x in media_mix_model_attributes_to_check_for
  ]):
    raise lightweight_mmm.NotFittedModelError(
        "Model needs to be fit first in order to plot the posterior.")

  features = media_mix_model._prior_names
  if not media_mix_model._weekday_seasonality:
    features = features.difference([models._WEEKDAY])

  if media_mix_model._extra_features is None:
    features = features.difference(["coef_extra_features"])

  if media_mix_model.media.ndim == 2:
    features = features.difference(models.GEO_ONLY_PRIORS)
    features = features.union(["coef_media"])
  else:
    features = features.union(["coef_media", "channel_coef_media"])

  if selected_features:
    extraneous_features = set(selected_features).difference(features)
    if extraneous_features:
      raise ValueError(
          f"Selected_features {extraneous_features} not in media_mix_model.")
    features = selected_features

  geo_level_features = [
      models._COEF_SEASONALITY,
      models._COEF_TREND,
      models._INTERCEPT,
      models._SIGMA,
  ]
  channel_level_features = [
      models._AD_EFFECT_RETENTION_RATE,
      models._EXPONENT,
      models._HALF_MAX_EFFECTIVE_CONCENTRATION,
      models._LAG_WEIGHT,
      models._PEAK_EFFECT_DELAY,
      models._SLOPE,
      "channel_coef_media",
      "coef_media",
  ]
  seasonal_features = [models._GAMMA_SEASONALITY]
  if media_mix_model._weekday_seasonality:
    seasonal_features.append(models._WEEKDAY)
  other_features = list(set(features) - set(geo_level_features) -
                        set(channel_level_features) - set(seasonal_features))

  return (list(features), geo_level_features, channel_level_features,
          seasonal_features, other_features)


def plot_prior_and_posterior(
    media_mix_model: lightweight_mmm.LightweightMMM,
    fig_size: Optional[Tuple[int, int]] = None,
    selected_features: Optional[List[str]] = None,
    number_of_samples_for_prior: int = 5000,
    kde_bandwidth_adjust_for_posterior: float = 1,
    seed: Optional[int] = None,
) -> matplotlib.figure.Figure:
  """Plots prior and posterior distributions for parameters in media_mix_model.

  Args:
    media_mix_model: Fitted media mix model.
    fig_size: Size of the figure to plot as used by matplotlib. Default is a
      width of 8 and a height of 1.5 for each subplot.
    selected_features: Optional list of feature names to select. If not
      specified (the default), all features are selected.
    number_of_samples_for_prior: Controls the level of smoothing for the plotted
      version of the prior distribution. The default should be fine unless you
      want to decrease it to speed up runtime.
    kde_bandwidth_adjust_for_posterior: Multiplicative factor to adjust the
      bandwidth of the kernel density estimator, to control the level of
      smoothing for the posterior distribution. Passed to seaborn.kdeplot as the
      bw_adjust parameter there.
    seed: Seed to use for PRNGKey during sampling. For replicability run
        this function and any other function that utilises predictions with the
        same seed.

  Returns:
    Plot with Kernel density estimate smoothing showing prior and posterior
    distributions for every parameter in the given media_mix_model.

  Raises:
    NotFittedModelError: media_mix_model has not yet been fit.
    ValueError: A feature has been created without a well-defined prior.
  """

  (features, geo_level_features, channel_level_features, seasonal_features,
   other_features) = _collect_features_for_prior_posterior_plot(
       media_mix_model, selected_features)

  number_of_subplots = int(
      sum([
          np.product(media_mix_model.trace[x].shape[1:]) for x in features
      ]))

  if not fig_size:
    fig_size = (8, 1.5 * number_of_subplots)

  fig = plt.figure(figsize=fig_size, constrained_layout=True)
  gridspec_fig = matplotlib.gridspec.GridSpec(
      nrows=number_of_subplots, ncols=1, figure=fig, hspace=0.1)

  default_priors = {
      **models._get_default_priors(),
      **models._get_transform_default_priors()[media_mix_model.model_name]
  }

  kwargs_for_helper_function = {
      "fig": fig,
      "gridspec_fig": gridspec_fig,
      "number_of_samples_for_prior": number_of_samples_for_prior,
      "kde_bandwidth_adjust_for_posterior": kde_bandwidth_adjust_for_posterior,
      "seed": seed,
  }

  i_ax = 0
  for feature in (geo_level_features + channel_level_features +
                  seasonal_features + other_features):
    if feature not in features:
      continue

    if feature in media_mix_model.custom_priors:
      prior_distribution = media_mix_model.custom_priors[feature]
      if not isinstance(prior_distribution, numpyro.distributions.Distribution):
        raise ValueError(f"{feature} cannot be plotted.")
    elif feature in default_priors.keys():
      prior_distribution = default_priors[feature]
    elif feature in ("channel_coef_media", "coef_media"):
      # We have to fill this in later since the prior varies by channel.
      prior_distribution = None
    else:
      # This should never happen.
      raise ValueError(f"{feature} has no prior specified.")
    kwargs_for_helper_function["prior_distribution"] = prior_distribution

    if feature == models._COEF_EXTRA_FEATURES:
      for i_feature in range(media_mix_model.trace[feature].shape[1]):
        for j_geo in range(media_mix_model.n_geos):
          subplot_title = f"{feature} feature {i_feature}, geo {j_geo}"
          if media_mix_model.n_geos == 1:
            posterior_samples = np.array(
                media_mix_model.trace[feature][:, i_feature])
          else:
            posterior_samples = np.array(
                media_mix_model.trace[feature][:, i_feature, j_geo])
          (fig, gridspec_fig,
           i_ax) = _make_prior_and_posterior_subplot_for_one_feature(
               posterior_samples=posterior_samples,
               subplot_title=subplot_title,
               i_ax=i_ax,
               **kwargs_for_helper_function)

    if feature in geo_level_features:
      for i_geo in range(media_mix_model.n_geos):
        subplot_title = f"{feature}, geo {i_geo}"
        posterior_samples = np.array(media_mix_model.trace[feature][:, i_geo])
        (fig, gridspec_fig,
         i_ax) = _make_prior_and_posterior_subplot_for_one_feature(
             posterior_samples=posterior_samples,
             subplot_title=subplot_title,
             i_ax=i_ax,
             **kwargs_for_helper_function)

    if feature in channel_level_features:
      for i_channel in range(media_mix_model.n_media_channels):
        subplot_title = f"{feature}, channel {i_channel}"
        if feature in ("channel_coef_media", "coef_media"):
          prior_distribution = numpyro.distributions.continuous.HalfNormal(
              scale=jnp.squeeze(media_mix_model._media_prior[i_channel]))
        posterior_samples = np.array(
            jnp.squeeze(media_mix_model.trace[feature][:, i_channel]))
        kwargs_for_helper_function["prior_distribution"] = prior_distribution
        hyperprior = feature == "channel_coef_media"
        (fig, gridspec_fig,
         i_ax) = _make_prior_and_posterior_subplot_for_one_feature(
             posterior_samples=posterior_samples,
             subplot_title=subplot_title,
             i_ax=i_ax,
             hyperprior=hyperprior,
             **kwargs_for_helper_function)

    if feature in seasonal_features:
      for i_season in range(media_mix_model._degrees_seasonality):
        for j_season in range(2):
          sin_or_cos = "sin" if j_season == 0 else "cos"
          subplot_title = f"{feature}, seasonal mode {i_season}:{sin_or_cos}"
          posterior_samples = np.array(media_mix_model.trace[feature][:,
                                                                      i_season,
                                                                      j_season])
          (fig, gridspec_fig,
           i_ax) = _make_prior_and_posterior_subplot_for_one_feature(
               posterior_samples=posterior_samples,
               subplot_title=subplot_title,
               i_ax=i_ax,
               **kwargs_for_helper_function)

    if feature in other_features and feature != models._COEF_EXTRA_FEATURES:
      subplot_title = f"{feature}"
      posterior_samples = np.array(media_mix_model.trace[feature])
      (fig, gridspec_fig,
       i_ax) = _make_prior_and_posterior_subplot_for_one_feature(
           posterior_samples=posterior_samples,
           subplot_title=subplot_title,
           i_ax=i_ax,
           **kwargs_for_helper_function)
  return fig
