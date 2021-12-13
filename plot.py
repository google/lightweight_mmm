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

"""Plotting functions pre and post model fitting."""

from typing import Any, List, Optional, Sequence, Tuple

import arviz
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from lightweight_mmm import lightweight_mmm
from lightweight_mmm import preprocessing


def _make_single_prediction(media_mix_model: lightweight_mmm.LightweightMMM,
                            mock_media: jnp.array,
                            extra_features: Optional[jnp.array]) -> jnp.array:
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


def plot_response_curves(
    media_mix_model: lightweight_mmm.LightweightMMM,
    target_scaler: Optional[preprocessing.CustomScaler] = None,
    steps: int = 100,
    percentage_add: float = 0.1) -> matplotlib.figure.Figure:
  """Plots the response curves of each media channel based on the model.

  Args:
    media_mix_model: Media mix model to use for plotting the response curves.
    target_scaler: Scaler used for scaling the target, to unscaled values and
      plot in the original scale.
    steps: Number of steps to simulate.
    percentage_add: Percentage too exceed the maximum historic spend for the
      simulation of the response curve.

  Returns:
    Plot of Response curve.
  """
  if not hasattr(media_mix_model, "trace"):
    raise lightweight_mmm.NotFittedModelError(
        "Model needs to be fit first before attempting to plot its response "
        "curves.")
  media_maxes = media_mix_model.media.max(axis=0) * (1 + percentage_add)
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

  mock_media = media_ranges * diagonal
  predictions = make_predictions(media_mix_model, mock_media, extra_features)
  predictions = jnp.transpose(jnp.squeeze(a=predictions))
  if target_scaler:
    predictions = target_scaler.inverse_transform(predictions)

  fig, ax = plt.subplots(1, 1)
  for i in range(media_ranges.shape[2]):
    ax = sns.lineplot(
        x=jnp.squeeze(media_ranges)[:, i],
        y=predictions[:, i],
        label=media_mix_model.media_names[i])
  ax.set_title("Response curves")
  ax.set_ylabel("KPI")
  ax.set_xlabel("Spend per channel")
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
  return fig


def plot_model_fit(media_mix_model: lightweight_mmm.LightweightMMM,
                   target_scaler: Optional[jnp.array] = None,
                   interval_mid_range: float = .9) -> matplotlib.figure.Figure:
  """Plots the ground truth, predicted value and interval for the training data.

  Model needs to be fit before calling this function to plot.

  Args:
    media_mix_model: Media mix model.
    target_scaler: Scaler used for scaling the target, to unscaled values and
      plot in the original scale.
    interval_mid_range: Mid range interval to take for plotting. Eg. .9 will use
      .05 and .95 as the lower and upper quantiles. Must be a float number.
      between 0 and 1.

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

  upper_quantile = 1 - (1 - interval_mid_range) / 2
  lower_quantile = (1 - interval_mid_range) / 2
  upper_bound = jnp.quantile(a=posterior_pred, q=upper_quantile, axis=0)
  lower_bound = jnp.quantile(a=posterior_pred, q=lower_quantile, axis=0)

  r2, _ = arviz.r2_score(y_true=target_train, y_pred=posterior_pred)

  fig, ax = plt.subplots(1, 1)
  ax.plot(jnp.arange(target_train.shape[0]), target_train, c="grey", alpha=.9)
  ax.plot(
      jnp.arange(target_train.shape[0]),
      posterior_pred.mean(axis=0),
      c="green",
      alpha=.9)
  ax.fill_between(
      x=jnp.arange(target_train.shape[0]),
      y1=lower_bound,
      y2=upper_bound,
      alpha=.35,
      color="green")
  ax.legend(["True KPI", "Predicted KPI"])
  ax.yaxis.grid(color="gray", linestyle="dashed", alpha=0.3)
  ax.xaxis.grid(color="gray", linestyle="dashed", alpha=0.3)
  ax.title.set_text(f"True and predicted KPI.\n R2 = {r2}")
  return fig


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
  if not channel_names:
    channel_names = np.arange(np.shape(media_channel_posteriors)[1])
  fig, axes = plt.subplots(n_rows, n_columns, figsize=(10, 10))
  for index, ax in enumerate(axes.flatten()[:n_media_channels]):
    ax = arviz.plot_kde(
        media_channel_posteriors[:, index], quantiles=quantiles, ax=ax)
    ax.set_xlabel(f"media channel {channel_names[index]}")

  for index, ax in enumerate(axes.flatten()[n_media_channels:]):
    fig.delaxes(ax)
  fig.tight_layout()
  return fig


def plot_bars_media_effects(
    effect: jnp.ndarray,
    channel_names: Optional[Sequence[Any]] = None,
    interval_mid_range: float = .9
    ) -> matplotlib.figure.Figure:
  """Plots a barchart of estimated media effects with their percentile interval.

  The lower and upper percentile need to be between 0-1.

  Args:
    effect: Estimated media effects as returned by
      lightweight_mmm.get_posterior_metrics()
    channel_names: Names of media channels to be added to plot.
    interval_mid_range: Mid range interval to take for plotting. Eg. .9 will use
      .05 and .95 as the lower and upper quantiles. Must be a float number.

  Returns:
    Barplot of estimated media effects with defined percentile-bars.
  """
  if not channel_names:
    channel_names = np.arange(np.shape(effect)[1])
  upper_quantile = 1 - (1 - interval_mid_range) / 2
  lower_quantile = (1 - interval_mid_range) / 2

  fig, ax = plt.subplots(1, 1)
  sns.barplot(data=effect, ci=None, ax=ax)
  quantile_bounds = np.quantile(
      effect, q=[lower_quantile, upper_quantile], axis=0)
  quantile_bounds[0] = effect.mean(axis=0) - quantile_bounds[0]
  quantile_bounds[1] = quantile_bounds[1] - effect.mean(axis=0)

  ax.errorbar(
      x=np.arange(np.shape(effect)[1]),
      y=effect.mean(axis=0),
      yerr=quantile_bounds,
      fmt="none",
      c="black")
  fig.suptitle(
      f"Estimated media channel effects, error bars show {lower_quantile} - {upper_quantile} credibility interval"
  )
  return fig
