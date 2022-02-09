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

"""Utilities for optimizing your media based on media mix models."""
import functools
from typing import List, Optional, Tuple, Union
from absl import logging
import jax
import jax.numpy as jnp
from scipy import optimize

from lightweight_mmm.lightweight_mmm import lightweight_mmm
from lightweight_mmm.lightweight_mmm import preprocessing


@functools.partial(
    jax.jit,
    static_argnames=("media_mix_model", "media_input_shape", "target_scaler",
                     "media_scaler"))
def _objective_function(extra_features: jnp.ndarray,
                        media_mix_model: lightweight_mmm.LightweightMMM,
                        media_input_shape: Tuple[int,
                                                 int], media_gap: Optional[int],
                        target_scaler: Optional[preprocessing.CustomScaler],
                        media_scaler: preprocessing.CustomScaler,
                        media_values: jnp.ndarray) -> jnp.float64:
  """Objective function to calculate the sum of all predictions of the model.

  Args:
    extra_features: Extra features the model requires for prediction.
    media_mix_model: Media mix model to use. Must have a predict method to be
      used.
    media_input_shape: Input shape of the data required by the model to get
      predictions. This is needed since optimization might flatten some arrays
      and they need to be reshaped before running new predictions.
    media_gap: Media data gap between the end of training data and the start of
      the out of sample media given. Eg. if 100 weeks of data were used for
      training and prediction starts 2 months after training data finished we
      need to provide the 8 weeks missing between the training data and the
      prediction data so data transformations (adstock, carryover, ...) can take
      place correctly.
    target_scaler: Scaler that was used to scale the target before training.
    media_scaler: Scaler that was used to scale the media data before training.
    media_values: Media values required by the model to run predictions.

  Returns:
    The negative value of the sum of all predictions.
  """
  media_values = jnp.tile(
      media_values / media_input_shape[0], reps=media_input_shape[0])
  media_values = jnp.reshape(a=media_values, newshape=media_input_shape)
  media_values = media_scaler.transform(media_values)
  return -jnp.sum(
      media_mix_model.predict(
          media=media_values.reshape(media_input_shape),
          extra_features=extra_features,
          media_gap=media_gap,
          target_scaler=target_scaler).mean(axis=0))


@jax.jit
def _budget_constraint(media: jnp.ndarray, prices: jnp.ndarray,
                       budget: jnp.ndarray) -> jnp.float64:
  """Calculates optimization constraint to keep spend equal to the budget.

  Args:
    media: Array with the values of the media for this iteration.
    prices: Prices of each media channel at any given time.
    budget: Total budget of the optimization.

  Returns:
    The result from substracting the total spending and the budget.
  """
  media = media.reshape((-1, len(prices)))
  return jnp.sum(media * prices) - budget


def _get_lower_and_upper_bounds(
    media: jnp.ndarray,
    n_time_periods: int,
    lower_pct: float,
    upper_pct: float,
    media_scaler: Optional[preprocessing.CustomScaler] = None
) -> List[Tuple[float, float]]:
  """Gets the lower and upper bounds for optimisation based on historic data.

  It creates an upper bound based on a percentage above the maximum value on
  each channel and a lower bound based on a relative decrease of the minimum
  value.

  Args:
    media: Media data to get historic maxes and mins.
    n_time_periods: Number of time periods to optimize for. If model is built on
      weekly data, this would be the number of weeks ahead to optimize.
    lower_pct: Relative percentage decrease from the min value to consider as
      new lower bound.
    upper_pct: Relative percentage increase from the max value to consider as
      new upper bound.
    media_scaler: Scaler that was used to scale the media data before training.

  Returns:
    A list of tuples with the lower and upper bound for each media channel.
  """
  lower_bounds = jnp.maximum(media.min(axis=0) *
                             (1 - lower_pct), 0) * n_time_periods
  upper_bounds = (media.max(axis=0) * (1 + upper_pct)) * n_time_periods
  if media_scaler:
    lower_bounds = media_scaler.inverse_transform(lower_bounds)
    upper_bounds = media_scaler.inverse_transform(upper_bounds)
  return list(zip(lower_bounds.tolist(), upper_bounds.tolist()))


def _generate_starting_values(
    n_time_periods: int,
    media: jnp.ndarray,
    media_scaler: preprocessing.CustomScaler,
    budget: Union[float, int]
    ) -> jnp.ndarray:
  """Generates starting values based on historic allocation and budget.

  In order to make a comparison we can take the allocation of the last
  `n_time_periods` and scale it based on the given budget. Given this, one can
  compare how this initial values (based on average historic allocation) compare
  to the output of the optimisation in terms of sales/KPI.

  Args:
    n_time_periods: Number of time periods the optimization will be done with.
    media: Historic media data the model was trained with.
    media_scaler: Scaler that was used to scale the media data before training.
    budget: Total budget to allocate during the optimization time.

  Returns:
    An array with the starting value for each media channel for the
      optimization.
  """
  previous_allocation = media.mean(axis=0) * n_time_periods
  if media_scaler:
    previous_allocation = media_scaler.inverse_transform(previous_allocation)

  multiplier = budget / previous_allocation.sum()
  return previous_allocation * multiplier


def find_optimal_budgets(
    n_time_periods: int,
    media_mix_model: lightweight_mmm.LightweightMMM,
    budget: Union[float, int],
    prices: jnp.ndarray,
    extra_features: Optional[jnp.ndarray] = None,
    media_gap: Optional[jnp.ndarray] = None,
    target_scaler: Optional[preprocessing.CustomScaler] = None,
    media_scaler: Optional[preprocessing.CustomScaler] = None,
    bounds_lower_pct: Union[float, jnp.ndarray] = .2,
    bounds_upper_pct: Union[float, jnp.ndarray] = .2,
    max_iterations: int = 500) -> optimize.OptimizeResult:
  """Finds the best media allocation based on MMM model, prices and a budget.

  Args:
    n_time_periods: Number of time periods to optimize for. If model is built on
      weekly data, this would be the number of weeks ahead to optimize.
    media_mix_model: Media mix model to use for the optimization.
    budget: Total budget to allocate during the optimization time.
    prices: An array with shape (n_media_channels,) for the cost of each media
      channel unit.
    extra_features: Extra features needed for the model to predict.
    media_gap: Media data gap between the end of training data and the start of
      the out of sample media given. Eg. if 100 weeks of data were used for
      training and prediction starts 8 weeks after training data finished we
      need to provide the 8 weeks missing between the training data and the
      prediction data so data transformations (adstock, carryover, ...) can take
      place correctly.
    target_scaler: Scaler that was used to scale the target before training.
    media_scaler: Scaler that was used to scale the media data before training.
    bounds_lower_pct: Relative percentage decrease from the min value to
      consider as new lower bound.
    bounds_upper_pct: Relative percentage increase from the max value to
      consider as new upper bound.
    max_iterations: Number of max iterations to use for the SLSQP scipy
      optimizer. Default is 500.

  Returns:
    OptimizeResult object containing the results of the optimization.
  """
  if not hasattr(media_mix_model, "media"):
    raise ValueError(
        "The passed model has not been trained. Please fit the model before "
        "running optimization.")
  jax.config.update("jax_enable_x64", True)

  if isinstance(bounds_lower_pct, float):
    bounds_lower_pct = jnp.repeat(a=bounds_lower_pct, repeats=len(prices))
  if isinstance(bounds_upper_pct, float):
    bounds_upper_pct = jnp.repeat(a=bounds_upper_pct, repeats=len(prices))

  bounds = _get_lower_and_upper_bounds(
      media=media_mix_model.media,
      n_time_periods=n_time_periods,
      lower_pct=bounds_lower_pct,
      upper_pct=bounds_upper_pct,
      media_scaler=media_scaler)

  if sum([lower_bound for lower_bound, _ in bounds]) > budget:
    logging.warning(
        "Budget given is smaller than the lower bounds of the constraints for "
        "optimization. This will lead to faulty optimization. Please either "
        "increase the budget or change the lower bound by increasing the "
        "percentage decrease with the `bounds_lower_pct` parameter.")
  if sum([upper_bound for _, upper_bound in bounds]) < budget:
    logging.warning(
        "Budget given is larger than the upper bounds of the constraints for "
        "optimization. This will lead to faulty optimization. Please either "
        "reduce the budget or change the upper bound by increasing the "
        "percentage increase with the `bounds_upper_pct` parameter.")

  starting_values = _generate_starting_values(n_time_periods=n_time_periods,
                                              media=media_mix_model.media,
                                              media_scaler=media_scaler,
                                              budget=budget)

  if not media_scaler:
    media_scaler = preprocessing.CustomScaler(multiply_by=1, divide_by=1)
  media_input_shape = (n_time_periods, media_mix_model.n_media_channels)
  partial_objective_function = functools.partial(
      _objective_function, extra_features, media_mix_model,
      media_input_shape, media_gap,
      target_scaler, media_scaler)

  solution = optimize.minimize(
      fun=partial_objective_function,
      x0=starting_values,
      bounds=bounds,
      method="SLSQP",
      options={
          "maxiter": max_iterations,
          "disp": True
      },
      constraints={
          "type": "eq",
          "fun": _budget_constraint,
          "args": (prices, budget)
      })

  kpi_without_optim = _objective_function(extra_features=extra_features,
                                          media_mix_model=media_mix_model,
                                          media_input_shape=media_input_shape,
                                          media_gap=media_gap,
                                          target_scaler=target_scaler,
                                          media_scaler=media_scaler,
                                          media_values=starting_values)
  logging.info("KPI without optimization: %r", -1 * kpi_without_optim.item())
  logging.info("KPI with optimization: %r", -1 * solution.fun)

  jax.config.update("jax_enable_x64", False)
  return solution
