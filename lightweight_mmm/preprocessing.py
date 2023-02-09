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

"""Utilities for preprocessing dataset for training LightweightMMM."""

import copy
from typing import Callable, List, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import pandas as pd
from sklearn import base

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from lightweight_mmm.core import core_utils


class NotFittedScalerError(Exception):
  pass


class CustomScaler(base.TransformerMixin):
  """Class to scale your data based on multiplications and divisions.

  This scaler can be used in two fashions for both the multiplication and
  division operation.
  - By specifying a value to use for the scaling operation.
  - By specifying an operation used at column level to calculate the value
    for the actual scaling operation.

  Eg. if one wants to scale the dataset by multiply by 100 you can directly
  pass multiply_by=100. Value can also be an array with as many values
  as column has the data being scaled. But if you want to multiply by the mean
  value of each column, then you can pass multiply_operation=jnp.mean (or any
  other operation desired).

  Operation parameters have the upper hand in the cases where both values and
  operations are passed, values will be ignored in this case.

  Scaler must be fit first in order to call the transform method.

  Attributes.
    divide_operation: Operation to apply over axis 0 of the fitting data to
      obtain the value that will be used for division during scaling.
    divide_by: Numbers(s) by which to divide data in the scaling process. Since
      the scaler is applied to axis 0 of the data, the shape of divide_by must
      be consistent with division into the data. For example, if data.shape =
      (100, 3, 5) then divide_by.shape can be (3, 5) or (5,) or a number. If
      divide_operation is given, this divide_by value will be ignored.
    multiply_operation: Operation to apply over axis 0 of the fitting data to
      obtain the value that will be used for multiplication during scaling.
    multiply_by: Numbers(s) by which to multiply data in the scaling process.
      Since the scaler is applied to axis 0 of the data, the shape of
      multiply_by must be consistent with multiplication into the data. For
      example, if data.shape = (100, 3, 5) then multiply_by.shape can be (3, 5)
      or (5,) or a number. If multiply_operation is given, this multiply_by
      value will be ignored.
  """

  def __init__(
      self,
      divide_operation: Optional[Callable[[jnp.ndarray], jnp.float32]] = None,
      divide_by: Optional[Union[float, int, jnp.ndarray]] = 1,
      multiply_operation: Optional[Callable[[jnp.ndarray], jnp.float32]] = None,
      multiply_by: Optional[Union[float, int, jnp.ndarray]] = 1.) -> None:
    """Constructor for the CustomScaler class."""
    if all([
        divide_by is None, divide_operation is None, multiply_by is None,
        multiply_operation is None
    ]):
      raise ValueError("No values for transformations were provided and this "
                       "scaler will fail. Please instantiate a valid one")

    if divide_operation is None and divide_by is None:
      raise ValueError(
          "Either a division operation or value needs to be passed. If "
          "you dont want to use a division to scale your data just "
          "pass divide_by=1.")
    elif divide_operation is not None:
      self.divide_operation = divide_operation
    else:
      self.divide_by = divide_by

    if multiply_operation is None and multiply_by is None:
      raise ValueError(
          "Either a multiplication operation or value needs to be passed. If "
          "you dont want to use a multiplication to scale your data just "
          "pass multiply_by=1.")
    elif multiply_operation is not None:
      self.multiply_operation = multiply_operation
    else:
      self.multiply_by = multiply_by

  def fit(self, data: jnp.ndarray) -> None:
    """Figures out values for transformations based on the specified operations.

    Args:
      data: Input dataset to use for fitting.
    """
    if hasattr(self, "divide_operation"):
      self.divide_by = jnp.apply_along_axis(
          func1d=self.divide_operation, axis=0, arr=data)
    elif isinstance(self.divide_by, int) or isinstance(self.divide_by, float):
      self.divide_by = self.divide_by * jnp.ones(data.shape[1:])
    if hasattr(self, "multiply_operation"):
      self.multiply_by = jnp.apply_along_axis(
          func1d=self.multiply_operation, axis=0, arr=data)
    elif isinstance(self.multiply_by, int) or isinstance(
        self.multiply_by, float):
      self.multiply_by = self.multiply_by * jnp.ones(data.shape[1:])

  def transform(self, data: jnp.ndarray) -> jnp.ndarray:
    """Applies transformation based on fitted values.

    It can only be called if scaler was fit first.

    Args:
      data: Input dataset to transform.

    Returns:
      Transformed array.
    """
    if not hasattr(self, "divide_by") or not hasattr(self, "multiply_by"):
      raise NotFittedScalerError(
          "transform is called without fit being called previously. Please "
          "fit scaler first.")
    return self.multiply_by * data / self.divide_by

  def fit_transform(self, data: jnp.ndarray) -> jnp.ndarray:
    """Fits the values and applies transformation to the input data.

    Args:
      data: Input dataset.

    Returns:
      Transformed array.
    """
    self.fit(data)
    return self.transform(data)

  def inverse_transform(self, data: jnp.ndarray) -> jnp.ndarray:
    """Runs inverse transformation to get original values.

    Args:
      data: Input dataset.

    Returns:
      Dataset with the inverse transformation applied.
    """
    return self.divide_by * data / self.multiply_by


def _compute_correlations(
    features: jnp.ndarray,
    target: jnp.ndarray,
    feature_names: List[str],
    ) -> List[pd.DataFrame]:
  """Computes feature-feature and feature-target correlations.

  Helper function for DataQualityCheck.

  Args:
    features: Features for media mix model (media and non-media variables).
    target: Target variable for media mix model.
    feature_names: Names of media channels to be added to the output dataframes.

  Returns:
    List of dataframes containing Pearson correlation coefficients between each
      feature, as well as between features and the target variable. For
      national-level data the list contains just one dataframe, and for
      geo-level data the list contains one dataframe for each geo.

  Raises:
    ValueError: If features and target have incompatible shapes (e.g. one is
      geo-level and the other national-level).
  """
  if not ((features.ndim == 2 and target.ndim == 1) or
          (features.ndim == 3 and target.ndim == 2)):
    raise ValueError(f"Incompatible shapes between features {features.shape}"
                     f" and target {target.shape}.")

  number_of_geos = core_utils.get_number_geos(features)
  correlation_matrix_output = []
  for i_geo in range(number_of_geos):

    if number_of_geos == 1:
      features_and_target = jnp.concatenate(
          [features, jnp.expand_dims(target, axis=1)], axis=1)
    else:
      features_and_target = jnp.concatenate(
          [features[:, :, i_geo],
           jnp.expand_dims(target[:, i_geo], axis=1)],
          axis=1)

    covariance_matrix = jnp.cov(features_and_target, rowvar=False)
    standard_deviations = jnp.std(features_and_target, axis=0, ddof=1)
    correlation_matrix = covariance_matrix / jnp.outer(standard_deviations,
                                                       standard_deviations)
    correlation_matrix = pd.DataFrame(
        correlation_matrix,
        columns=feature_names + ["target"],
        index=feature_names + ["target"],
        dtype=float)
    correlation_matrix_output.append(correlation_matrix)

  return correlation_matrix_output


def _compute_variances(
    features: jnp.ndarray,
    feature_names: Sequence[str],
    geo_names: Sequence[str],
) -> pd.DataFrame:
  """Computes variances over time for each feature.

  In general, higher variance is better since it creates more signal for the
  regression analysis. However, if the features have not been scaled (divided by
  the mean), then the variance can take any value and this analysis is not
  meaningful.

  Args:
    features: Features for media mix model (media and non-media variables).
    feature_names: Names of media channels to be added to the output dataframe.
    geo_names: Names of geos to be added to the output dataframes.

  Returns:
    Dataframe containing the variance over time for each feature. This dataframe
      contains one row per geo, and just a single row for national data.

  Raises:
    ValueError: If the number of geos in features does not match the number of
    supplied geo_names.
  """
  number_of_geos = core_utils.get_number_geos(features)

  if len(geo_names) != number_of_geos:
    raise ValueError("The number of geos in features does not match the length "
                     "of geo_names")

  variances_as_series = []
  for i_geo in range(number_of_geos):
    features_for_this_geo = features[...,
                                     i_geo] if number_of_geos > 1 else features
    variances_as_series.append(
        pd.DataFrame(data=features_for_this_geo).var(axis=0, ddof=0))

  variances = pd.concat(variances_as_series, axis=1)
  variances.columns = geo_names
  variances.index = copy.copy(feature_names)

  return variances


def _compute_spend_fractions(
    cost_data: jnp.ndarray,
    channel_names: Optional[Sequence[str]] = None,
    output_column_name: str = "fraction of spend") -> pd.DataFrame:
  """Computes fraction of total spend for each media channel.

  Args:
    cost_data: Spend (can be normalized or not) per channel.
    channel_names: Names of media channels to be added to the output dataframe.
    output_column_name: Name of the column in the output dataframe, denoting the
      fraction of the total spend in each media channel.

  Returns:
    Dataframe containing fraction of the total spend in each channel.

  Raises:
    ValueError if any of the costs are zero or negative.
  """
  cost_df = pd.DataFrame(
      cost_data, index=channel_names, columns=[output_column_name])

  if (cost_df[output_column_name] <= 0).any():
    raise ValueError("Values in cost_data must all be positive.")

  normalized_cost_df = cost_df.div(cost_df.sum(axis=0), axis=1).round(4)
  return normalized_cost_df


def _compute_variance_inflation_factors(
    features: jnp.ndarray, feature_names: Sequence[str],
    geo_names: Sequence[str]) -> pd.DataFrame:
  """Computes variance inflation factors for all features.

  Helper function for DataQualityCheck.

  Args:
    features: Features for media mix model (media and non-media variables).
    feature_names: Names of media channels to be added to the output dataframe.
    geo_names: Names of geos to be added to the output dataframes.

  Returns:
    Dataframe containing variance inflation factors for each feature. For
      national-level data the dataframe contains just one column, and for
      geo-level data the list contains one column for each geo.

  Raises:
    ValueError: If the number of geos in features does not match the number of
    supplied geo_names.
  """
  number_of_geos = core_utils.get_number_geos(features)

  if len(geo_names) != number_of_geos:
    raise ValueError("The number of geos in features does not match the length "
                     "of geo_names")

  vifs_for_each_geo = []
  for i_geo in range(number_of_geos):
    features_for_this_geo = features[...,
                                     i_geo] if number_of_geos > 1 else features
    features_for_this_geo = add_constant(
        pd.DataFrame(features_for_this_geo, dtype=float), has_constant="skip")

    vifs_for_this_geo = []
    for i, feature in enumerate(features_for_this_geo.columns):
      if feature != "const":
        vifs_for_this_geo.append(
            variance_inflation_factor(features_for_this_geo.values, i))

    vifs_for_each_geo.append(vifs_for_this_geo)

  vif_df = pd.DataFrame(data=zip(*vifs_for_each_geo), dtype=float)
  vif_df.columns = geo_names
  vif_df.index = copy.copy(feature_names)

  return vif_df


def check_data_quality(
    media_data: jnp.ndarray,
    target_data: jnp.ndarray,
    cost_data: jnp.ndarray,
    extra_features_data: Optional[jnp.ndarray] = None,
    channel_names: Optional[Sequence[str]] = None,
    extra_features_names: Optional[Sequence[str]] = None,
    geo_names: Optional[Sequence[str]] = None,
    ) -> Tuple[List[pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Checks LMMM data quality, to be used before fitting a model.

  Args:
    media_data: National-level or geo-level media impressions data, such as
      media_data_train or media_data in the example Collaboratory. This dataset
      should be scaled so that it has a similar order of magnitude to the
      target_data and extra_features_data (if applicable).
    target_data: National-level or geo-level sales or revenue data, such as
      target_train or target in the example Colabs. This dataset should be
      scaled so that it has a similar order of magnitude to the media_data and
      extra_features_data (if applicable).
    cost_data: National-level cost data, identified as "costs" in the example
      Colabs, with one value per media channel denoting the total cost for that
      channel over the time period covered by the media_data. The costs can be
      scaled (mean-normalized) or not scaled.
    extra_features_data: Optional national-level or geo-level extra features
      data, such as extra_features_train or extra_features in the example
      Colabs. This dataset should be scaled so that it has a similar order of
      magnitude to the media_data and target_data.
    channel_names: Names of media channels to be added to the output dataframes.
    extra_features_names: Names of extra features to be added to the output
      dataframes.
    geo_names: Names of geos to be added to the output dataframes.

  Returns:
    correlations: List of dataframes containing Pearson correlation coefficients
      between each feature, as well as between features and the target variable.
      For national-level data the list contains just one dataframe, and for
      geo-level data the list contains one dataframe for each geo.
    variances: Dataframe containing the variance over time for each feature. For
      national-level data the dataframe contains just one column, and for
      geo-level data the list contains one column for each geo.
    spend_fractions: Dataframe containing fraction of the total spend in each
      channel.
    variance_inflation_factors: Dataframes containing variance inflation factors
      for each feature. For national-level data the dataframe contains just one
      column, and for geo-level data the list contains one column for each geo.

  Raises:
    ValueError: If the number of channel_names does not match size of media_data
      or cost_data, or if the number of extra_features_names does not match size
      of extra_features_data.
  """

  if channel_names is not None and media_data.shape[1] != len(channel_names):
    raise ValueError("Number of channels in media_data does not match length "
                     "of channel_names.")

  if channel_names is not None and len(cost_data) != len(channel_names):
    raise ValueError("Number of channels in cost_data does not match length "
                     "of channel_names.")

  if (extra_features_data is not None and
      extra_features_names is not None and
      extra_features_data.shape[1] != len(extra_features_names)):
    raise ValueError("Number of features in extra_features_data does not match "
                     "length of extra_features_names.")

  if channel_names is None:
    all_features_names = [f"feature_{i}" for i in range(media_data.shape[1])]
  else:
    all_features_names = list(channel_names)

  if geo_names is None:
    geo_names = [
        f"geo_{i}" for i in range(core_utils.get_number_geos(media_data))
    ]

  # Spend fractions are computed for the media channels only, so we run this
  # before concatentating the extra_features_names.
  spend_fractions = _compute_spend_fractions(cost_data, all_features_names)

  if extra_features_data is not None:
    all_features_data = jnp.concatenate(
        [media_data, extra_features_data], axis=1
    )
    if extra_features_names is None:
      extra_features_names = [
          f"extra_feature_{i}" for i in range(extra_features_data.shape[1])
      ]
    all_features_names += list(extra_features_names)
  else:
    all_features_data = jnp.array(media_data)

  correlations = _compute_correlations(
      features=all_features_data,
      target=target_data,
      feature_names=all_features_names)

  variance_inflation_factors = _compute_variance_inflation_factors(
      features=all_features_data,
      feature_names=all_features_names,
      geo_names=geo_names)

  variances = _compute_variances(
      features=all_features_data,
      feature_names=all_features_names,
      geo_names=geo_names)

  # TODO(): clean up output list
  return correlations, variances, spend_fractions, variance_inflation_factors
