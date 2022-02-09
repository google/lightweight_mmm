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

"""Utilities for preprocessing dataset for training LightweightMMM."""

from typing import Callable, Optional, Union

import jax.numpy as jnp

from sklearn import base


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

  Attributes:
    divide_operation: Operation to apply at each column of the fitting data to
      obtain the value that will be used for division during scaling.
    divide_by: Number to divide data by in the scaling process. If data has
      multiple dimensions, divide by can be an array of the length that allows
        to divide the data. Eg: data.shape = (100, 3), divide_by.shape = (3,).
          If divide_operation is given, this divide_by value will be ignored.
    multiply_operation: Operation to apply at each column of the fitting data to
      obtain the value that will be used for multiplication during scaling.
    multiply_by: Factor to multiply data by. If data has multiple dimensions,
      multiply by can be an array of the a length that allows to divide the
      data. Eg. data.shape = (100, 3), multiply_by.shape = (3,). If
      multiply_operation is given, this multiply_by value will be ignored.
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
      if data.ndim > 1:
        self.divide_by = jnp.repeat(self.divide_by, data.shape[1])
      else:
        self.divide_by = jnp.array([self.divide_by])
    if hasattr(self, "multiply_operation"):
      self.multiply_by = jnp.apply_along_axis(
          func1d=self.multiply_operation, axis=0, arr=data)
    elif isinstance(self.multiply_by, int) or isinstance(
        self.multiply_by, float):
      if data.ndim > 1:
        self.multiply_by = jnp.repeat(self.multiply_by, data.shape[1])
      else:
        self.multiply_by = jnp.array([self.multiply_by])

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
