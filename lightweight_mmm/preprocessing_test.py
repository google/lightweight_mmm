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

"""Tests for preprocessing."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

from lightweight_mmm.lightweight_mmm import preprocessing


class PreprocessingTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="all_nones",
          divide_operation=None,
          divide_by=None,
          multiply_operation=None,
          multiply_by=None),
      dict(
          testcase_name="both_divides",
          divide_operation=1,
          divide_by=1,
          multiply_operation=None,
          multiply_by=None),
      dict(
          testcase_name="both_multiplies",
          divide_operation=None,
          divide_by=None,
          multiply_operation=1,
          multiply_by=1),
  ])
  def test_custom_scaler_constructor_wrong_params_raises_valueerror(
      self, divide_operation, divide_by, multiply_operation, multiply_by):
    with self.assertRaises(ValueError):
      preprocessing.CustomScaler(
          divide_operation=divide_operation,
          divide_by=divide_by,
          multiply_operation=multiply_operation,
          multiply_by=multiply_by)

  @parameterized.named_parameters([
      dict(
          testcase_name="1",
          divide_operation=jnp.mean,
          divide_by=1,
          multiply_operation=jnp.mean,
          multiply_by=1,
          has_attributes=["divide_operation", "multiply_operation"],
          missing_attributes=["divide_by", "multiply_by"]),
      dict(
          testcase_name="2",
          divide_operation=jnp.mean,
          divide_by=None,
          multiply_operation=jnp.mean,
          multiply_by=1,
          has_attributes=["divide_operation", "multiply_operation"],
          missing_attributes=["divide_by", "multiply_by"]),
      dict(
          testcase_name="3",
          divide_operation=jnp.mean,
          divide_by=1,
          multiply_operation=jnp.mean,
          multiply_by=None,
          has_attributes=["divide_operation", "multiply_operation"],
          missing_attributes=["divide_by", "multiply_by"]),
      dict(
          testcase_name="4",
          divide_operation=jnp.mean,
          divide_by=None,
          multiply_operation=jnp.mean,
          multiply_by=None,
          has_attributes=["divide_operation", "multiply_operation"],
          missing_attributes=["divide_by", "multiply_by"]),
      dict(
          testcase_name="5",
          divide_operation=None,
          divide_by=5,
          multiply_operation=None,
          multiply_by=5,
          has_attributes=["divide_by", "multiply_by"],
          missing_attributes=["divide_operation", "multiply_operation"]),
      dict(
          testcase_name="6",
          divide_operation=jnp.mean,
          divide_by=5,
          multiply_operation=None,
          multiply_by=5,
          has_attributes=["divide_operation", "multiply_by"],
          missing_attributes=["divide_by", "multiply_operation"]),
      dict(
          testcase_name="7",
          divide_operation=None,
          divide_by=5,
          multiply_operation=jnp.mean,
          multiply_by=5,
          has_attributes=["divide_by", "multiply_operation"],
          missing_attributes=["divide_operation", "multiply_by"]),
  ])
  def test_custom_scaler_constructor_sets_correct_attributes(
      self, divide_operation, divide_by, multiply_operation, multiply_by,
      has_attributes, missing_attributes):
    custom_scaler = preprocessing.CustomScaler(
        divide_operation=divide_operation,
        divide_by=divide_by,
        multiply_operation=multiply_operation,
        multiply_by=multiply_by)

    for attribute in has_attributes:
      self.assertTrue(hasattr(custom_scaler, attribute))

    for attribute in missing_attributes:
      self.assertFalse(hasattr(custom_scaler, attribute))

  @parameterized.named_parameters([
      dict(
          testcase_name="1",
          divide_operation=jnp.mean,
          divide_by=[1, 1, 1],
          multiply_operation=jnp.mean,
          multiply_by=[1, 1, 1],
          expected_divide_by=[2, 2, 2],
          expected_multiply_by=[2, 2, 2]),
      dict(
          testcase_name="2",
          divide_operation=None,
          divide_by=[1, 1, 1],
          multiply_operation=jnp.mean,
          multiply_by=[1, 1, 1],
          expected_divide_by=[1, 1, 1],
          expected_multiply_by=[2, 2, 2]),
      dict(
          testcase_name="3",
          divide_operation=jnp.mean,
          divide_by=[1, 1, 1],
          multiply_operation=None,
          multiply_by=[1, 1, 1],
          expected_divide_by=[2, 2, 2],
          expected_multiply_by=[1, 1, 1]),
      dict(
          testcase_name="4",
          divide_operation=None,
          divide_by=[1, 1, 1],
          multiply_operation=None,
          multiply_by=[1, 1, 1],
          expected_divide_by=[1, 1, 1],
          expected_multiply_by=[1, 1, 1]),
  ])
  def test_fit_overrides_or_sets_correct_values(self, divide_operation,
                                                divide_by, multiply_operation,
                                                multiply_by, expected_divide_by,
                                                expected_multiply_by):
    data = jnp.ones((10, 3)) * 2
    custom_scaler = preprocessing.CustomScaler(
        divide_operation=divide_operation,
        divide_by=jnp.array(divide_by),
        multiply_operation=multiply_operation,
        multiply_by=jnp.array(multiply_by))

    custom_scaler.fit(data)

    self.assertTrue(hasattr(custom_scaler, "divide_by"))
    self.assertTrue(hasattr(custom_scaler, "multiply_by"))
    np.testing.assert_array_equal(custom_scaler.divide_by,
                                  jnp.array(expected_divide_by))
    np.testing.assert_array_equal(custom_scaler.multiply_by,
                                  jnp.array(expected_multiply_by))

  @parameterized.named_parameters([
      dict(
          testcase_name="1",
          multiply_by=1,
          divide_by=1,
          expected_transformed=[[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]),
      dict(
          testcase_name="2",
          multiply_by=5,
          divide_by=8,
          expected_transformed=[[0., 0.625, 1.25], [1.875, 2.5, 3.125],
                                [3.75, 4.375, 5.]]),
      dict(
          testcase_name="3",
          multiply_by=2,
          divide_by=1,
          expected_transformed=[[0., 2., 4.], [6., 8., 10.], [12., 14., 16.]]),
      dict(
          testcase_name="4",
          multiply_by=1,
          divide_by=4,
          expected_transformed=[[0., 0.25, 0.5], [0.75, 1., 1.25],
                                [1.5, 1.75, 2.]]),
      dict(
          testcase_name="5",
          multiply_by=[1, 2, 3],
          divide_by=[1, 2, 3],
          expected_transformed=[[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]),
      dict(
          testcase_name="6",
          multiply_by=[1, 1, 1],
          divide_by=[3, 2, 3],
          expected_transformed=[[0., 0.5, 0.66666667], [1., 2., 1.66666667],
                                [2., 3.5, 2.66666667]]),
      dict(
          testcase_name="7",
          multiply_by=[1, 2, 3],
          divide_by=[3, 2, 1],
          expected_transformed=[[0., 1., 6.], [1., 4., 15.], [2., 7., 24.]]),
      dict(
          testcase_name="8",
          multiply_by=[1, 1, 1],
          divide_by=[1, 1, 1],
          expected_transformed=[[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]),
  ])
  def test_transform_produces_correct_values(self, multiply_by, divide_by,
                                             expected_transformed):
    data = jnp.arange(9).reshape((3, 3))

    if isinstance(multiply_by, int) and isinstance(divide_by, int):
      scaler = preprocessing.CustomScaler(
          divide_by=divide_by, multiply_by=multiply_by)
    else:
      scaler = preprocessing.CustomScaler(
          divide_by=jnp.array(divide_by), multiply_by=jnp.array(multiply_by))
    scaler.fit(data)
    transformed_data = scaler.transform(data)

    np.testing.assert_array_almost_equal(transformed_data,
                                         jnp.array(expected_transformed))

  @parameterized.named_parameters([
      dict(
          testcase_name="one_one",
          multiply_by=1,
          divide_by=1,
          expected_transformed=[[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]),
      dict(
          testcase_name="five_eight",
          multiply_by=5,
          divide_by=8,
          expected_transformed=[[0., 0.625, 1.25], [1.875, 2.5, 3.125],
                                [3.75, 4.375, 5.]]),
      dict(
          testcase_name="two_one",
          multiply_by=2,
          divide_by=1,
          expected_transformed=[[0., 2., 4.], [6., 8., 10.], [12., 14., 16.]]),
      dict(
          testcase_name="one_four",
          multiply_by=1,
          divide_by=4,
          expected_transformed=[[0., 0.25, 0.5], [0.75, 1., 1.25],
                                [1.5, 1.75, 2.]]),
      dict(
          testcase_name="arange_arange",
          multiply_by=[1, 2, 3],
          divide_by=[1, 2, 3],
          expected_transformed=[[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]),
      dict(
          testcase_name="ones_arange",
          multiply_by=[1, 1, 1],
          divide_by=[3, 2, 3],
          expected_transformed=[[0., 0.5, 0.66666667], [1., 2., 1.66666667],
                                [2., 3.5, 2.66666667]]),
      dict(
          testcase_name="arange_invarange",
          multiply_by=[1, 2, 3],
          divide_by=[3, 2, 1],
          expected_transformed=[[0., 1., 6.], [1., 4., 15.], [2., 7., 24.]]),
      dict(
          testcase_name="ones_ones",
          multiply_by=[1, 1, 1],
          divide_by=[1, 1, 1],
          expected_transformed=[[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]),
  ])
  def test_fit_transform_produces_correct_values(self, multiply_by, divide_by,
                                                 expected_transformed):
    data = jnp.arange(9).reshape((3, 3))

    scaler = preprocessing.CustomScaler(
        divide_by=jnp.array(divide_by), multiply_by=jnp.array(multiply_by))
    transformed_data = scaler.fit_transform(data)

    np.testing.assert_array_almost_equal(transformed_data,
                                         jnp.array(expected_transformed))

  @parameterized.named_parameters([
      dict(testcase_name="one_one", multiply_by=1, divide_by=1),
      dict(testcase_name="five_eight", multiply_by=5, divide_by=8),
      dict(testcase_name="two_one", multiply_by=2, divide_by=1),
      dict(testcase_name="one_four", multiply_by=1, divide_by=4),
      dict(
          testcase_name="arange_arange",
          multiply_by=[1, 2, 3],
          divide_by=[1, 2, 3]),
      dict(
          testcase_name="ones_arange",
          multiply_by=[1, 1, 1],
          divide_by=[3, 2, 3]),
      dict(
          testcase_name="arange_invarange",
          multiply_by=[1, 2, 3],
          divide_by=[3, 2, 1]),
      dict(
          testcase_name="ones_ones", multiply_by=[1, 1, 1], divide_by=[1, 1,
                                                                       1]),
  ])
  def test_reverse_transform_returns_original_values(self, multiply_by,
                                                     divide_by):
    data = jnp.arange(9).reshape((3, 3))

    scaler = preprocessing.CustomScaler(
        divide_by=jnp.array(divide_by), multiply_by=jnp.array(multiply_by))
    transformed_data = scaler.fit_transform(data)
    inverse_transformed_data = scaler.inverse_transform(transformed_data)

    np.testing.assert_array_almost_equal(data, inverse_transformed_data)


if __name__ == "__main__":
  absltest.main()
