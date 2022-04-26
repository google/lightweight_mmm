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

from lightweight_mmm import preprocessing


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

  @parameterized.named_parameters([
      dict(
          testcase_name="one_one",
          multiply_by=1,
          divide_by=1,
          expected_transformed=[[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                                [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
                                [[18, 19, 20], [21, 22, 23], [24, 25, 26]]]),
      dict(
          testcase_name="five_eight",
          multiply_by=5,
          divide_by=8,
          expected_transformed=[[[0., 0.625, 1.25], [1.875, 2.5, 3.125],
                                 [3.75, 4.375, 5.]],
                                [[5.625, 6.25, 6.875], [7.5, 8.125, 8.75],
                                 [9.375, 10., 10.625]],
                                [[11.25, 11.875, 12.5], [13.125, 13.75, 14.375],
                                 [15., 15.625, 16.25]]]),
      dict(
          testcase_name="two_one",
          multiply_by=2,
          divide_by=1,
          expected_transformed=[[[0, 2, 4], [6, 8, 10], [12, 14, 16]],
                                [[18, 20, 22], [24, 26, 28], [30, 32, 34]],
                                [[36, 38, 40], [42, 44, 46], [48, 50, 52]]]),
      dict(
          testcase_name="one_four",
          multiply_by=1,
          divide_by=4,
          expected_transformed=[[[0., 0.25, 0.5], [0.75, 1., 1.25],
                                 [1.5, 1.75, 2.]],
                                [[2.25, 2.5, 2.75], [3., 3.25, 3.5],
                                 [3.75, 4., 4.25]],
                                [[4.5, 4.75, 5.], [5.25, 5.5, 5.75],
                                 [6., 6.25, 6.5]]]),
      dict(
          testcase_name="arange_arange",
          multiply_by=[1, 2, 3],
          divide_by=[1, 2, 3],
          expected_transformed=[[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                                [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
                                [[18, 19, 20], [21, 22, 23], [24, 25, 26]]]),
      dict(
          testcase_name="ones_arange",
          multiply_by=[1, 1, 1],
          divide_by=[3, 2, 3],
          expected_transformed=[[[0., 0.5, 0.6666667], [1., 2., 1.6666666],
                                 [2., 3.5, 2.6666667]],
                                [[3., 5., 3.6666667], [4., 6.5, 4.6666665],
                                 [5., 8., 5.6666665]],
                                [[6., 9.5, 6.6666665], [7., 11., 7.6666665],
                                 [8., 12.5, 8.666667]]]),
      dict(
          testcase_name="arange_invarange",
          multiply_by=[1, 2, 3],
          divide_by=[3, 2, 1],
          expected_transformed=[[[0, 1, 6], [1, 4, 15], [2, 7, 24]],
                                [[3, 10, 33], [4, 13, 42], [5, 16, 51]],
                                [[6, 19, 60], [7, 22, 69], [8, 25, 78]]]),
      dict(
          testcase_name="two_dimensional_multiply_by",
          multiply_by=[[1, 2, 3], [3, 2, 1], [1, 2, 3]],
          divide_by=1,
          expected_transformed=[[[0., 2., 6.], [9., 8., 5.], [6., 14., 24.]],
                                [[9., 20., 33.], [36., 26., 14.],
                                 [15., 32., 51.]],
                                [[18., 38., 60.], [63., 44., 23.],
                                 [24., 50., 78.]]]),
      dict(
          testcase_name="two_dimensional_divide_by",
          multiply_by=1,
          divide_by=[[1, 2, 3], [3, 2, 1], [1, 2, 3]],
          expected_transformed=[[[0., 0.5, 0.6666667], [1., 2., 5.],
                                 [6., 3.5, 2.6666667]],
                                [[9., 5., 3.6666667], [4., 6.5, 14.],
                                 [15., 8., 5.6666665]],
                                [[18., 9.5, 6.6666665], [7., 11., 23.],
                                 [24., 12.5, 8.666667]]]),
      dict(
          testcase_name="two_dimensional_multiply_by_and_divide_by",
          multiply_by=[[1, 2, 3], [3, 2, 1], [1, 2, 3]],
          divide_by=[[3, 2, 1], [1, 2, 3], [2, 1, 3]],
          expected_transformed=[[[0., 1., 6.], [9., 4., 1.6666666],
                                 [3., 14., 8.]],
                                [[3., 10., 33.], [36., 13., 4.6666665],
                                 [7.5, 32., 17.]],
                                [[6., 19., 60.], [63., 22., 7.6666665],
                                 [12., 50., 26.]]]),
  ])
  def test_fit_transform_produces_correct_values_in_three_dimensions(
      self, multiply_by, divide_by, expected_transformed):
    data = jnp.arange(27).reshape((3, 3, 3))

    scaler = preprocessing.CustomScaler(
        divide_by=jnp.array(divide_by), multiply_by=jnp.array(multiply_by))
    transformed_data = scaler.fit_transform(data)

    np.testing.assert_array_almost_equal(transformed_data,
                                         jnp.array(expected_transformed))

  @parameterized.named_parameters([
      dict(
          testcase_name="mean_multiply",
          multiply_operation=jnp.mean,
          divide_operation=None,
          expected_transformed=[[[0., 21., 44., 69., 96.],
                                 [125., 156., 189., 224., 261.],
                                 [300., 341., 384., 429., 476.],
                                 [525., 576., 629., 684., 741.]],
                                [[400., 441., 484., 529., 576.],
                                 [625., 676., 729., 784., 841.],
                                 [900., 961., 1024., 1089., 1156.],
                                 [1225., 1296., 1369., 1444., 1521.]],
                                [[800., 861., 924., 989., 1056.],
                                 [1125., 1196., 1269., 1344., 1421.],
                                 [1500., 1581., 1664., 1749., 1836.],
                                 [1925., 2016., 2109., 2204., 2301.]]]),
      dict(
          testcase_name="mean_divide",
          multiply_operation=None,
          divide_operation=jnp.mean,
          expected_transformed=[
              [[0., 0.04761905, 0.09090909, 0.13043478, 0.16666667],
               [0.2, 0.23076923, 0.25925925, 0.2857143, 0.31034482],
               [0.33333334, 0.3548387, 0.375, 0.3939394, 0.4117647],
               [0.42857143, 0.44444445, 0.45945945, 0.47368422, 0.4871795]],
              [[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1.]],
              [[2., 1.9523809, 1.9090909, 1.8695652, 1.8333334],
               [1.8, 1.7692307, 1.7407408, 1.7142857, 1.6896552],
               [1.6666666, 1.6451613, 1.625, 1.6060606, 1.5882353],
               [1.5714285, 1.5555556, 1.5405406, 1.5263158, 1.5128205]]
          ]),
      dict(
          testcase_name="min_multiply",
          multiply_operation=jnp.min,
          divide_operation=None,
          expected_transformed=[[[0, 1, 4, 9, 16], [25, 36, 49, 64, 81],
                                 [100, 121, 144, 169, 196],
                                 [225, 256, 289, 324, 361]],
                                [[0, 21, 44, 69, 96], [125, 156, 189, 224, 261],
                                 [300, 341, 384, 429, 476],
                                 [525, 576, 629, 684, 741]],
                                [[0, 41, 84, 129, 176],
                                 [225, 276, 329, 384, 441],
                                 [500, 561, 624, 689, 756],
                                 [825, 896, 969, 1044, 1121]]]),
      dict(
          testcase_name="max_divide",
          multiply_operation=None,
          divide_operation=jnp.max,
          expected_transformed=[
              [[0., 0.02439024, 0.04761905, 0.06976745, 0.09090909],
               [0.11111111, 0.13043478, 0.14893617, 0.16666667, 0.18367347],
               [0.2, 0.21568628, 0.23076923, 0.24528302, 0.25925925],
               [0.27272728, 0.2857143, 0.2982456, 0.31034482, 0.3220339]],
              [[0.5, 0.5121951, 0.52380955, 0.53488374, 0.54545456],
               [0.5555556, 0.5652174, 0.5744681, 0.5833333, 0.59183675],
               [0.6, 0.60784316, 0.61538464, 0.6226415, 0.6296296],
               [0.6363636, 0.64285713, 0.64912283, 0.6551724, 0.66101694]],
              [[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1.]]
          ]),
      dict(
          testcase_name="min_multiply_mean_divide",
          multiply_operation=jnp.min,
          divide_operation=jnp.mean,
          expected_transformed=[
              [[0., 0.04761905, 0.18181819, 0.39130434, 0.6666667],
               [1., 1.3846154, 1.8148148, 2.2857144, 2.7931035],
               [3.3333333, 3.903226, 4.5, 5.121212, 5.7647057],
               [6.428571, 7.111111, 7.810811, 8.526316, 9.256411]],
              [[0., 1., 2., 3., 4.], [5., 6., 7., 8., 9.],
               [10., 11., 12., 13., 14.], [15., 16., 17., 18., 19.]],
              [[0., 1.9523809, 3.8181818, 5.6086955, 7.3333335],
               [9., 10.615385, 12.185185, 13.714286, 15.206897],
               [16.666666, 18.096775, 19.5, 20.878788, 22.235294],
               [23.571428, 24.88889, 26.18919, 27.473684, 28.74359]]
          ]),
  ])
  def test_fit_transform_works_with_operations_in_three_dimensions(
      self, multiply_operation, divide_operation, expected_transformed):
    data = jnp.arange(60).reshape((3, 4, 5))

    scaler = preprocessing.CustomScaler(
        multiply_operation=multiply_operation,
        divide_operation=divide_operation)
    transformed_data = scaler.fit_transform(data)

    np.testing.assert_array_almost_equal(transformed_data,
                                         jnp.array(expected_transformed))

  @parameterized.named_parameters([
      dict(
          testcase_name="four_dims",
          number_of_dimensions=4,
          multiply_by=5,
          divide_by=2,
          expected_transformed=[[[[0., 2.5], [5, 7.5]], [[10, 12.5], [15,
                                                                      17.5]]],
                                [[[20, 22.5], [25, 27.5]],
                                 [[30, 32.5], [35, 37.5]]]]),
      dict(
          testcase_name="five_dims",
          number_of_dimensions=5,
          multiply_by=3,
          divide_by=2,
          expected_transformed=[[[[[0, 1.5], [3, 4.5]], [[6, 7.5], [9, 10.5]]],
                                 [[[12, 13.5], [15, 16.5]],
                                  [[18, 19.5], [21, 22.5]]]],
                                [[[[24, 25.5], [27, 28.5]],
                                  [[30, 31.5], [33, 34.5]]],
                                 [[[36, 37.5], [39, 40.5]],
                                  [[42, 43.5], [45, 46.5]]]]]),
  ])
  def test_fit_transform_produces_correct_values_in_higher_dimensions(
      self, number_of_dimensions, multiply_by, divide_by, expected_transformed):
    data = jnp.arange(2**number_of_dimensions).reshape([2] *
                                                       number_of_dimensions)

    scaler = preprocessing.CustomScaler(
        divide_by=jnp.array(divide_by), multiply_by=jnp.array(multiply_by))
    transformed_data = scaler.fit_transform(data)

    np.testing.assert_array_almost_equal(transformed_data,
                                         jnp.array(expected_transformed))

if __name__ == "__main__":
  absltest.main()
