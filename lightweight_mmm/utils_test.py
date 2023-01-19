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

"""Tests for utils."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
import pandas as pd

from lightweight_mmm import lightweight_mmm
from lightweight_mmm import utils

_MEDIA_DATAFRAME = pd.DataFrame(
    data=[["2020-01-01", "geo1", 10, 2, 3, 2, 1, 1, 1],
          ["2020-01-01", "geo2", 40, 6, 2, 2, 1, 0, 1],
          ["2020-01-01", "geo3", 33, 7, 5, 2, 1, 0, 1],
          ["2020-01-08", "geo1", 21, 1, 7, 2, 1, 1, 1],
          ["2020-01-08", "geo2", 27, 3, 9, 2, 1, 1, 1],
          ["2020-01-08", "geo3", 20, 5, 3, 2, 1, 1, 1]],
    columns=[
        "date", "geo", "kpi", "channel1_imp", "channel2_imp",
        "channel1_cost", "channel2_cost", "promo_1", "promo_2"
    ])

_MEDIA_DATAFRAME_UNSORTED = pd.DataFrame(
    data=[["2020-01-01", "geo2", 40, 6, 2, 2, 1, 0, 1],
          ["2020-01-08", "geo1", 21, 1, 7, 2, 1, 1, 1],
          ["2020-01-08", "geo3", 20, 5, 3, 2, 1, 1, 1],
          ["2020-01-01", "geo1", 10, 2, 3, 2, 1, 1, 1],
          ["2020-01-08", "geo2", 27, 3, 9, 2, 1, 1, 1],
          ["2020-01-01", "geo3", 33, 7, 5, 2, 1, 0, 1]],
    columns=[
        "date", "geo", "kpi", "channel1_imp", "channel2_imp",
        "channel1_cost", "channel2_cost", "promo_1", "promo_2"
    ])


class UtilsTest(parameterized.TestCase):

  def test_save_model_file_is_correctly_saved(self):
    media = jnp.ones((20, 2), dtype=jnp.float32)
    extra_features = jnp.arange(20).reshape((20, 1))
    costs = jnp.arange(1, 3)
    target = jnp.arange(1, 21)
    mmm_object = lightweight_mmm.LightweightMMM()
    mmm_object.fit(
        media=media,
        extra_features=extra_features,
        media_prior=costs,
        target=target,
        number_warmup=10,
        number_samples=100,
        number_chains=1)

    file_path = os.path.join(self.create_tempdir().full_path, "model.pkl")
    utils.save_model(media_mix_model=mmm_object,
                     file_path=file_path)

    self.assertTrue(os.path.exists(file_path))

  def test_load_model_with_all_attributes(self):
    media = jnp.ones((20, 2), dtype=jnp.float32)
    extra_features = jnp.arange(20).reshape((20, 1))
    costs = jnp.arange(1, 3)
    target = jnp.arange(1, 21)
    mmm_object = lightweight_mmm.LightweightMMM()
    mmm_object.fit(
        media=media,
        extra_features=extra_features,
        media_prior=costs,
        target=target,
        number_warmup=10,
        number_samples=100,
        number_chains=1)
    file_path = os.path.join(self.create_tempdir().full_path, "model.pkl")
    utils.save_model(media_mix_model=mmm_object,
                     file_path=file_path)

    loaded_mmm = utils.load_model(file_path)

    self.assertEqual(mmm_object, loaded_mmm)

  @parameterized.named_parameters([
      dict(
          testcase_name="shape_100_3_3",
          data_size=100,
          n_media_channels=3,
          n_extra_features=3),
      dict(
          testcase_name="shape_200_8_1",
          data_size=200,
          n_media_channels=8,
          n_extra_features=1),
      dict(
          testcase_name="shape_300_2_2",
          data_size=300,
          n_media_channels=2,
          n_extra_features=2),
      dict(
          testcase_name="shape_400_4_10",
          data_size=400,
          n_media_channels=4,
          n_extra_features=10)
  ])
  def test_simulate_dummy_data_produces_correct_shape(self,
                                                      data_size,
                                                      n_media_channels,
                                                      n_extra_features):
    media_data, extra_features, target, costs = utils.simulate_dummy_data(
        data_size=data_size,
        n_media_channels=n_media_channels,
        n_extra_features=n_extra_features)

    self.assertEqual(media_data.shape, (data_size, n_media_channels))
    self.assertEqual(extra_features.shape, (data_size, n_extra_features))
    self.assertEqual(target.shape, (data_size,))
    self.assertLen(costs, n_media_channels)

  @parameterized.named_parameters([
      dict(
          testcase_name="one_geo_in_geo_feature",
          dataframe=_MEDIA_DATAFRAME[_MEDIA_DATAFRAME["geo"] == "geo1"]),
      dict(
          testcase_name="unequal_weeks_in_each_geo",
          dataframe=_MEDIA_DATAFRAME.iloc[0:5, :])
  ])
  def test_dataframe_to_jax_wrong_params_raises_valueerror(self, dataframe):
    with self.assertRaises(ValueError):
      utils.dataframe_to_jax(
          dataframe=dataframe,
          media_features=["channel1_imp", "channel2_imp"],
          extra_features=["promo_1"],
          geo_feature="geo",
          date_feature="date",
          target="kpi",
          cost_features=None)

  @parameterized.named_parameters([
      dict(
          testcase_name="array_shape_without_cost_feature_regional_model",
          cost_features=None),
      dict(
          testcase_name="array_shape_with_cost_feature_regional_model",
          cost_features=["channel1_cost", "channel2_cost"])
  ])
  def test_dataframe_to_jax_produce_correct_shape_with_or_without_cost_regional(
      self, cost_features):
    n_weeks = _MEDIA_DATAFRAME["date"].nunique()
    n_geos = _MEDIA_DATAFRAME["geo"].nunique()
    n_media_channels = len(["channel1_imp", "channel2_imp"])
    n_extra_features = len(["promo_1", "promo_2"])

    media_data, extra_features_data, target_data, costs_data = utils.dataframe_to_jax(
        dataframe=_MEDIA_DATAFRAME,
        media_features=["channel1_imp", "channel2_imp"],
        extra_features=["promo_1", "promo_2"],
        geo_feature="geo",
        date_feature="date",
        target="kpi",
        cost_features=cost_features
        )

    self.assertEqual(media_data.shape, (n_weeks, n_media_channels, n_geos))
    self.assertEqual(extra_features_data.shape,
                     (n_weeks, n_extra_features, n_geos))
    self.assertEqual(target_data.shape, (n_weeks, n_geos))
    self.assertLen(costs_data, n_media_channels)

  @parameterized.named_parameters([
      dict(
          testcase_name="array_shape_without_cost_feature_national_model",
          cost_features=None),
      dict(
          testcase_name="array_shape_with_cost_feature_national_model",
          cost_features=["channel1_cost", "channel2_cost"])
  ])
  def test_dataframe_to_jax_produce_correct_shape_with_or_without_cost_national_multiple_features(
      self, cost_features):
    n_weeks = _MEDIA_DATAFRAME.loc[
        _MEDIA_DATAFRAME["geo"] == "geo1", "date"].nunique()
    n_media_channels = len(["channel1_imp", "channel2_imp"])
    n_extra_features = len(["promo_1", "promo_2"])

    media_data, extra_features_data, target_data, costs_data = utils.dataframe_to_jax(
        dataframe=_MEDIA_DATAFRAME[_MEDIA_DATAFRAME["geo"] == "geo1"],
        media_features=["channel1_imp", "channel2_imp"],
        extra_features=["promo_1", "promo_2"],
        date_feature="date",
        target="kpi",
        cost_features=cost_features
    )

    self.assertEqual(media_data.shape, (n_weeks, n_media_channels))
    self.assertEqual(extra_features_data.shape,
                     (n_weeks, n_extra_features))
    self.assertLen(target_data, n_weeks)
    self.assertLen(costs_data, n_media_channels)

  @parameterized.named_parameters([
      dict(
          testcase_name="array_shape_without_cost_feature_national_model",
          cost_features=None),
      dict(
          testcase_name="array_shape_with_cost_feature_national_model",
          cost_features=["channel1_cost"])
  ])
  def test_dataframe_to_jax_produce_correct_shape_with_or_without_cost_national_one_feature(
      self, cost_features):
    n_weeks = _MEDIA_DATAFRAME.loc[
        _MEDIA_DATAFRAME["geo"] == "geo1", "date"].nunique()
    n_media_channels = len(["channel1_imp"])
    n_extra_features = len(["promo_1"])

    media_data, extra_features_data, target_data, costs_data = utils.dataframe_to_jax(
        dataframe=_MEDIA_DATAFRAME[_MEDIA_DATAFRAME["geo"] == "geo1"],
        media_features=["channel1_imp"],
        extra_features=["promo_1"],
        date_feature="date",
        target="kpi",
        cost_features=cost_features
    )

    self.assertEqual(media_data.shape, (n_weeks, n_media_channels))
    self.assertEqual(extra_features_data.shape,
                     (n_weeks, n_extra_features))
    self.assertLen(target_data, n_weeks)
    self.assertLen(costs_data, n_media_channels)

  @parameterized.named_parameters([
      dict(
          testcase_name="array_value_without_cost_feature_regional_model",
          cost_features=None,
          actual_cost_features_list=["channel1_imp", "channel2_imp"]),
      dict(
          testcase_name="array_value_with_cost_feature_regional_model",
          cost_features=["channel1_cost", "channel2_cost"],
          actual_cost_features_list=["channel1_cost", "channel2_cost"])
  ])
  def test_dataframe_to_jax_produce_correct_value_with_or_without_cost_regional(
      self, cost_features, actual_cost_features_list):
    n_weeks = _MEDIA_DATAFRAME["date"].nunique()
    geo1_media = _MEDIA_DATAFRAME.loc[_MEDIA_DATAFRAME["geo"] == "geo1",
                                      ["channel1_imp", "channel2_imp"]].values
    geo1_extra_features = _MEDIA_DATAFRAME.loc[_MEDIA_DATAFRAME["geo"] ==
                                               "geo1", ["promo_1"]].values
    geo1_target = _MEDIA_DATAFRAME.loc[_MEDIA_DATAFRAME["geo"] ==
                                       "geo1", ["kpi"]].values
    geo1_target = geo1_target.reshape(n_weeks,)
    cost_by_channel = _MEDIA_DATAFRAME.loc[
        :, actual_cost_features_list].sum().values

    media_data, extra_features_data, target_data, costs_data = utils.dataframe_to_jax(
        dataframe=_MEDIA_DATAFRAME,
        media_features=["channel1_imp", "channel2_imp"],
        extra_features=["promo_1"],
        geo_feature="geo",
        date_feature="date",
        target="kpi",
        cost_features=cost_features)

    np.testing.assert_array_equal(media_data[:, :, 0], geo1_media)
    np.testing.assert_array_equal(extra_features_data[:, :, 0],
                                  geo1_extra_features)
    np.testing.assert_array_equal(target_data[:, 0],
                                  geo1_target.reshape(n_weeks,))
    np.testing.assert_array_equal(costs_data, cost_by_channel)

  @parameterized.named_parameters([
      dict(
          testcase_name="array_value_without_cost_feature_national_model",
          dataframe=_MEDIA_DATAFRAME[_MEDIA_DATAFRAME["geo"] == "geo1"],
          cost_features=None,
          actual_cost_features_list=["channel1_imp", "channel2_imp"]),
      dict(
          testcase_name="array_value_with_cost_feature_national_model",
          dataframe=_MEDIA_DATAFRAME[_MEDIA_DATAFRAME["geo"] == "geo1"],
          cost_features=["channel1_cost", "channel2_cost"],
          actual_cost_features_list=["channel1_cost", "channel2_cost"])
  ])
  def test_dataframe_to_jax_produce_correct_value_with_or_without_cost_national(
      self, dataframe, cost_features, actual_cost_features_list):
    media_from_test_dataframe = dataframe[
        ["channel1_imp", "channel2_imp"]].values
    extra_features_from_test_dataframe = dataframe[["promo_1"]].values
    target_from_test_dataframe = dataframe["kpi"].values
    cost_by_channel = dataframe.loc[
        :, actual_cost_features_list].sum().values

    media_data, extra_features_data, target_data, costs_data = utils.dataframe_to_jax(
        dataframe=dataframe,
        media_features=["channel1_imp", "channel2_imp"],
        extra_features=["promo_1"],
        date_feature="date",
        target="kpi",
        cost_features=cost_features)

    np.testing.assert_array_equal(media_data, media_from_test_dataframe)
    np.testing.assert_array_equal(
        extra_features_data,
        extra_features_from_test_dataframe)
    np.testing.assert_array_equal(target_data, target_from_test_dataframe)
    np.testing.assert_array_equal(costs_data, cost_by_channel)

  @parameterized.named_parameters([
      dict(
          testcase_name="array_value_without_cost_feature_regional_model_unsorted",
          cost_features=None),
      dict(
          testcase_name="array_value_with_cost_feature_regional_model_unsorted",
          cost_features=["channel1_cost", "channel2_cost"])
  ])
  def test_dataframe_to_jax_produce_correct_value_with_unsorted_dataframe(
      self, cost_features):
    media_data_sorted, extra_features_data_sorted, target_data_sorted, costs_data_sorted = utils.dataframe_to_jax(
        dataframe=_MEDIA_DATAFRAME,
        media_features=["channel1_imp", "channel2_imp"],
        extra_features=["promo_1"],
        geo_feature="geo",
        date_feature="date",
        target="kpi",
        cost_features=cost_features)

    media_data_unsorted, extra_features_data_unsorted, target_data_unsorted, costs_data_unsorted = utils.dataframe_to_jax(
        dataframe=_MEDIA_DATAFRAME_UNSORTED,
        media_features=["channel1_imp", "channel2_imp"],
        extra_features=["promo_1"],
        geo_feature="geo",
        date_feature="date",
        target="kpi",
        cost_features=cost_features)

    np.testing.assert_array_equal(media_data_sorted, media_data_unsorted)
    np.testing.assert_array_equal(
        extra_features_data_sorted, extra_features_data_unsorted)
    np.testing.assert_array_equal(target_data_sorted, target_data_unsorted)
    np.testing.assert_array_equal(costs_data_sorted, costs_data_unsorted)

  @parameterized.named_parameters([
      dict(
          testcase_name="shape_0_3_3",
          data_size=0,
          n_media_channels=3,
          n_extra_features=3),
      dict(
          testcase_name="shape_200_-1_1",
          data_size=200,
          n_media_channels=-1,
          n_extra_features=1),
      dict(
          testcase_name="shape_300_2_-2",
          data_size=300,
          n_media_channels=2,
          n_extra_features=-2),
      dict(
          testcase_name="shape_-400_-4_-10",
          data_size=-400,
          n_media_channels=-4,
          n_extra_features=-10)
  ])
  def test_simulate_dummy_data_with_zero_or_neg_parameter_raises_value_error(
      self, data_size, n_media_channels, n_extra_features):

    with self.assertRaises(ValueError):
      utils.simulate_dummy_data(
          data_size=data_size,
          n_media_channels=n_media_channels,
          n_extra_features=n_extra_features)

  def test_simulate_geo_data_has_right_shape(self):
    data_size = 100
    geos = 3
    media_data, _, target, _ = utils.simulate_dummy_data(
        data_size, 2, 2, geos=geos)
    self.assertEqual(target.shape, (data_size, geos))
    self.assertEqual(media_data.shape, (data_size, 2, geos))

  def test_halfnormal_mean_and_scale(self):
    mean = 1.
    scale = utils.get_halfnormal_scale_from_mean(mean)
    new_mean = utils.get_halfnormal_mean_from_scale(scale)
    self.assertEqual(scale, mean * np.sqrt(np.pi) / np.sqrt(2))
    self.assertEqual(mean, new_mean)

  def test_beta_params_match(self):
    a, b = 2., 3.
    # Expected mean is 2 / 5.
    mu = a / (a + b)
    sigma = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    ahat, bhat = utils.get_beta_params_from_mu_sigma(mu, sigma)
    self.assertAlmostEqual(ahat / (ahat + bhat), 2 / 5)

  def test_prior_posterior_distance_discrete(self):
    p = jnp.array([0] * 2 + [1] * 3)
    q = jnp.array([0] * 3 + [1] * 2 + [2] * 1)
    ks = utils.distance_pior_posterior(p, q, method="KS", discrete=True)
    js = utils.distance_pior_posterior(p, q, method="JS", discrete=True)
    hell = utils.distance_pior_posterior(
        p, q, method="Hellinger", discrete=True)
    mindist = utils.distance_pior_posterior(p, q, method="min", discrete=True)
    print(ks, js, hell, mindist)
    self.assertAlmostEqual(ks, 1 / 6)
    self.assertAlmostEqual(js, 0.283, 3)
    self.assertAlmostEqual(hell, 0.325, 3)
    self.assertAlmostEqual(mindist, 0.267, 3)

  def test_prior_posterior_distance_continuous(self):
    p = jnp.array([0] * 2 + [.5] * 3 + [1] * 2)
    q = jnp.array([0] * 2 + [.5] * 4 + [1] * 2 + [1.5] * 3)
    ks = utils.distance_pior_posterior(p, q, method="KS", discrete=False)
    js = utils.distance_pior_posterior(p, q, method="JS", discrete=False)
    hell = utils.distance_pior_posterior(
        p, q, method="Hellinger", discrete=False)
    mindist = utils.distance_pior_posterior(p, q, method="min", discrete=False)
    print(ks, js, hell, mindist)
    self.assertAlmostEqual(ks, 0.2727, 4)
    self.assertAlmostEqual(js, 0.034, 3)
    self.assertAlmostEqual(hell, 0.034, 3)
    self.assertAlmostEqual(mindist, 0.041, 3)

  def test_outlier_interpolation_straight_line(self):
    x = np.arange(10) * 1.
    x[3:5] += 10
    x = jnp.array(x)
    outlier_idx = jnp.array([3, 4])
    new_x = utils.interpolate_outliers(x, outlier_idx)
    self.assertTrue(all(np.equal(new_x[outlier_idx], [3, 4])))


if __name__ == "__main__":
  absltest.main()
