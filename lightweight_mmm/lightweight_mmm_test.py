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

"""Tests for lightweight_mmm."""

import copy

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from lightweight_mmm import lightweight_mmm
from lightweight_mmm import models


class LightweightMmmTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(LightweightMmmTest, cls).setUpClass()
    cls.national_mmm = lightweight_mmm.LightweightMMM()
    cls.national_mmm.fit(
        media=jnp.ones((50, 5)),
        target=jnp.ones(50),
        media_prior=jnp.ones(5) * 50,
        extra_features=jnp.ones((50, 2)),
        number_warmup=2,
        number_samples=4,
        number_chains=1)
    cls.geo_mmm = lightweight_mmm.LightweightMMM()
    cls.geo_mmm.fit(
        media=jnp.ones((50, 5, 3)),
        target=jnp.ones((50, 3)),
        media_prior=jnp.ones(5) * 50,
        extra_features=jnp.ones((50, 2, 3)),
        number_warmup=2,
        number_samples=4,
        number_chains=1)

  @parameterized.named_parameters([
      dict(
          testcase_name="hill_adstock",
          model_name="hill_adstock",
          expected_trans_func=models.transform_hill_adstock),
      dict(
          testcase_name="adstock",
          model_name="adstock",
          expected_trans_func=models.transform_adstock),
      dict(
          testcase_name="carryover",
          model_name="carryover",
          expected_trans_func=models.transform_carryover),
  ])
  def test_instantiate_model_correctly_from_available_models(
      self, model_name, expected_trans_func):
    mmm_object = lightweight_mmm.LightweightMMM(model_name=model_name)

    self.assertEqual(mmm_object._model_transform_function, expected_trans_func)

  def test_instantiate_model_wrong_raises_valueerror(self):
    with self.assertRaises(ValueError):
      lightweight_mmm.LightweightMMM(model_name="non_existing_model")

  @parameterized.named_parameters([
      dict(
          testcase_name="negative_values_national",
          media=-np.ones((20, 3)),
          target_shape=(20,),
          total_costs_shape=(3,)),
      dict(
          testcase_name="negative_values_geo",
          media=-np.ones((20, 3, 3)),
          target_shape=(20, 3),
          total_costs_shape=(3, 1)),
      dict(
          testcase_name="wrong_media_shape",
          media=-np.ones((20, 2)),
          target_shape=(20,),
          total_costs_shape=(3,)),
      dict(
          testcase_name="extra_dims",
          media=np.ones((20, 2, 4, 5)),
          target_shape=(20,),
          total_costs_shape=(3,))
  ])
  def test_fit_wrong_inputs_raises_value_error(
      self, media, target_shape, total_costs_shape):
    media = jnp.array(media)
    extra_features = jnp.ones((20, 3))
    costs = jnp.ones(total_costs_shape)
    target = jnp.ones(target_shape)
    mmm_object = lightweight_mmm.LightweightMMM()

    with self.assertRaises(ValueError):
      mmm_object.fit(
          media=media,
          extra_features=extra_features,
          media_prior=costs,
          target=target,
          number_warmup=5,
          number_samples=5,
          number_chains=1)

  @parameterized.named_parameters([
      dict(
          testcase_name="hill_adstock",
          model_name="hill_adstock",
          custom_priors={models._EXPONENT: 3.}),
      dict(
          testcase_name="carryover",
          model_name="carryover",
          custom_priors={models._HALF_MAX_EFFECTIVE_CONCENTRATION: 6.}),
      dict(
          testcase_name="adstock",
          model_name="adstock",
          custom_priors={models._AD_EFFECT_RETENTION_RATE: 5.})
  ])
  def test_fit_with_custom_prior_raises_valueerror_not_used_priors(
      self, model_name, custom_priors):
    media = jnp.ones((20, 3))
    target = jnp.ones((20,))
    extra_features = jnp.ones((20, 3))
    costs = jnp.ones(3)

    mmm_object = lightweight_mmm.LightweightMMM(model_name=model_name)

    with self.assertRaisesRegex(
        ValueError,
        "The following passed custom priors dont have a match in the model."):
      mmm_object.fit(
          media=media,
          target=target,
          extra_features=extra_features,
          media_prior=costs,
          custom_priors=custom_priors)

  def test_fit_with_geo_custom_prior_raises_valueerror_if_national_data(self):
    media = jnp.ones((20, 3))
    target = jnp.ones((20,))
    extra_features = jnp.ones((20, 3))
    costs = jnp.ones(3)

    mmm_object = lightweight_mmm.LightweightMMM()

    with self.assertRaisesRegex(
        ValueError,
        "The given data is for national models but custom_prior contains "):
      mmm_object.fit(
          media=media,
          target=target,
          extra_features=extra_features,
          media_prior=costs,
          custom_priors={models._COEF_SEASONALITY: dist.HalfNormal(3)})

  @parameterized.named_parameters([
      dict(
          testcase_name="too_many_args",
          custom_priors={models._INTERCEPT: (3., 4.)}),
      dict(
          testcase_name="dict_wrong_keys",
          custom_priors={models._INTERCEPT: {"foo": 6.}})
  ])
  def test_fit_with_custom_prior_raises_numpyro_error_if_wrong_args(
      self, custom_priors):
    media = jnp.ones((20, 3))
    target = jnp.ones((20,))
    extra_features = jnp.ones((20, 3))
    costs = jnp.ones(3)

    mmm_object = lightweight_mmm.LightweightMMM()

    with self.assertRaises(TypeError):
      mmm_object.fit(
          media=media,
          target=target,
          extra_features=extra_features,
          media_prior=costs,
          custom_priors=custom_priors)

  @parameterized.named_parameters([
      dict(
          testcase_name="wrong_type1",
          custom_priors={models._INTERCEPT: "hello"}),
      dict(
          testcase_name="wrong_type2",
          custom_priors={models._INTERCEPT: lightweight_mmm.LightweightMMM()}),
  ])
  def test_fit_with_custom_prior_raises_valueerror_if_wrong_format(
      self, custom_priors):
    media = jnp.ones((20, 3))
    target = jnp.ones((20,))
    extra_features = jnp.ones((20, 3))
    costs = jnp.ones(3)

    mmm_object = lightweight_mmm.LightweightMMM()

    with self.assertRaisesRegex(
        ValueError,
        "Priors given must be a Numpyro distribution or one of the "):
      mmm_object.fit(
          media=media,
          target=target,
          extra_features=extra_features,
          media_prior=costs,
          custom_priors=custom_priors)

  def test_fit_with_custom_priors_uses_correct_given_priors(self):
    pass

  def test_daily_data_returns_weekday_parameter(self):
    n = 50
    media = jnp.arange(2 * n).reshape((n, 2)).astype(jnp.float32)
    target = 1 + 1 * (jnp.arange(n) % 7 == 1) + media[:, 1]
    costs = jnp.array([1, 2])
    mmm_object = lightweight_mmm.LightweightMMM()
    mmm_object.fit(media=media, media_prior=costs, target=target,
                   weekday_seasonality=True, number_warmup=5, number_samples=5,
                   number_chains=1)
    self.assertEqual(mmm_object.trace["weekday"].shape, (5, 7))

  @parameterized.named_parameters([
      dict(testcase_name="trace", attribute_name="trace"),
      dict(testcase_name="n_media_channels", attribute_name="n_media_channels"),
      dict(testcase_name="n_geos", attribute_name="n_geos"),
      dict(testcase_name="_number_warmup", attribute_name="_number_warmup"),
      dict(testcase_name="_number_samples", attribute_name="_number_samples"),
      dict(testcase_name="_number_chains", attribute_name="_number_chains"),
      dict(testcase_name="_target", attribute_name="_target"),
      dict(
          testcase_name="_train_media_size",
          attribute_name="_train_media_size"),
      dict(
          testcase_name="_degrees_seasonality",
          attribute_name="_degrees_seasonality"),
      dict(
          testcase_name="_seasonality_frequency",
          attribute_name="_seasonality_frequency"),
      dict(
          testcase_name="_weekday_seasonality",
          attribute_name="_weekday_seasonality"),
      dict(testcase_name="extra_features", attribute_name="extra_features"),
      dict(testcase_name="media", attribute_name="media"),
      dict(testcase_name="custom_priors", attribute_name="custom_priors"),
  ])
  def test_fitting_attributes_do_not_exist_before_fitting(self, attribute_name):
    mmm_object = lightweight_mmm.LightweightMMM()
    self.assertFalse(hasattr(mmm_object, attribute_name))

  @parameterized.named_parameters([
      dict(
          testcase_name="geo_mmm",
          media_mix_model="geo_mmm"),
      dict(
          testcase_name="national_mmm",
          media_mix_model="national_mmm")
  ])
  def test_predict_fit_sets_correct_attributes(self, media_mix_model):
    expected_attributes = ("n_media_channels", "_media_prior", "trace",
                           "_number_warmup", "_number_samples",
                           "_number_chains", "_target", "_train_media_size",
                           "_degrees_seasonality", "_seasonality_frequency",
                           "_weekday_seasonality", "media", "_extra_features",
                           "_mcmc", "media_names", "custom_priors")

    mmm_object = getattr(self, media_mix_model)

    for attribute in expected_attributes:
      self.assertTrue(hasattr(mmm_object, attribute))

  # TODO(): Add testing for more scaled/unscaled options.
  def test_get_posterior_metrics_produces_without_scaling_expected_output(self):
    mmm_object = lightweight_mmm.LightweightMMM()
    mmm_object.media = jnp.ones((140, 3))
    mmm_object._media_prior = jnp.array([2., 1., 3.]) * 15
    mmm_object._target = jnp.ones((140, 1)) * 5
    mmm_object.trace = {
        "media_transformed": jnp.ones((500, 140, 3)) * jnp.arange(1, 4),
        "mu": jnp.ones((500, 140)),
        "coef_media": jnp.ones((500, 3)) * 6
    }
    contribution, roi = mmm_object.get_posterior_metrics()
    np.testing.assert_array_almost_equal(
        contribution.mean(axis=0), jnp.array([6., 12., 18.]), decimal=3)
    np.testing.assert_array_almost_equal(
        roi.mean(axis=0), jnp.array([28., 112., 56.]), decimal=3)

  @parameterized.named_parameters([
      dict(
          testcase_name="national_mmm",
          media_mix_model="national_mmm"),
      dict(
          testcase_name="geo_mmm",
          media_mix_model="geo_mmm")
  ])
  def test_get_posterior_metrics_produces_correct_shapes(self, media_mix_model):
    mmm_object = getattr(self, media_mix_model)

    contribution, roi = mmm_object.get_posterior_metrics()

    self.assertEqual(contribution.shape, (4, *mmm_object.media.shape[1:]))
    self.assertEqual(roi.shape, (4, *mmm_object.media.shape[1:]))

  @parameterized.named_parameters([
      dict(
          testcase_name="national_mmm",
          media_mix_model="national_mmm"),
      dict(
          testcase_name="geo_mmm",
          media_mix_model="geo_mmm")
  ])
  def test_predict_produces_output_with_expected_shape_without_gap(
      self, media_mix_model):
    mmm_object = getattr(self, media_mix_model)

    predictions = mmm_object.predict(
        media=mmm_object.media, extra_features=mmm_object._extra_features)

    self.assertEqual(predictions.shape, (4, *mmm_object._target.shape))

  @parameterized.named_parameters([
      dict(
          testcase_name="national_mmm",
          media_mix_model="national_mmm"),
      dict(
          testcase_name="geo_mmm",
          media_mix_model="geo_mmm")
  ])
  def test_predict_produces_output_with_expected_shape_with_gap(
      self, media_mix_model):
    mmm_object = getattr(self, media_mix_model)

    predictions = mmm_object.predict(
        media=mmm_object.media[-10:],
        extra_features=mmm_object._extra_features[-10:],
        media_gap=mmm_object.media[-10:])

    self.assertEqual(predictions.shape, (4, 10, *mmm_object._target.shape[1:]))

  @parameterized.named_parameters([
      dict(
          testcase_name="national_mmm",
          media_mix_model="national_mmm"),
      dict(
          testcase_name="geo_mmm",
          media_mix_model="geo_mmm")
  ])
  def test_trace_after_fit_matches_nsample(self, media_mix_model):
    mmm_object = getattr(self, media_mix_model)

    mmm_object.reduce_trace(2)

    self.assertLen(mmm_object.trace["sigma"], 2)

  @parameterized.named_parameters([
      dict(
          testcase_name="national_mmm",
          media_mix_model="national_mmm"),
      dict(
          testcase_name="geo_mmm",
          media_mix_model="geo_mmm")
  ])
  def test_trace_after_fit_raise_error_with_wrong_nsample(
      self, media_mix_model):
    mmm_object = getattr(self, media_mix_model)

    with self.assertRaises(ValueError):
      mmm_object.reduce_trace(200)

  @parameterized.named_parameters([
      dict(
          testcase_name="national_mmm",
          media_mix_model="national_mmm"),
      dict(
          testcase_name="geo_mmm",
          media_mix_model="geo_mmm"),
  ])
  def test_equality_method_lmmm_instance_equals_itself(self, media_mix_model):
    mmm_object = getattr(self, media_mix_model)
    self.assertEqual(mmm_object, mmm_object)

  @parameterized.named_parameters([
      dict(
          testcase_name="national_mmm",
          media_mix_model_1="national_mmm",
          media_mix_model_2="national_mmm"),
      dict(
          testcase_name="geo_mmm",
          media_mix_model_1="geo_mmm",
          media_mix_model_2="geo_mmm"),
  ])
  def test_two_lmmm_instances_equal_each_other(self, media_mix_model_1,
                                               media_mix_model_2):
    mmm_object_1 = getattr(self, media_mix_model_1)
    mmm_object_2 = copy.copy(getattr(self, media_mix_model_2))
    self.assertEqual(mmm_object_1, mmm_object_2)

  @parameterized.named_parameters([
      dict(
          testcase_name="national_mmm",
          media_mix_model_1="national_mmm",
          media_mix_model_2="national_mmm"),
      dict(
          testcase_name="geo_mmm",
          media_mix_model_1="geo_mmm",
          media_mix_model_2="geo_mmm"),
  ])
  def test_two_lmmm_instances_equal_each_other_deepcopy(self, media_mix_model_1,
                                                        media_mix_model_2):
    mmm_object_1 = getattr(self, media_mix_model_1)
    mmm_object_2 = copy.deepcopy(getattr(self, media_mix_model_2))
    self.assertEqual(mmm_object_1, mmm_object_2)

  def test_different_lmmms_are_not_equal(self):
    self.assertNotEqual(self.national_mmm, self.geo_mmm)

  def test_default_mmm_instances_equal_each_other(self):
    self.assertEqual(lightweight_mmm.LightweightMMM(),
                     lightweight_mmm.LightweightMMM())

  @parameterized.named_parameters([
      dict(
          testcase_name="hill_adstock",
          model_name="hill_adstock",
          expected_equal=False),
      dict(
          testcase_name="adstock",
          model_name="adstock",
          expected_equal=False),
      dict(
          testcase_name="carryover",
          model_name="carryover",
          expected_equal=True),
  ])
  def test_default_carryover_mmm_instance_only_equals_carryover_mmms(
      self, model_name, expected_equal):
    carryover_mmm = lightweight_mmm.LightweightMMM(model_name="carryover")
    other_mmm = lightweight_mmm.LightweightMMM(model_name=model_name)

    if expected_equal:
      self.assertEqual(carryover_mmm, other_mmm)
    else:
      self.assertNotEqual(carryover_mmm, other_mmm)

  @parameterized.named_parameters([
      dict(testcase_name="national_mmm", media_mix_model="national_mmm"),
      dict(testcase_name="geo_mmm", media_mix_model="geo_mmm"),
  ])
  def test_fitted_mmm_does_not_equal_default_mmm(self, media_mix_model):
    default_mmm_object = lightweight_mmm.LightweightMMM()
    fitted_mmm_object = getattr(self, media_mix_model)
    self.assertNotEqual(default_mmm_object, fitted_mmm_object)

if __name__ == "__main__":
  absltest.main()
