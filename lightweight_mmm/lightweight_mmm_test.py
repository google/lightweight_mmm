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

"""Tests for lightweight_mmm."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

from lightweight_mmm import lightweight_mmm
from lightweight_mmm import models


class LightweightMmmTest(parameterized.TestCase):

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
          total_costs=costs,
          target=target,
          number_warmup=5,
          number_samples=5,
          number_chains=1)

  def test_daily_data_returns_weekday_parameter(self):
    n = 50
    media = jnp.arange(2 * n).reshape((n, 2)).astype(jnp.float32)
    target = 1 + 1 * (jnp.arange(n) % 7 == 1) + media[:, 1]
    costs = jnp.array([1, 2])
    mmm_object = lightweight_mmm.LightweightMMM()
    mmm_object.fit(media=media, total_costs=costs, target=target,
                   weekday_seasonality=True, number_warmup=5, number_samples=5,
                   number_chains=1)
    self.assertEqual(mmm_object.trace["weekday"].shape, (5, 7))

  def test_predict_fit_sets_correct_attributes(self):
    media = jnp.ones((20, 3), dtype=jnp.float32)
    extra_features = jnp.arange(40).reshape((20, 2))
    costs = jnp.arange(1, 4)
    target = jnp.arange(1, 21)
    expected_attributes = ("n_media_channels", "_total_costs", "trace",
                           "_number_warmup", "_number_samples",
                           "_number_chains", "_target", "_train_media_size",
                           "_degrees_seasonality", "_seasonality_frequency",
                           "_weekday_seasonality", "media", "_extra_features",
                           "_mcmc", "media_names")

    mmm_object = lightweight_mmm.LightweightMMM()
    mmm_object.fit(
        media=media,
        extra_features=extra_features,
        total_costs=costs,
        target=target,
        number_warmup=5,
        number_samples=5,
        number_chains=1)

    for attribute in expected_attributes:
      self.assertTrue(hasattr(mmm_object, attribute))

  # TODO(): Add testing for more scaled/unscaled options.
  def test_get_posterior_metrics_produces_without_scaling_expected_output(self):
    mmm_object = lightweight_mmm.LightweightMMM()
    mmm_object.media = jnp.ones((140, 3))
    mmm_object._total_costs = jnp.array([2., 1., 3.]) * 15
    mmm_object._target = jnp.ones((140, 1)) * 5
    mmm_object.trace = {
        "media_transformed": jnp.ones((500, 140, 3)) * jnp.arange(1, 4),
        "mu": jnp.ones((500, 140)),
        "beta_media": jnp.ones((500, 3)) * 6
    }
    effect, roi = mmm_object.get_posterior_metrics()
    np.testing.assert_array_almost_equal(
        effect.mean(axis=0), jnp.array([1.2, 2.4, 3.6]), decimal=3)
    np.testing.assert_array_almost_equal(
        roi.mean(axis=0), jnp.array([28., 112., 56.]), decimal=3)

  @parameterized.named_parameters([
      dict(
          testcase_name="national",
          media_shape=(10, 5),
          target_shape=(10,),
          total_costs_shape=(5,)),
      dict(
          testcase_name="geo",
          media_shape=(10, 5, 3),
          target_shape=(10, 3),
          total_costs_shape=(5, 1))
  ])
  def test_get_posterior_metrics_produces_correct_shapes(
      self, media_shape, target_shape, total_costs_shape):
    data = jnp.ones(media_shape, dtype=jnp.float32)
    target = jnp.ones(target_shape, dtype=jnp.float32)
    total_costs = jnp.ones(total_costs_shape)
    mmm_object = lightweight_mmm.LightweightMMM()
    mmm_object.fit(
        media=data, target=target, total_costs=total_costs, number_warmup=10,
        number_samples=25, number_chains=1)

    effect, roi = mmm_object.get_posterior_metrics()
    self.assertEqual(effect.shape, (25, *media_shape[1:]))
    self.assertEqual(roi.shape, (25, *media_shape[1:]))

  @parameterized.named_parameters([
      dict(
          testcase_name="national",
          media_shape=(20, 5),
          target_shape=(20,),
          total_costs_shape=(5,)),
      dict(
          testcase_name="geo",
          media_shape=(20, 5, 3),
          target_shape=(20, 3),
          total_costs_shape=(5, 1))
  ])
  def test_predict_produces_output_with_expected_shape_without_gap(
      self, media_shape, target_shape, total_costs_shape):
    media = jnp.ones(media_shape, dtype=jnp.float32)
    extra_features = jnp.ones(media_shape)
    total_costs = jnp.ones(total_costs_shape)
    target = jnp.ones(target_shape)
    mmm_object = lightweight_mmm.LightweightMMM()
    mmm_object.fit(
        media=media,
        extra_features=extra_features,
        total_costs=total_costs,
        target=target,
        number_warmup=5,
        number_samples=25,
        number_chains=1)

    predictions = mmm_object.predict(media=media, extra_features=extra_features)

    self.assertEqual(predictions.shape, (25, *target.shape))

  @parameterized.named_parameters([
      dict(
          testcase_name="national",
          media_shape=(20, 5),
          target_shape=(20,),
          total_costs_shape=(5,)),
      dict(
          testcase_name="geo",
          media_shape=(20, 5, 3),
          target_shape=(20, 3),
          total_costs_shape=(5, 1))
  ])
  def test_predict_produces_output_with_expected_shape_with_gap(
      self, media_shape, target_shape, total_costs_shape):
    media = jnp.ones(media_shape, dtype=jnp.float32)
    extra_features = jnp.ones(media_shape)
    total_costs = jnp.ones(total_costs_shape)
    target = jnp.ones(target_shape)
    mmm_object = lightweight_mmm.LightweightMMM()
    mmm_object.fit(
        media=media,
        extra_features=extra_features,
        total_costs=total_costs,
        target=target,
        number_warmup=5,
        number_samples=25,
        number_chains=1)

    predictions = mmm_object.predict(
        media=media[10:],
        extra_features=extra_features[10:],
        media_gap=media[:10])

    self.assertEqual(predictions.shape, (25, 10, *target.shape[1:]))

  @parameterized.named_parameters([
      dict(
          testcase_name="national",
          media_shape=(10, 5),
          target_shape=(10,),
          total_costs_shape=(5,)),
      dict(
          testcase_name="geo",
          media_shape=(10, 5, 3),
          target_shape=(10, 3),
          total_costs_shape=(5, 1))
  ])
  def test_trace_after_fit_matches_nsample(
      self, media_shape, target_shape, total_costs_shape):
    media = jnp.ones(media_shape, dtype=jnp.float32)
    costs = jnp.ones(total_costs_shape)
    target = jnp.ones(target_shape)
    mmm_object = lightweight_mmm.LightweightMMM()
    mmm_object.fit(
        media=media,
        total_costs=costs,
        target=target,
        number_warmup=10,
        number_samples=100,
        number_chains=1)

    mmm_object.reduce_trace(50)

    self.assertLen(mmm_object.trace["sigma"], 50)

  def test_trace_after_fit_raise_error_with_wrong_nsample(self):
    media = jnp.ones((20, 2), dtype=jnp.float32)
    extra_features = jnp.arange(20).reshape((20, 1))
    costs = jnp.arange(1, 3)
    target = jnp.arange(1, 21)
    mmm_object = lightweight_mmm.LightweightMMM()
    mmm_object.fit(
        media=media,
        extra_features=extra_features,
        total_costs=costs,
        target=target,
        number_warmup=10,
        number_samples=100,
        number_chains=1)
    with self.assertRaises(ValueError):
      mmm_object.reduce_trace(200)

if __name__ == "__main__":
  absltest.main()
