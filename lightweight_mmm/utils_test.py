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

"""Tests for utils."""

import os

from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as jnp

from lightweight_mmm.lightweight_mmm import lightweight_mmm
from lightweight_mmm.lightweight_mmm import utils


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
        total_costs=costs,
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
        total_costs=costs,
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


if __name__ == "__main__":
  absltest.main()
