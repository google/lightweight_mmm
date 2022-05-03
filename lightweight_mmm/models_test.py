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

"""Tests for models."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpyro

from lightweight_mmm import models


class ModelsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="one_channel", shape=(10, 1)),
      dict(testcase_name="five_channel", shape=(10, 5)),
      dict(testcase_name="same_channels_as_rows", shape=(10, 10)),
      dict(testcase_name="geo_shape_1", shape=(10, 10, 5)),
      dict(testcase_name="geo_shape_2", shape=(10, 5, 2)),
      dict(testcase_name="one_channel_one_row", shape=(1, 1)))
  def test_transform_adstock_produces_correct_output_shape(self, shape):

    def mock_model_function(media_data):
      numpyro.deterministic("transformed_media",
                            models.transform_adstock(media_data))

    media = jnp.ones(shape)
    kernel = numpyro.infer.NUTS(model=mock_model_function)
    mcmc = numpyro.infer.MCMC(
        sampler=kernel, num_warmup=10, num_samples=10, num_chains=1)
    rng_key = jax.random.PRNGKey(0)

    mcmc.run(rng_key, media_data=media)
    transformed_media = mcmc.get_samples()["transformed_media"].mean(axis=0)

    self.assertEqual(media.shape, transformed_media.shape)

  @parameterized.named_parameters(
      dict(testcase_name="one_channel", shape=(10, 1)),
      dict(testcase_name="five_channel", shape=(10, 5)),
      dict(testcase_name="same_channels_as_rows", shape=(10, 10)),
      dict(testcase_name="geo_shape_1", shape=(10, 10, 5)),
      dict(testcase_name="geo_shape_2", shape=(10, 5, 2)),
      dict(testcase_name="one_channel_one_row", shape=(1, 1)))
  def test_transform_hill_adstock_produces_correct_output_shape(self, shape):

    def mock_model_function(media_data):
      numpyro.deterministic(
          "transformed_media",
          models.transform_hill_adstock(media_data))

    media = jnp.ones(shape)
    kernel = numpyro.infer.NUTS(model=mock_model_function)
    mcmc = numpyro.infer.MCMC(
        sampler=kernel, num_warmup=10, num_samples=10, num_chains=1)
    rng_key = jax.random.PRNGKey(0)

    mcmc.run(rng_key, media_data=media)
    transformed_media = mcmc.get_samples()["transformed_media"].mean(axis=0)

    self.assertEqual(media.shape, transformed_media.shape)

  @parameterized.named_parameters(
      dict(testcase_name="one_channel", shape=(10, 1)),
      dict(testcase_name="five_channel", shape=(10, 5)),
      dict(testcase_name="same_channels_as_rows", shape=(10, 10)),
      dict(testcase_name="geo_shape_1", shape=(10, 10, 5)),
      dict(testcase_name="geo_shape_2", shape=(10, 5, 2)),
      dict(testcase_name="one_channel_one_row", shape=(1, 1)))
  def test_transform_carryover_produces_correct_output_shape(self, shape):

    def mock_model_function(media_data):
      numpyro.deterministic(
          "transformed_media",
          models.transform_carryover(media_data))

    media = jnp.ones(shape)
    kernel = numpyro.infer.NUTS(model=mock_model_function)
    mcmc = numpyro.infer.MCMC(
        sampler=kernel, num_warmup=10, num_samples=10, num_chains=1)
    rng_key = jax.random.PRNGKey(0)

    mcmc.run(rng_key, media_data=media)
    transformed_media = mcmc.get_samples()["transformed_media"].mean(axis=0)

    self.assertEqual(media.shape, transformed_media.shape)

  @parameterized.named_parameters(
      dict(
          testcase_name="national_no_extra",
          media_shape=(10, 3),
          extra_features_shape=(),
          target_shape=(10,),
          total_costs_shape=(3,)),
      dict(
          testcase_name="national_extra",
          media_shape=(10, 5),
          extra_features_shape=(10, 2),
          target_shape=(10,),
          total_costs_shape=(5,)),
      dict(
          testcase_name="geo_extra_3d",
          media_shape=(10, 7, 5),
          extra_features_shape=(10, 8, 5),
          target_shape=(10, 5),
          total_costs_shape=(7, 1)),
      dict(
          testcase_name="geo_no_extra",
          media_shape=(10, 7, 5),
          extra_features_shape=(),
          target_shape=(10, 5),
          total_costs_shape=(7, 1)))
  def test_media_mix_model_parameters_have_correct_shapes(
      self, media_shape, extra_features_shape, target_shape, total_costs_shape):
    media = jnp.ones(media_shape)
    extra_features = None if not extra_features_shape else jnp.ones(
        extra_features_shape)
    costs_prior = jnp.ones(total_costs_shape)
    degrees = 2
    target = jnp.ones(target_shape)
    kernel = numpyro.infer.NUTS(model=models.media_mix_model)
    mcmc = numpyro.infer.MCMC(
        sampler=kernel, num_warmup=10, num_samples=10, num_chains=1)
    rng_key = jax.random.PRNGKey(0)

    mcmc.run(
        rng_key,
        media_data=media,
        extra_features=extra_features,
        target_data=target,
        cost_prior=costs_prior,
        degrees_seasonality=degrees,
        frequency=52,
        transform_function=models.transform_carryover)
    trace = mcmc.get_samples()

    self.assertEqual(
        jnp.squeeze(trace["intercept"].mean(axis=0)).shape, target_shape[1:])
    self.assertEqual(
        jnp.squeeze(trace["sigma"].mean(axis=0)).shape, target_shape[1:])
    self.assertEqual(
        jnp.squeeze(trace["expo_trend"].mean(axis=0)).shape, ())
    self.assertEqual(
        jnp.squeeze(trace["beta_trend"].mean(axis=0)).shape, target_shape[1:])
    self.assertEqual(
        jnp.squeeze(trace["beta_media"].mean(axis=0)).shape, media_shape[1:])
    if extra_features_shape:
      self.assertEqual(trace["beta_extra_features"].mean(axis=0).shape,
                       extra_features.shape[1:])
    self.assertEqual(trace["gamma_seasonality"].mean(axis=0).shape,
                     (degrees, 2))
    self.assertEqual(trace["mu"].mean(axis=0).shape, target_shape)

if __name__ == "__main__":
  absltest.main()
