# Copyright 2021 Google LLC.
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
import jax.numpy as jnp

from lightweight_mmm import lightweight_mmm
from lightweight_mmm import utils


class UtilsTest(absltest.TestCase):

  def test_save_model_file_is_correctly_saved(self):
    media = jnp.ones((20, 2), dtype=jnp.float32)
    extra_features = jnp.arange(20).reshape((20, 1))
    costs = jnp.arange(1, 3)
    target = jnp.arange(1, 21)
    mmm_object = lightweight_mmm.LightweightMMM()
    mmm_object.fit(
        media=media,
        extra_features=extra_features,
        costs=costs,
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
        costs=costs,
        target=target,
        number_warmup=10,
        number_samples=100,
        number_chains=1)
    file_path = os.path.join(self.create_tempdir().full_path, "model.pkl")
    utils.save_model(media_mix_model=mmm_object,
                     file_path=file_path)

    loaded_mmm = utils.load_model(file_path)

    self.assertEqual(mmm_object, loaded_mmm)

if __name__ == "__main__":
  absltest.main()
