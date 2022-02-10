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

"""Set of utilities for LightweighMMM package."""
import pickle
from typing import Tuple

from jax import random
import jax.numpy as jnp
import numpy as np
from tensorflow.io import gfile

from lightweight_mmm.lightweight_mmm import lightweight_mmm
from lightweight_mmm.lightweight_mmm import media_transforms


def save_model(
    media_mix_model: lightweight_mmm.LightweightMMM,
    file_path: str
    ) -> None:
  """Saves the given model in the given path.

  Args:
    media_mix_model: Model to save on disk.
    file_path: File path where the model should be placed.
  """
  with gfile.GFile(file_path, "wb") as file:
    pickle.dump(obj=media_mix_model, file=file)


def load_model(file_path: str) -> lightweight_mmm.LightweightMMM:
  """Loads a model given a string path.

  Args:
    file_path: Path of the file containing the model.

  Returns:
    The LightweightMMM object that was stored in the given path.
  """
  with gfile.GFile(file_path, "rb") as file:
    media_mix_model = pickle.load(file=file)

  for attr in dir(media_mix_model):
    if attr.startswith("__"):
      continue
    attr_value = getattr(media_mix_model, attr)
    if isinstance(attr_value, np.ndarray):
      setattr(media_mix_model, attr, jnp.array(attr_value))

  return media_mix_model


def simulate_dummy_data(
    data_size: int,
    n_media_channels: int,
    n_extra_features: int,
    seed: int = 0
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Simulates dummy data needed for media mix modelling.

  This function's goal is to be super simple and not have many parameters,
  although it does not generate a fully realistic dataset is only meant to be
  used for demos/tutorial purposes. Uses adstock for lagging but has no
  saturation and no trend.

  The data simulated includes the media data, extra features, a target/KPI and
  costs.

  Args:
    data_size: Number of rows to generate.
    n_media_channels: Number of media channels to generate.
    n_extra_features: Number of extra features to generate.
    seed: Random seed.

  Returns:
    The simulated media, extra features, target and costs.
  """
  if data_size < 1 or n_media_channels < 1 or n_extra_features < 1:
    raise ValueError(
        "Data size, n_media_channels and n_extra_features must be greater than"
        " 0. Please check the values introduced are greater than zero.")
  key = random.PRNGKey(seed)
  sub_keys = random.split(key=key, num=5)
  media_data = random.normal(key=sub_keys[0],
                             shape=(data_size, n_media_channels)) + 10

  extra_features = random.normal(key=sub_keys[1],
                                 shape=(data_size, n_extra_features)) + 10
  costs = media_data.sum(axis=0)

  seasonality = media_transforms.calculate_seasonality(
      number_periods=data_size,
      degrees=2,
      frequency=52,
      gamma_seasonality=1)
  target_noise = random.normal(key=sub_keys[2], shape=(data_size,)) + 2

  media_data_transformed = media_transforms.adstock(media_data)
  beta_media = random.normal(key=sub_keys[3], shape=(n_media_channels,)) + 3
  beta_extra_features = random.normal(key=sub_keys[4],
                                      shape=(n_extra_features,))
  # There is no trend to keep this very simple.
  target = 10 + seasonality + media_data_transformed.dot(
      beta_media) + extra_features.dot(beta_extra_features) + target_noise

  return media_data, extra_features, target, costs
