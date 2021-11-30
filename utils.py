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

"""Set of utilities for LightweighMMM package."""

import pickle

import jax.numpy as jnp
import numpy as np
from tensorflow.io import gfile

from lightweight_mmm import lightweight_mmm


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
