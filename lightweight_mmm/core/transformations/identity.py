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

"""Module for identity transformations."""

from typing import Any
import jax.numpy as jnp


def identity_transform(
    data: jnp.ndarray,  # pylint-ignore: unused-argument
    *args: Any,
    **kwargs: Any,
) -> jnp.ndarray:
  """Identity transform. Returns the main input as is."""
  return data
