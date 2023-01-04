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

"""Module for modeling the intercept."""

from typing import Mapping

import immutabledict
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

from lightweight_mmm.core import core_utils
from lightweight_mmm.core import priors


def simple_intercept(
    data: jnp.ndarray,
    custom_priors: Mapping[str,
                           dist.Distribution] = immutabledict.immutabledict(),
) -> jnp.ndarray:
  """Calculates a national or geo incercept.
  Note that this intercept is constant over time.

  Args:
    data: Media input data. Media data must have either 2 dims for national
      model or 3 for geo models.
    custom_priors: The custom priors we want the model to take instead of the
      default ones. Refer to the full documentation on custom priors for
      details.

  Returns:
    The values of the intercept.
  """
  default_priors = priors.get_default_priors()
  n_geos = core_utils.get_number_geos(data=data)

  with numpyro.plate(name=f"{priors.INTERCEPT}_plate", size=n_geos):
    intercept = numpyro.sample(
        name=priors.INTERCEPT,
        fn=custom_priors.get(priors.INTERCEPT,
                             default_priors[priors.INTERCEPT]),
    )
  return intercept
