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

"""A simple and lightweight library for Media Mix Modelling.

Simple usage of this class goes as following:

```
mmm = lightweight_mmm.LightweightMMM()
mmm.fit(media=media_data,
        extra_features=extra_features,
        costs=costs,
        target=target,
        number_samples=1000,
        number_chains=2)

# For obtaining media effect and ROI
predictions, media_effect_hat, roi_hat = mmm.get_posterior_metrics()

# For running predictions on unseen data
mmm.predict(media=media_data_test, extra_features=extra_features_test)
```
"""

import dataclasses
import functools
import logging
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import frozendict
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import infer

from lightweight_mmm.lightweight_mmm import models
from lightweight_mmm.lightweight_mmm import preprocessing

_NAMES_TO_MODEL_TRANSFORMS = frozendict.frozendict({
    "hill_adstock": models.transform_hill_adstock,
    "adstock": models.transform_adstock,
    "carryover": models.transform_carryover
})
_MODEL_FUNCTION = models.media_mix_model


class NotFittedModelError(Exception):
  pass


@dataclasses.dataclass(unsafe_hash=True)
class LightweightMMM:
  """Lightweight Media Mix Modelling wrapper for bayesian models.

  The currently available models are the following:
   - hill_adstock
   - adstock
   - carryover

  It also offers the necessary utilities for calculating media effect and media
  ROI based on models' results.

  Attributes:
    trace: Sampling trace of the bayesian model once fitted.
    n_media_channels: Number of media channels the model was trained with.
    model_name: Name of the model.
    media: The media data the model is trained on. Usefull for a variety of
      insights post model fitting.
    media_names: Names of the media channels passed at fitting time.
    seed: Starting seed to use for PRNGKeys.
  """
  model_name: str = "hill_adstock"
  seed: int = 0

  def __post_init__(self):
    if self.model_name not in _NAMES_TO_MODEL_TRANSFORMS:
      raise ValueError("Model name passed not valid. Please use any of the"
                       "following: 'hill_adstock', 'adstock', 'carryover'.")
    self._model_function = _MODEL_FUNCTION
    self._model_transform_function = _NAMES_TO_MODEL_TRANSFORMS[self.model_name]
    self._main_rng = jax.random.PRNGKey(self.seed)
    self._fit_rng, self._predict_rng = jax.random.split(self._main_rng)

  def fit(
      self,
      media: jnp.ndarray,
      total_costs: jnp.ndarray,
      target: jnp.ndarray,
      extra_features: Optional[jnp.ndarray] = None,
      degrees_seasonality: int = 3,
      seasonality_frequency: int = 52,
      media_names: Optional[Sequence[str]] = None,
      number_warmup: int = 1000,
      number_samples: int = 1000,
      number_chains: int = 2,
      target_accept_prob: float = .85,
      init_strategy: Callable[[Mapping[Any, Any], Any],
                              jnp.ndarray] = numpyro.infer.init_to_median
      ) -> None:
    """Fits MMM given the media data, extra features, costs and sales/KPI.

    For detailed information on the selected model please refer to its
    respective function in the models.py file.

    Args:
      media: Media input data.
      total_costs: Costs of each media channel. The number of cost values must
        be equal to the number of media channels.
      target: Target KPI to use, like for example sales.
      extra_features: Other variables to add to the model.
      degrees_seasonality: Number of degrees to use for seasonality. Default is
        3.
      seasonality_frequency: Frequency of the time period used. Default is 52 as
        in 52 weeks per year.
      media_names: Names of the media channels passed.
      number_warmup: Number of warm up samples. Default is 1000.
      number_samples: Number of samples during sampling. Default is 1000.
      number_chains: Number of chains to sample. Default is 2.
      target_accept_prob: Target acceptance probability for step size in the
        NUTS sampler. Default is .85.
      init_strategy: Initialization function for numpyro NUTS. The available
        options can be found in
        https://num.pyro.ai/en/stable/utilities.html#initialization-strategies.
        Default is numpyro.infer.init_to_median.
    """
    if media.shape[1] != len(total_costs):
      raise ValueError("The number of data channels provided must match the "
                       "number of cost values.")
    if media.min() < 0:
      raise ValueError("Media values must be greater or equal to zero.")
    # TODO(): only positive based media_size denominator for cost prior.
    train_media_size = media.shape[0]
    kernel = numpyro.infer.NUTS(
        model=self._model_function,
        target_accept_prob=target_accept_prob,
        init_strategy=init_strategy)

    if extra_features is not None:
      extra_features = jnp.array(extra_features)

    mcmc = numpyro.infer.MCMC(
        sampler=kernel,
        num_warmup=number_warmup,
        num_samples=number_samples,
        num_chains=number_chains)
    rng_key = jax.random.PRNGKey(0)
    mcmc.run(
        rng_key,
        media_data=jnp.array(media),
        extra_features=extra_features,
        target_data=jnp.array(target),
        cost_prior=jnp.array(total_costs),
        degrees_seasonality=degrees_seasonality,
        frequency=seasonality_frequency,
        transform_function=self._model_transform_function)

    if media_names is not None:
      self.media_names = media_names
    else:
      self.media_names = [f"channel_{i}" for i in range(media.shape[1])]
    self.n_media_channels = media.shape[1]
    self._total_costs = total_costs
    self.trace = mcmc.get_samples()
    self._number_warmup = number_warmup
    self._number_samples = number_samples
    self._number_chains = number_chains
    self._target = target
    self._train_media_size = train_media_size
    self._degrees_seasonality = degrees_seasonality
    self._seasonality_frequency = seasonality_frequency
    self.media = media
    self._extra_features = extra_features
    self._mcmc = mcmc
    logging.info("Model has been fitted")

  def print_summary(self) -> None:
    """Calls print_summary function from numpyro to print parameters summary.
    """
    # TODO(): add name selection for print.
    self._mcmc.print_summary()

  @functools.partial(
      jax.jit,
      static_argnums=(0,),
      static_argnames=("degrees_seasonality", "transform_function", "model"))
  def _predict(self, rng_key: jnp.ndarray, media_data: jnp.ndarray,
               extra_features: Optional[jnp.ndarray], cost_prior: jnp.ndarray,
               degrees_seasonality: int, frequency: int,
               transform_function: Callable[[Any], jnp.ndarray],
               model: Callable[[Any], None],
               posterior_samples: Dict[str, jnp.ndarray]
               ) -> Dict[str, jnp.ndarray]:
    """Encapsulates the numpyro.infer.Predictive function for predict method.

    It serves as a helper jitted function for running predictions.

    Args:
      rng_key: A jax.random.PRNGKey.
      media_data: Media array for needed for the model to run predictions.
      extra_features: Extra features for needed for the model to run.
      cost_prior: Cost prior used for training the model.
      degrees_seasonality: Number of degrees for the seasonality.
      frequency: Frequency of the seasonality.
      transform_function: Media transform function to use within the model.
      model: Numpyro model to use for numpyro.infer.Predictive.
      posterior_samples: Mapping of the posterior samples.

    Returns:
      The predictions for the given data.
    """
    return infer.Predictive(
        model=model, posterior_samples=posterior_samples)(
            rng_key=rng_key,
            media_data=media_data,
            extra_features=extra_features,
            cost_prior=cost_prior,
            target_data=None,
            degrees_seasonality=degrees_seasonality,
            frequency=frequency,
            transform_function=transform_function)

  def predict(
      self,
      media: jnp.ndarray,
      extra_features: Optional[jnp.ndarray] = None,
      media_gap: Optional[jnp.ndarray] = None,
      target_scaler: Optional[preprocessing.CustomScaler] = None
  ) -> jnp.ndarray:
    """Runs the model to obtain predictions for the given input data.

    Predictions returned are distributions, if point estimates are desired one
    can calculate those based on the given distribution.

    Args:
      media: Media array for needed for the model to run predictions.
      extra_features: Extra features for needed for the model to run.
      media_gap: Media data gap between the end of training data and the start
        of the out of sample media given. Eg. if 100 weeks of data were used for
        training and prediction starts 2 months after training data finished we
        need to provide the 8 weeks missing between the training data and the
        prediction data so data transformations (adstock, carryover, ...) can
        take place correctly.
      target_scaler: Scaler that was used to scale the target before training.

    Returns:
      Predictions for the given media and extra features at a given date index.

    Raises:
      NotFittedModelError: When the model has not been fitted before running
        predict.
    """
    if not hasattr(self, "trace"):
      raise NotFittedModelError("Need to fit the model before running "
                                "predictions.")
    if media_gap is not None:
      if media.ndim != media_gap.ndim:
        raise ValueError("Original media data and media gap must have the same "
                         "number of dimensions.")
      if media.ndim > 1 and media.shape[1] != media_gap.shape[1]:
        raise ValueError("Media gap must have the same numer of media channels"
                         "as the original media data.")
      previous_media = jnp.concatenate(arrays=[self.media, media_gap], axis=0)
      if extra_features is not None:
        previous_extra_features = jnp.concatenate(
            arrays=[
                self._extra_features,
                jnp.zeros((media_gap.shape[0], extra_features.shape[1]))
            ],
            axis=0)
    else:
      previous_media = self.media
      previous_extra_features = self._extra_features

    full_media = jnp.concatenate(arrays=[previous_media, media], axis=0)
    if extra_features is not None:
      full_extra_features = jnp.concatenate(
          arrays=[previous_extra_features, extra_features], axis=0)
    else:
      full_extra_features = None
    prediction = self._predict(
        rng_key=self._predict_rng,
        media_data=full_media,
        extra_features=full_extra_features,
        cost_prior=jnp.array(self._total_costs),
        degrees_seasonality=self._degrees_seasonality,
        frequency=self._seasonality_frequency,
        transform_function=self._model_transform_function,
        model=self._model_function,
        posterior_samples=self.trace)["mu"][:, previous_media.shape[0]:]
    if target_scaler:
      prediction = target_scaler.inverse_transform(prediction)

    return prediction

  def reduce_trace(self, nsample: int = 100, seed: int = 0) -> None:
    """Reduces the samples in `trace` to speed up `predict` and optimize.

    Please note this step is not reversible. Only do this after you have
    investigated convergence of the model.

    Args:
      nsample: Target number of samples.
      seed: Random seed for down sampling.

    Raises:
      ValueError: if `nsample` is too big.
    """
    ntrace = len(self.trace["sigma"])
    if ntrace < nsample:
      raise ValueError("nsample is bigger than the actual posterior samples")
    key = jax.random.PRNGKey(seed)
    samples = jax.random.choice(key, ntrace, (nsample,), replace=False)
    for name in self.trace.keys():
      self.trace[name] = self.trace[name][samples]
    logging.info("Reduction is complete")

  def get_posterior_metrics(
      self,
      unscaled_costs: Optional[np.ndarray] = None,
      cost_scaler: Optional[preprocessing.CustomScaler] = None,
      target_scaler: Optional[preprocessing.CustomScaler] = None
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """It estimates the media effect and ROI of each channel.

    If data was scaled prior to training then the target and costs scalers need
    to be passed to this function to correctly calculate effect and ROI in the
    unscaled space.

    Args:
      unscaled_costs: Optionally you can pass new costs to get these set of
        metrics. If None, the costs used for training will be used for
        calculating ROI.
      cost_scaler: Scaler that was used to scale the cost data before training.
        It is ignored if 'unscaled_costs' is provided.
      target_scaler: Scaler that was used to scale the target before training.

    Returns:
      predictions: The mean of the posterior distribution for the taget's mean
        parameter. If target scaler is provided predictions are unscaled.
      media_effect_hat: The average media effect for each channel.
      roi_hat: The return on investment of each channel calculated as its effect
        divided by the cost.

    Raises:
      NotFittedModelError: When the this method is called without the model
        being trained previously.
    """
    if not hasattr(self, "trace"):
      raise NotFittedModelError(
          "LightweightMMM has not been fitted and cannot run estimations. "
          "Please first fit the model.")
    if unscaled_costs is None and not cost_scaler:
      logging.warning("Unscaled cost data or cost scaler were not given and  "
                      "therefore unscaling wont be applied to calculcate effect"
                      " and ROI. If data was not scaled prior to training "
                      "please ignore this warning.")
    if not target_scaler:
      logging.warning("Target scaler was not given and unscaling of the target "
                      "will not occur. If your target was not scaled prior to "
                      "training you can ignore this warning.")
    if unscaled_costs is None:
      if cost_scaler:
        unscaled_costs = cost_scaler.inverse_transform(self._total_costs)
      else:
        unscaled_costs = self._total_costs

    predictions = self.trace["mu"].mean(axis=1)
    if target_scaler:
      predictions = target_scaler.inverse_transform(predictions)
    # s for samples, d for data (time dim), c for channels (media channels)
    effect = jnp.einsum("sdc, sc -> sc", self.trace["media_transformed"],
                        self.trace["beta_media"])
    percent_change = effect / self._target.sum()

    if target_scaler:
      unscaled_target_sum = target_scaler.inverse_transform(self._target).sum()
    else:
      unscaled_target_sum = self._target.sum()

    roi_hat = unscaled_target_sum * percent_change / unscaled_costs
    return predictions, percent_change, roi_hat
