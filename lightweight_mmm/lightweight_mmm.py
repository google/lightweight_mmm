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

"""A simple and lightweight library for Media Mix Modelling.

Simple usage of this class goes as following:

```
mmm = lightweight_mmm.LightweightMMM()
mmm.fit(media=media_data,
        extra_features=extra_features,
        media_prior=costs,
        target=target,
        number_samples=1000,
        number_chains=2)

# For obtaining media contribution percentage and ROI
predictions, media_contribution_hat_pct, roi_hat = mmm.get_posterior_metrics()

# For running predictions on unseen data
mmm.predict(media=media_data_test, extra_features=extra_features_test)
```
"""

import collections
import dataclasses
import functools
import itertools
import logging
import numbers
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

from absl import logging
import immutabledict
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import distributions as dist
from numpyro import infer

from lightweight_mmm import models
from lightweight_mmm import preprocessing
from lightweight_mmm import utils

Prior = Union[
    dist.Distribution,
    Dict[str, float],
    Sequence[float],
    float
]

_NAMES_TO_MODEL_TRANSFORMS = immutabledict.immutabledict({
    "hill_adstock": models.transform_hill_adstock,
    "adstock": models.transform_adstock,
    "carryover": models.transform_carryover
})
_MODEL_FUNCTION = models.media_mix_model


def _compare_equality_for_lmmm(item_1: Any, item_2: Any) -> bool:
  """Compares two items for equality.

  Helper function for the __eq__ method of LightweightmMM. First checks if items
  are strings or lists of strings (it's okay if empty lists compare True), then
  uses jnp.array_equal if the items are jax.numpy.DeviceArray or other related
  sequences, and uses items' __eq__ otherwise.

  Note: this implementation does not cover every possible data structure, but
  it does cover all the data structures seen in attributes used by
  LightweightMMM. Sometimes the DeviceArray is hidden in the value of a
  MutableMapping, hence the recursion.

  Args:
    item_1: First item to be compared.
    item_2: Second item to be compared.

  Returns:
    Boolean for whether item_1 equals item_2.
  """

  # This is pretty strict but LMMM classes don't need to compare equal unless
  # they are exact copies.
  if type(item_1) != type(item_2):
    is_equal = False
  elif isinstance(item_1, str):
    is_equal = item_1 == item_2
  elif isinstance(item_1, (jax.Array, np.ndarray)) or (
      isinstance(item_1, Sequence)
      and not all(isinstance(x, str) for x in item_1)
  ):
    is_equal = np.array_equal(item_1, item_2, equal_nan=True)
  elif isinstance(item_1, MutableMapping):
    is_equal = all(
        [
            _compare_equality_for_lmmm(item_1[x], item_2[x])
            for x in item_1.keys() | item_2.keys()
        ]
    )
  else:
    is_equal = item_1 == item_2

  return is_equal


class NotFittedModelError(Exception):
  pass


@dataclasses.dataclass(unsafe_hash=True, eq=False)
class LightweightMMM:
  """Lightweight Media Mix Modelling wrapper for bayesian models.

  The currently available models are the following:
   - hill_adstock
   - adstock
   - carryover

  It also offers the necessary utilities for calculating media contribution and
  media ROI based on models' results.

  Attributes:
    trace: Sampling trace of the bayesian model once fitted.
    n_media_channels: Number of media channels the model was trained with.
    n_geos: Number of geos for geo models or 1 for national models.
    model_name: Name of the model.
    media: The media data the model is trained on. Usefull for a variety of
      insights post model fitting.
    media_names: Names of the media channels passed at fitting time.
    custom_priors: The set of custom priors the model was trained with. An empty
      dictionary if none were passed.
  """
  model_name: str = "hill_adstock"
  n_media_channels: int = dataclasses.field(init=False, repr=False)
  n_geos: int = dataclasses.field(init=False, repr=False)
  media: jax.Array = dataclasses.field(
      init=False, repr=False, hash=False, compare=True)
  media_names: Sequence[str] = dataclasses.field(
      init=False, repr=False, hash=False, compare=True)
  trace: Dict[str, jax.Array] = dataclasses.field(
      init=False, repr=False, hash=False, compare=False)
  custom_priors: MutableMapping[str, Prior] = dataclasses.field(
      init=False, repr=False, hash=False, compare=True)
  _degrees_seasonality: int = dataclasses.field(init=False, repr=False)
  _weekday_seasonality: bool = dataclasses.field(init=False, repr=False)
  _media_prior: jax.Array = dataclasses.field(
      init=False, repr=False, hash=False, compare=True)
  _extra_features: jax.Array = dataclasses.field(
      init=False, repr=False, hash=False, compare=True)
  _target: jax.Array = dataclasses.field(
      init=False, repr=False, hash=False, compare=True)
  _train_media_size: int = dataclasses.field(
      init=False, repr=False, hash=True, compare=False)
  _mcmc: numpyro.infer.MCMC = dataclasses.field(
      init=False, repr=False, hash=False, compare=False)

  def __post_init__(self):
    if self.model_name not in _NAMES_TO_MODEL_TRANSFORMS:
      raise ValueError("Model name passed not valid. Please use any of the"
                       "following: 'hill_adstock', 'adstock', 'carryover'.")
    self._model_function = _MODEL_FUNCTION
    self._model_transform_function = _NAMES_TO_MODEL_TRANSFORMS[self.model_name]
    self._prior_names = models.MODEL_PRIORS_NAMES.union(
        models.TRANSFORM_PRIORS_NAMES[self.model_name])

  def __eq__(self, other: Any) -> bool:
    """Equality method for LightweightMMMM.

    We need a special method here to handle a couple of issues. First, some of
    the attributes for LightweightMMM are arrays, which contain multiple values
    and cannot be evaluated with the default __eq__ method. Second, some
    attributes are initially undefined and only get values after fitting a
    model. The latter is dealt with within this function, and the former within
    the helper function _compare_equality_for_lmmm().

    Args:
      other: Dataclass to compare against.

    Returns:
      Boolean for whether self == other; NotImplemented if other is not a
      LightweightMMM.
    """
    if not isinstance(other, LightweightMMM):
      return NotImplemented

    def _create_list_of_attributes_to_compare(
        mmm_instance: Any) -> Sequence[str]:
      all_attributes_that_can_be_compared = sorted(
          [x.name for x in dataclasses.fields(mmm_instance) if x.compare])
      attributes_which_have_been_instantiated = [
          x for x in all_attributes_that_can_be_compared
          if hasattr(mmm_instance, x)
      ]
      return attributes_which_have_been_instantiated

    self_attributes = _create_list_of_attributes_to_compare(self)
    other_attributes = _create_list_of_attributes_to_compare(other)

    return all(
        _compare_equality_for_lmmm(getattr(self, a1), getattr(other, a2))
        for a1, a2 in itertools.zip_longest(self_attributes, other_attributes))

  def _preprocess_custom_priors(
      self,
      custom_priors: Dict[str, Prior]) -> MutableMapping[str, Prior]:
    """Preprocesses the user input custom priors to Numpyro distributions.

    If numpyro distributions are given they remains untouched, however if any
    other option is passed, it is passed to the default distribution to alter
    its constructor values.

    Args:
      custom_priors: Mapping of the name of the prior to its custom value.

    Returns:
      A mapping of names to numpyro distributions based on user input and
        default values.
    """
    default_priors = {
        **models._get_default_priors(),
        **models._get_transform_default_priors()[self.model_name]
    }
    # Checking that the key is contained in custom_priors has already been done
    # at this point in the fit function.
    for prior_name in custom_priors:
      if isinstance(custom_priors[prior_name], numbers.Number):
        custom_priors[prior_name] = default_priors[prior_name].__class__(
            custom_priors[prior_name])
      elif (isinstance(custom_priors[prior_name], collections.abc.Sequence) and
            not isinstance(custom_priors[prior_name], str)):
        custom_priors[prior_name] = default_priors[prior_name].__class__(
            *custom_priors[prior_name])
      elif isinstance(custom_priors[prior_name], dict):
        custom_priors[prior_name] = default_priors[prior_name].__class__(
            **custom_priors[prior_name])
      elif not isinstance(custom_priors[prior_name], dist.Distribution):
        raise ValueError(
            "Priors given must be a Numpyro distribution or one of the "
            "following to fit in the constructor of our default Numpyro "
            "distribution. It could be given as args or kwargs as long as it "
            "is the correct format for such object. Please refer to our "
            "documentation on custom priors to know more.")
    return custom_priors

  def fit(
      self,
      media: jnp.ndarray,
      media_prior: jnp.ndarray,
      target: jnp.ndarray,
      extra_features: Optional[jnp.ndarray] = None,
      degrees_seasonality: int = 2,
      seasonality_frequency: int = 52,
      weekday_seasonality: bool = False,
      media_names: Optional[Sequence[str]] = None,
      number_warmup: int = 1000,
      number_samples: int = 1000,
      number_chains: int = 2,
      target_accept_prob: float = .85,
      init_strategy: Callable[[Mapping[Any, Any], Any],
                              jnp.ndarray] = numpyro.infer.init_to_median,
      custom_priors: Optional[Dict[str, Prior]] = None,
      seed: Optional[int] = None) -> None:
    """Fits MMM given the media data, extra features, costs and sales/KPI.

    For detailed information on the selected model please refer to its
    respective function in the models.py file.

    Args:
      media: Media input data. Media data must have either 2 dims for national
        model or 3 for geo models.
      media_prior: Costs of each media channel. The number of cost values must
        be equal to the number of media channels.
      target: Target KPI to use, like for example sales.
      extra_features: Other variables to add to the model.
      degrees_seasonality: Number of degrees to use for seasonality. Default is
        2.
      seasonality_frequency: Frequency of the time period used. Default is 52 as
        in 52 weeks per year.
      weekday_seasonality: In case of daily data, also estimate seven weekday
        parameters.
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
      custom_priors: The custom priors we want the model to take instead of the
        default ones. Refer to the full documentation on custom priors for
        details.
      seed: Seed to use for PRNGKey during training. For better replicability
        run all different trainings with the same seed.
    """
    if media.ndim not in (2, 3):
      raise ValueError(
          "Media data must have either 2 dims for national model or 3 for geo "
          "models.")
    if media.ndim == 3 and media_prior.ndim == 1:
      media_prior = jnp.expand_dims(media_prior, axis=-1)

    if media.shape[1] != len(media_prior):
      raise ValueError("The number of data channels provided must match the "
                       "number of cost values.")
    if media.min() < 0:
      raise ValueError("Media values must be greater or equal to zero.")

    if custom_priors:
      not_used_custom_priors = set(custom_priors.keys()).difference(
          self._prior_names)
      if not_used_custom_priors:
        raise ValueError(
            "The following passed custom priors dont have a match in the model."
            " Please double check the names have been written correctly: %s" %
            not_used_custom_priors)
      custom_priors = self._preprocess_custom_priors(
          custom_priors=custom_priors)
      geo_custom_priors = set(custom_priors.keys()).intersection(
          models.GEO_ONLY_PRIORS)
      if media.ndim == 2 and geo_custom_priors:
        raise ValueError(
            "The given data is for national models but custom_prior contains "
            "priors for the geo version of the model. Please either remove geo "
            "priors for national model or pass media data with geo dimension.")
    else:
      custom_priors = {}

    if weekday_seasonality and seasonality_frequency == 52:
      logging.warn("You have chosen daily seasonality and frequency 52 "
                   "(weekly), please check you made the right seasonality "
                   "choices.")

    if extra_features is not None:
      extra_features = jnp.array(extra_features)

    if seed is None:
      seed = utils.get_time_seed()

    train_media_size = media.shape[0]
    kernel = numpyro.infer.NUTS(
        model=self._model_function,
        target_accept_prob=target_accept_prob,
        init_strategy=init_strategy)

    mcmc = numpyro.infer.MCMC(
        sampler=kernel,
        num_warmup=number_warmup,
        num_samples=number_samples,
        num_chains=number_chains)
    mcmc.run(
        rng_key=jax.random.PRNGKey(seed),
        media_data=jnp.array(media),
        extra_features=extra_features,
        target_data=jnp.array(target),
        media_prior=jnp.array(media_prior),
        degrees_seasonality=degrees_seasonality,
        frequency=seasonality_frequency,
        transform_function=self._model_transform_function,
        weekday_seasonality=weekday_seasonality,
        custom_priors=custom_priors)

    self.custom_priors = custom_priors
    if media_names is not None:
      self.media_names = media_names
    else:
      self.media_names = [f"channel_{i}" for i in range(media.shape[1])]
    self.n_media_channels = media.shape[1]
    self.n_geos = media.shape[2] if media.ndim == 3 else 1
    self._media_prior = media_prior
    self.trace = mcmc.get_samples()
    self._number_warmup = number_warmup
    self._number_samples = number_samples
    self._number_chains = number_chains
    self._target = target
    self._train_media_size = train_media_size
    self._degrees_seasonality = degrees_seasonality
    self._seasonality_frequency = seasonality_frequency
    self._weekday_seasonality = weekday_seasonality
    self.media = media
    self._extra_features = extra_features# jax-devicearray
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
      static_argnames=("degrees_seasonality", "weekday_seasonality",
                       "transform_function", "model"))
  def _predict(
      self,
      rng_key: jnp.ndarray,
      media_data: jnp.ndarray,
      extra_features: Optional[jnp.ndarray],
      media_prior: jnp.ndarray,
      degrees_seasonality: int, frequency: int,
      transform_function: Callable[[Any], jnp.ndarray],
      weekday_seasonality: bool,
      model: Callable[[Any], None],
      posterior_samples: Dict[str, jnp.ndarray],
      custom_priors: Dict[str, Prior]
      ) -> Dict[str, jnp.ndarray]:
    """Encapsulates the numpyro.infer.Predictive function for predict method.

    It serves as a helper jitted function for running predictions.

    Args:
      rng_key: A jax.random.PRNGKey.
      media_data: Media array for needed for the model to run predictions.
      extra_features: Extra features for needed for the model to run.
      media_prior: Cost prior used for training the model.
      degrees_seasonality: Number of degrees for the seasonality.
      frequency: Frequency of the seasonality.
      transform_function: Media transform function to use within the model.
      weekday_seasonality: Allow daily weekday estimation.
      model: Numpyro model to use for numpyro.infer.Predictive.
      posterior_samples: Mapping of the posterior samples.
      custom_priors: The custom priors we want the model to take instead of the
        default ones. Refer to the full documentation on custom priors for
        details.

    Returns:
      The predictions for the given data.
    """
    return infer.Predictive(
        model=model, posterior_samples=posterior_samples)(
            rng_key=rng_key,
            media_data=media_data,
            extra_features=extra_features,
            media_prior=media_prior,
            target_data=None,
            degrees_seasonality=degrees_seasonality,
            frequency=frequency,
            transform_function=transform_function,
            custom_priors=custom_priors,
            weekday_seasonality=weekday_seasonality)

  def predict(
      self,
      media: jnp.ndarray,
      extra_features: Optional[jnp.ndarray] = None,
      media_gap: Optional[jnp.ndarray] = None,
      target_scaler: Optional[preprocessing.CustomScaler] = None,
      seed: Optional[int] = None
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
      seed: Seed to use for PRNGKey during sampling. For replicability run
        this function and any other function that utilises predictions with the
        same seed.

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
                jnp.zeros((media_gap.shape[0], *self._extra_features.shape[1:]))
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
    if seed is None:
      seed = utils.get_time_seed()
    prediction = self._predict(
        rng_key=jax.random.PRNGKey(seed=seed),
        media_data=full_media,
        extra_features=full_extra_features,
        media_prior=jnp.array(self._media_prior),
        degrees_seasonality=self._degrees_seasonality,
        frequency=self._seasonality_frequency,
        weekday_seasonality=self._weekday_seasonality,
        transform_function=self._model_transform_function,
        model=self._model_function,
        custom_priors=self.custom_priors,
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
      unscaled_costs: Optional[jnp.ndarray] = None,
      cost_scaler: Optional[preprocessing.CustomScaler] = None,
      target_scaler: Optional[preprocessing.CustomScaler] = None
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """It estimates the media contribution percentage and ROI of each channel.

    If data was scaled prior to training then the target and costs scalers need
    to be passed to this function to correctly calculate media contribution
    percentage and ROI in the unscaled space.

    Args:
      unscaled_costs: Optionally you can pass new costs to get these set of
        metrics. If None, the costs used for training will be used for
        calculating ROI.
      cost_scaler: Scaler that was used to scale the cost data before training.
        It is ignored if 'unscaled_costs' is provided.
      target_scaler: Scaler that was used to scale the target before training.

    Returns:
      media_contribution_hat_pct: The average media contribution percentage for
      each channel.
      roi_hat: The return on investment of each channel calculated as its
      contribution divided by the cost.

    Raises:
      NotFittedModelError: When the this method is called without the model
        being trained previously.
    """
    if not hasattr(self, "trace"):
      raise NotFittedModelError(
          "LightweightMMM has not been fitted and cannot run estimations. "
          "Please first fit the model.")
    if unscaled_costs is None and not cost_scaler:
      logging.warning(
          "Unscaled cost data or cost scaler were not given and  "
          "therefore unscaling wont be applied to calculcate contribution"
          " and ROI. If data was not scaled prior to training "
          "please ignore this warning.")
    if not target_scaler:
      logging.warning("Target scaler was not given and unscaling of the target "
                      "will not occur. If your target was not scaled prior to "
                      "training you can ignore this warning.")
    if unscaled_costs is None:
      if cost_scaler:
        unscaled_costs = cost_scaler.inverse_transform(self._media_prior)
      else:
        unscaled_costs = self._media_prior

    if self.media.ndim == 3:
      # cost shape (channel, geo) -> add a new axis to (channel, geo, sample)
      unscaled_costs = unscaled_costs = unscaled_costs[:, :, jnp.newaxis]
      # reshape cost to (sample, channel, geo)
      unscaled_costs = jnp.einsum("cgs->scg", unscaled_costs)

    # get the scaled posterior prediction
    posterior_pred = self.trace["mu"]
    if target_scaler:
      unscaled_posterior_pred = target_scaler.inverse_transform(posterior_pred)
    else:
      unscaled_posterior_pred = posterior_pred

    if self.media.ndim == 2:
      # s for samples, t for time, c for media channels
      einsum_str = "stc, sc -> sc"
    elif self.media.ndim == 3:
      # s for samples, t for time, c for media channels, g for geo
      einsum_str = "stcg, scg -> scg"

    media_contribution = jnp.einsum(einsum_str, self.trace["media_transformed"],
                                    jnp.squeeze(self.trace["coef_media"]))

    # aggregate posterior_pred across time:
    sum_scaled_prediction = jnp.sum(posterior_pred, axis=1)
    # aggregate unscaled_posterior_pred across time:
    sum_unscaled_prediction = jnp.sum(unscaled_posterior_pred, axis=1)

    if self.media.ndim == 2:
      # add a new axis to represent channel:(sample,) -> (sample,channel)
      sum_scaled_prediction = sum_scaled_prediction[:, jnp.newaxis]
      sum_unscaled_prediction = sum_unscaled_prediction[:, jnp.newaxis]

    elif self.media.ndim == 3:
      # add a new axis to represent channel:(sample,geo) -> (sample,geo,channel)
      # note: the total prediction value stays the same for all channels
      sum_scaled_prediction = sum_scaled_prediction[:, jnp.newaxis, :]
      # add a new axis to represent channel:(sample,geo) -> (sample,geo,channel)
      # note: the total prediction value stays the same for all channels
      sum_unscaled_prediction = sum_unscaled_prediction[:, :, jnp.newaxis]
      # reshape the array (sample,geo,channel) -> (sample,channel,geo)
      sum_unscaled_prediction = jnp.einsum("sgc->scg", sum_unscaled_prediction)

    # media contribution pct = media contribution / prediction
    # for geo level model:
    # media_contribution shape (sample, channel, geo)
    # sum_scaled_prediction shape (sample, channel, geo)
    # -> media_contribution_hat shape (sample, channel, geo)
    media_contribution_hat = media_contribution / sum_scaled_prediction

    # media roi = unscaled prediction * media contribution pct / unscaled costs
    # for geo leve model:
    # sum_unscaled_prediction shape (sample, channel, geo)
    # media_contribution_hat shape (sample, channel, geo)
    # unscaled_costs shape (sample, channel, geo)
    # -> roi_hat shape (sample, channel, geo)
    roi_hat = sum_unscaled_prediction * media_contribution_hat / unscaled_costs

    return media_contribution_hat, roi_hat
