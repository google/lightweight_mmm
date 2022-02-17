# Lightweight (Bayesian) Media Mix Model

##### This is not an official Google product.

LightweightMMM &#129415; is a lightweight Bayesian [media mix modeling](https://en.wikipedia.org/wiki/Marketing_mix_modeling)
library that allows users to easily train MMMs and obtain channel attribution 
information. The library also includes capabilities for optimizing media 
allocation as well as plotting common graphs in the field.

It is built in [python3](https://www.python.org/) and makes use of 
[Numpyro](https://github.com/pyro-ppl/numpyro) and [JAX](https://github.com/google/jax).

## What you can do with LightweightMMM

- Scale you data for training.
- Easily train your media mix model.
- Evaluate your model.
- Learn about your media attribution and ROI per media channel.
- Optimize your budget allocation.

## Installation

The recommended way of installing lightweight_mmm is through PyPi:

`pip install lightweight_mmm`

If you want to use the most recent and slightly less stable version you can 
install it from github:

`pip install --upgrade git+https://github.com/google/lightweight_mmm.git`

## The models

**For larger countries we recommend a geo-based model, this is not implemented
yet.**

We estimate a **national** weekly model where we use sales revenue (y) as the KPI.

<img src="https://raw.githubusercontent.com/google/lightweight_mmm/main/images/simplified_model_formula.png" alt="simplified_model_formula"></img>

where `X_m` is a media matrix and `X_o` is a matrix of other exogenous variables.

Seasonality is a latent sinusoidal parameter with a repeating pattern.

Media parameter `beta_m` is informed by costs. It uses a HalfNormal distribution and
the scale of the distribution is the total cost of each media channel.

`sat()` is a saturation function and `lag()` is a lagging function, eg Adstock.
Each of them can have their respective parameters.

We have three different versions of the MMM with different lagging and
saturation and we recommend you compare all three models. The Adstock and carryover
models have an exponent for diminishing returns. The Hill functions covers that
functionality for the Hill-Adstock model.
- [Adstock](https://en.wikipedia.org/wiki/Advertising_adstock): Applies an infinite lag that decreases its weight as time passes.
- [Hill-Adstock](https://en.wikipedia.org/wiki/Hill_equation_(biochemistry)): Applies a sigmoid like function for diminishing returns to the output of the adstock function.
- [Carryover](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46001.pdf): Applies a [causal convolution](https://paperswithcode.com/method/causal-convolution) giving more weight to the near values than distant ones.

## Scaling

Scaling is a bit of an art, Bayesian techniques work well if the input data is
small scale. We should not center variables at 0. Sales and media should have a
lower bound of 0.

1. `y` can be scaled as `y / jnp.mean(y)`.
2. `media` can be scaled as `X_m / jnp.mean(X_m, axis=0)`, which means the new column mean will be 1.

## Optimization

For optimization we will maximize the sales changing the media inputs such that
the summed cost of the media is constant. We can also allow reasonable bounds
on each media input (eg +- x%). We only optimise across channels and not over
time.


## Getting started

### Preparing the data
Here we use simulated data but it is assumed you have your data cleaned at this
point. The necessary data will be:

- Media data: Containing the metric per channel and time span (eg. impressions
  per week). Media values must not contain negative values.
- Extra features: Any other features that one might want to add to the analysis.
  These features need to be known ahead of time for optimization or you would need
  another model to estimate them.
- Target: Target KPI for the model to predict. This will also be the metric
  optimized during the optimization phase.
- Costs: The average cost per media unit per channel.

```
# Let's assume we have the following datasets with the following shapes (we use
the `simulate_dummy_data` function in utils for this example):
media_data, extra_features, target, costs = utils.simulate_dummy_data(
    data_size=160,
    n_media_channels=3,
    n_extra_features=2)
```
Scaling is a bit of an art, Bayesian techniques work well if the input data is
small scale. We should not center variables at 0. Sales and media should have a
lower bound of 0.

We provide a `CustomScaler` which can apply multiplications and division scaling
in case the wider used scalers don't fit your use case. Scale your data
accordingly before fitting the model.
Below is an example of usage of this `CustomScaler`:

```
# Simple split of the data based on time.
split_point = data_size - data_size // 10
media_data_train = media_data[:split_point, :]
target_train = target[:split_point]
extra_features_train = extra_features[:split_point, :]
extra_features_test = extra_features[split_point:, :]

# Scale data
media_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
extra_features_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
target_scaler = preprocessing.CustomScaler(
    divide_operation=jnp.mean)
# scale cost up by N since fit() will divide it by number of weeks
cost_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)

media_data_train = media_scaler.fit_transform(media_data_train)
extra_features_train = extra_features_scaler.fit_transform(
    extra_features_train)
target_train = target_scaler.fit_transform(target_train)
costs = cost_scaler.fit_transform(unscaled_costs)
```

### Training the model

The model requires the media data, the extra features, the costs of each media
unit per channel and the target. You can also pass how many samples you would
like to use as well as the number of chains. 

For running multiple chains in parallel the user would need to set `numpyro.set_host_device_count` to either the number of chains or the number of CPUs available.

See an example below:

```
# Fit model.
mmm = lightweight_mmm.LightweightMMM()
mmm.fit(media=media_data,
        extra_features=extra_features,
        total_costs=costs,
        target=target,
        number_warmup=1000,
        number_samples=1000,
        number_chains=2)
```

### Obtaining media effect and ROI

There are two ways of obtaining the media effect and ROI with `lightweightMMM`
depending on if you scaled the data or not prior to training. If you did not
scale your data you can simply call:
```
mmm.get_posterior_metrics()
```

### Running the optimization

For running the optimization one needs the following main parameters:

- `n_time_periods`: The number of time periods you want to simulate (eg. Optimize
  for the next 10 weeks if you trained a model on weekly data).
- The model that was trained.
- The `budget` you want to allocate for the next `n_time_periods`.
- The extra features used for training for the following `n_time_periods`.
- Price per media unit per channel.
- `media_gap` refers to the media data gap between the end of training data and
  the start of the out of sample media given. Eg. if 100 weeks of data were used
  for training and prediction starts 2 months after training data finished we
  need to provide the 8 weeks missing between the training data and the
  prediction data so data transformations (adstock, carryover, ...) can take
  place correctly.

See below and example of optimization:

```
# Run media optimization.
budget = 40
prices = np.array([0.1, 0.11, 0.12])
extra_features_test = extra_features_scaler.transform(extra_features_test)
solution = optimize_media.find_optimal_budgets(
    n_time_periods=extra_features_test.shape[0],
    media_mix_model=mmm,
    budget=budget,
    extra_features=extra_features_test,
    prices=prices)
```

## Run times

A model with 5 media variables and 1 other variable and 150 weeks, 1500 draws
and 2 chains should take 6 mins per chain to estimate (on CPU machine).

## References

- [Jin, Y., Wang, Y., Sun, Y., Chan, D., & Koehler, J. (2017). Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects. Google Inc.](https://research.google/pubs/pub46001/)

- [Chan, D., & Perry, M. (2017). Challenges and Opportunities in Media Mix Modeling.](https://research.google/pubs/pub45998/)

- [Sun, Y., Wang, Y., Jin, Y., Chan, D., & Koehler, J. (2017). Geo-level Bayesian Hierarchical Media Mix Modeling.](https://research.google/pubs/pub46000/)


