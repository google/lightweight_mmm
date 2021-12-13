# Lightweight (Bayesian) Media Mix Model

LightweightMMM is a lightweight Bayesian [media mix modeling](https://en.wikipedia.org/wiki/Marketing_mix_modeling)
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

## The models

**For larger countries we recommend a geo-based model, this is not implemented
yet**

We estimate a **national** weekly model where we use sales revenue (y) as the KPI.

$$\mu_t = a + trend_t + seasonality_t + \beta_m f(lag(X_{mt}, \phi_m), \theta_m) + ...$$

$$y_t \sim N(\mu_t, \sigma)$$

$$\sigma \sim \Gamma(1, 1)$$

$$\beta_m \sim N^+(0, ...)$$

$$X_m$$ is a media matrix.

Seasonality is a latent sinusoidal parameter with a repeating pattern.

Media parameter beta is informed by costs. It uses a HalfNormal distribution and
the scale of the distribution is the total cost of each media channel.

f() is a saturation function and lag() is a lagging function, eg Adstock.

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

1. `y` can be scaled as $$y / mean_y$$.
2. `media` can be scaled as $$X_m / mean_X$$, which means the new column mean will be 1.

## Optimization

For optimization we will maximize the sales changing the media inputs such that
the summed cost of the media is constant. We can also allow reasonable bounds
on each media input (eg +- x%). We only optimise across channels and not over
time.


## Getting started

### Preparing the data
Here we use simulated data but it is assumed you have you data cleaned at this
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
# Let's assume we have the following datasets with the following shapes:
media_data, extra_features, target, unscaled_costs, _ = data_simulation.simulate_all_data(
    data_size=120,
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
media_scaler = preprocessing.CustomScaler(divide_operation=np.mean)
extra_features_scaler = preprocessing.CustomScaler(divide_operation=np.mean)
target_scaler = preprocessing.CustomScaler(
    divide_operation=np.mean)
# scale cost up by N since fit() will divide it by number of weeks
cost_scaler = preprocessing.CustomScaler(divide_by=unscaled_costs.mean(),
    multiply_by=len(target_train))

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
        costs=costs,
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
However if you scaled your media data, target or both it is important that you
provide `get_posterior_metrics` with the necessary information to unscale the
data and calculate media effect and ROI.

- If only costs were scaled, the following two function calls are equivalent:

```
# Option 1
mmm.get_posterior_metrics(cost_scaler=cost_scaler)
# Option 2
mmm.get_posterior_metrics(unscaled_costs=unscaled_costs)
```

- If only the target was scaled:

```
mmm.get_posterior_metrics(target_scaler=target_scaler)
```

- If both were scaled:

```
mmm.get_posterior_metrics(cost_scaler=cost_scaler,
                          target_scaler=target_scaler)
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
and 2 chains should take 7 mins per chain to estimate (on CPU machine). This
excludes compile time.

## References

- [Jin, Y., Wang, Y., Sun, Y., Chan, D., & Koehler, J. (2017). Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects. Google Inc.](https://research.google/pubs/pub46001/)

- [Chan, D., & Perry, M. (2017). Challenges and Opportunities in Media Mix Modeling.](https://research.google/pubs/pub45998/)

- [Sun, Y., Wang, Y., Jin, Y., Chan, D., & Koehler, J. (2017). Geo-level Bayesian Hierarchical Media Mix Modeling.](https://research.google/pubs/pub46000/)


