# Lightweight (Bayesian) Media Mix Model

This Python package estimates a Bayesian MMM using Numypro.

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

Media parameter beta is informed by costs.

f() is a saturation function and lag() is a lagging function, eg Adstock.

We have three different versions of the MMM with different lagging and
saturation and we recommend you compare all three models.

## Scaling

Scaling is a bit of an art, Bayesian techniques work well if the input data is
small scale. We should not center variables at 0. Sales and media should have a
lower bound of 0.

1. For y we can scale as $$y / mean_y$$.
2. For media we can scale as $$X_m / mean_X$$, which means the new column
mean will be 1.

## Optimisation

For optimisation we will maximise the sales changing the media inputs such that
the summed cost of the media is constant. We can also allow reasonable bounds
on each media input (eg +- x%). We only optimise across channels and not over
time.

## Roadmap

- Geo data
- Daily data
- Experiments inform priors

## Getting started

### Preparing the data
Here we use simulated data but it is assumed you have you data cleaned at this
point. The necessary data will be:

- Media data: Containing the metric per channel and time span (eg. impressions
  per week). Media values must not contain negative values.
- Extra features: Any other features that one might want to add to the analysis.
  These features need to be known ahead of time for simulation or you would need
  another model to estimate them.
- Target: Target KPI for the model to predict. This will also be the metric
  optimized during the optimization phase.
- Costs: The average cost per media unit per channel.

```
# Lightweight MMM provides a utility for simulating data.
media_data, extra_features, target, unscaled_costs, _ = data_simulation.simulate_all_data(
    data_size=120,
    n_media_channels=3,
    n_extra_features=2)
```

We provide a `CustomScaler` which can apply multiplications and division scaling
in case the wider used scalers dont fit your use case. Scale your data
accordingly before fitting the model.
Below is an example of usage of this `CustomScaler`:

```
# Split data.
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
unit per channel and the target. You can also pass how many samples one would
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

There are two ways of obtaining the media effect and ROI with lightweightMMM
depending on if you scaled the data or not prior to training. If you did not
scale your data you can simply call:
```
mmm.get_posterior_metrics()
```
However if you scaled your media data, target or both it is important that you
provide `get_posterior_metrics` with the necessary information to unscale the
data and calculcate media effect and ROI.

- If only costs were scaled:

```
mmm.get_posterior_metrics(cost_scaler=cost_scaler)

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
# Run media optmization.
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

A model with 3 media variables and 1 other variable and 120 weeks, 1000 draws
and 2 chains should take [TBC] to estimate (on CPU machine). This
excludes compile time.

## How to debug the model

TODO
