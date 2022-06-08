LightweightMMM
===============
.. currentmodule:: lightweight_mmm


LightweightMMM object
======================
.. currentmodule:: lightweight_mmm.lightweight_mmm
.. autosummary::
      LightweightMMM

.. autoclass:: LightweightMMM
    :members:


Preprocessing / Scaling
========================
.. currentmodule:: lightweight_mmm.preprocessing
.. autosummary::
    CustomScaler

.. autoclass:: CustomScaler
    :members:


Optimize Media
===============
.. currentmodule:: lightweight_mmm.optimize_media
.. autosummary::
    find_optimal_budgets

.. autofunction:: find_optimal_budgets

Plot
=====
.. currentmodule:: lightweight_mmm.plot
.. autosummary::
    plot_response_curves
    plot_cross_correlate
    plot_var_cost
    plot_model_fit
    plot_out_of_sample_model_fit
    plot_media_channel_posteriors
    plot_bars_media_metrics

.. autofunction:: plot_response_curves
.. autofunction:: plot_cross_correlate
.. autofunction:: plot_var_cost
.. autofunction:: plot_model_fit
.. autofunction:: plot_out_of_sample_model_fit
.. autofunction:: plot_media_channel_posteriors
.. autofunction:: plot_bars_media_metrics

Models
=======
.. currentmodule:: lightweight_mmm.models
.. autosummary::
    transform_adstock
    transform_hill_adstock
    transform_carryover
    media_mix_model

.. autofunction:: transform_adstock
.. autofunction:: transform_hill_adstock
.. autofunction:: transform_carryover
.. autofunction:: media_mix_model

Media Transforms
=================
.. currentmodule:: lightweight_mmm.media_transforms
.. autosummary::
    calculate_seasonality
    adstock
    hill
    carryover
    apply_exponent_safe

.. autofunction:: calculate_seasonality
.. autofunction:: adstock
.. autofunction:: hill
.. autofunction:: carryover
.. autofunction:: apply_exponent_safe

Utils
======
.. currentmodule:: lightweight_mmm.utils
.. autosummary::
    save_model
    load_model
    simulate_dummy_data
    get_halfnormal_mean_from_scale
    get_halfnormal_scale_from_mean
    get_beta_params_from_mu_sigma
    distance_pior_posterior
    interpolate_outliers

.. autofunction:: save_model
.. autofunction:: load_model
.. autofunction:: simulate_dummy_data
.. autofunction:: get_halfnormal_mean_from_scale
.. autofunction:: get_halfnormal_scale_from_mean
.. autofunction:: get_beta_params_from_mu_sigma
.. autofunction:: distance_pior_posterior
.. autofunction:: interpolate_outliers
