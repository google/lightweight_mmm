## Input
*Which media channel metrics can be used as input?* 

You can use impressions, clicks or cost, especially for non digital data. For TV you could e.g. use TV rating points or cost. The model only takes the variation within a channel into account.

*Can I run MMM at campaign level?*

We generally don't recommend this. MMM is a macro tool that works well at the channel level. If you use distinct campaigns that have hard starts and stops, you risk losing the memory of the Adstock.
If you are interested in more granular insights, we recommend data-driven multi-touch attribution for your digital channels. You can find an example open-source package [here](https://github.com/google/fractribution/tree/master/py).

*What is the ideal ratio between %train and %test data for LMMM?*

Remember we treat LMMM not as a forecasting model, so test data is not always needed. When opting for a train / test split you can use at least 13 weeks for a test. However we recommend refraining from a too log testing period and possibly consider running a separate model that is specifically built for forecasting.

*What are best practices for lead generating businesses, with long sales cycles?*

It really depends on your target variable, i.e. what outcome you would like to measure. If generating a lead takes multiple months, you can take more immediate action KPIs like ‘number of conversions’ or ‘number of site visits, form entries’ into account.



## Modelling


*What can I do if the baseline is too low and total media contribution is too high?*

You can try various things:
1) You can include non-media variables.
2) You can lower the prior for the beta (in front of the transformed media).
3) You can set a bigger prior for the intercept.


*What are the different ways we can inform the media priors?*

By default, the media priors are informed by the costs, channels with more spend get bigger priors. 
You can also base media priors on (geo) experiments or use a heuristic like "the percentage of weeks a channel was used". The intuition behind this is that the more a channel is used, the more a marketer believes its contribution should be high.
Outputs from multi-touch attribution (MTA) can also be used as priors for an MMM.
Think with Google has recently published an [article](https://www.thinkwithgoogle.com/_qs/documents/13385/TwGxOP_Unified_Marketing_Measurement.pdf) on combining results from MTA and MMM. 


*How should I refresh my model and how often?*

This depends on the data frequency (daily, weekly) but also in what time frame the marketer makes decisions. If decisions are quarterly, we'd recommend to run the model each quarter. 
The data window can be expanded each time, so that older data still has an influence on the most recent estimate. Alternatively, old data can also be discarded, for instance if media effectiveness and strategies have changed more drastically over time. Note however that you can always use the posteriors from a previous modeling cycle as priors when you refresh the model.

*Why is your model additive?*

We might make the model multiplicative in future versions, but to keep simple and lightweight we have opted for the additive model for now.

*How does MCMC sampling works in LMMM?*

LMMM uses the [NUTS algorithm](https://mc-stan.org/docs/2_18/stan-users-guide/sampling-difficulties-with-problematic-priors.html) to solve the budget allocation question. NUTS only cares about priors and posteriors for each parameter and uses all of the data.

## Evaluation

*How important is OOS predictive performance?*

Remember MMM is not a forecasting model but an contribution and optimisation tool. Test performance should be looked at but contribution is more important.

*Which metric is recommended for evaluating goodness of fit on test data?*

We recommend looking at MAPE or median APE instead of the R-squared metric, as those are more interpretable from a business perspective and less influenced by outliers. 

*How is media effectiveness defined?*

Media effectiveness shows you how much each media channel percentually contributes to the target variable (e.g. y := Sum of sales).