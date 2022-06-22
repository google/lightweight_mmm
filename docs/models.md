# Lightweight MMM Models.

The Lightweight MMM can either be run using data aggregated at the national level (standard approach) or using data aggregated at a geo level (sub-national hierarchical approach). These models are documented below.



## Standard Approach (National Level)

All the parameters in our Bayesian model have priors which have been set based on simulated studies that produce stable results. We also set out our three different approaches to saturation and lagging media effects: carryover (with exponent), adstock (with exponent) and hill adstock. Please see Jin, Y. et al., (2017) for more details on these models and choice of priors.

 $$kpi_{t} = \alpha + trend_{t} + seasonality_{t} + media\_channels_{t} + other\_factors_{t} $$

*Intercept*<br>
- $$\alpha \sim HalfNormal(0,2)$$

*Trend*<br>
- $$trend_{t} = \mu t^{\kappa + 0.5}$$<br>
- $$\mu \sim Normal(0,1)$$<br>
- $$\kappa \sim Beta(1,1)$$<br>
- $$t$$ is a linear trend input

*Seasonality (for models using* **weekly observations**)<br>
- $$seasonality_{t} = \sum_{d=1}^{D} (\gamma_{1,d} cos(\frac{2 \pi d}{s}) + \gamma_{2,d} sin(\frac{2 \pi d}{s}))$$<br>
- $$\gamma_{1,d}, \gamma_{2,d} \sim Normal(0,1)$$<br>
- Where $$s=52$$ to model a repeating 52-week yearly pattern and $$D=2$$

*Seasonality (for models using* **daily observations**)<br>
- $$seasonality_{t} = \sum_{d=1}^{D} (\gamma_{1,d} cos(\frac{2 \pi d}{s}) + \gamma_{2,d} sin(\frac{2 \pi d}{s})) + \sum_{i=1}^{7} \delta_{i}$$<br>
- $$\gamma_{1,d}, \gamma_{2,d} \sim Normal(0,1)$$<br>
- $$\delta_{i} \sim Normal(0,0.5)$$<br>
- Where $$s=365$$ to model a repeating 365-day yearly pattern and $$D=2$$

*Other factors*<br>
- $$other\_factors_{t} = \sum_{i=1}^{N} \lambda_{i}Z_{i}$$<br>
- $$\lambda_{i}  \sim  Normal(0,1)$$<br>
- Where $$Z_{i}$$ are other factors and $$N$$ is the number of other factors.

*Media Effect*<br>
- $$\beta_{m} \sim HalfNormal(0,v_{m})$$<br>
- Where $$v_{m}$$ is a scalar equal to the sum of the total cost of media channel $$m$$.

*Media Channels (for the* **carryover** *model)*<br>
- $$media\_channels_{t} = \frac{\sum_{l=0}^{L-1} \tau_{m}^{(l-\theta_{m})^2}x_{t-l,m}}{\sum_{l=0}^{L-1}\tau_{m}^{(l-\theta_{m})^2}}$$  where $$L=13$$<br>
- $$\tau_{m} \sim Beta(1,1)$$<br>
- $$\theta_{m} \sim HalfNormal(0,2)$$<br>
- Where $$x_{t,m}$$ is the media spend or impressions in week $$t$$ from media channel $$m$$<br>

*Media Channels (for the* **adstock** *model)*<br>
- $$media\_channels_{t} = x_{t,m,s}^{*\rho_{m}}$$<br>
- $$x_{t,m,s}^{*} = \frac{x^{*}_{t,m}}{1/(1 - \lambda_{m})}$$<br>
- $$x_{t,m}^{*} = x_{t,m} + \lambda_{m} x_{t-1,m}^{*}$$ where $$t=2,..,N$$<br>
- $$x_{1,m}^{*} = x_{1,m}$$<br>
- $$\lambda_{m} \sim Beta(2,1)$$<br>
- $$\rho_{m} \sim Beta(9,1)$$

*Media Channels (for the* **hill_adstock** *model)*<br>
- $$media\_channels_{t} = \frac{1}{1+(x^{*}_{t,m,s} / K_{m})^{-S_{m}}}$$<br>
- $$x^{*}_{t,m,s} = \frac{x^{*}_{t,m}}{1/(1-\lambda_{m})}$$<br>
- $$x_{t,m}^{*} = x_{t,m} + \lambda_{m} x_{t-1,m}^{*}$$ where $$t=2,..,N$$<br>
- $$x_{1,m}^{*} = x_{1,m}$$<br>
- $$K_{m} \sim Gamma(1,1)$$<br>
- $$S_{m} \sim Gamma(1,1)$$<br>
- $$\lambda_{m} \sim Beta(2,1)$$

## Geo level (sub-national hierarchical approach)

The hierarchical model is analogous to the standard model except there is an additional dimension of region. In the geo level model seasonality is learned at the sub-national level and at the national level. For more details on this model, please see Sun, Y et al., (2017).