# Interpretable metrics and a daily forecasting experiment

Keep MAE in EUR/MWh, and report relative MAE (rMAE) alongside it. A megawatt is
power; a megawatt-hour is energy: 1 MW sustained for an hour, or 1,000 kWh.
An MAE of 21 EUR/MWh is 0.021 EUR/kWh, or 2.1 euro cents/kWh of average absolute
wholesale price error. This is not a household tariff or an estimate of profit.

For comparison with a price level, 21 EUR/MWh is 21% of a hypothetical price of
100 EUR/MWh. That does not make the model's MAPE 21%: actual prices vary by hour
and market. Percentage errors per observation are unstable near zero and hard
to interpret for negative electricity prices. A price-level summary could use
sum(abs(error))/sum(abs(price)), provided the denominator is positive; label it
as normalized error rather than an average hourly percentage error.

The recommended skill metric is instead:

    rMAE = sum(abs(model error)) / sum(abs(weekly-persistence error))
    skill improvement (%) = 100 * (1 - rMAE)

Use exactly the same target hours in numerator and denominator. Values below
one improve on persistence; 0.72 means 28% less absolute error, not 28% of the
electricity price. This is an evaluation-period benchmark ratio, distinct from
MASE's training-period scaling. Macro country ratios and a pooled ratio have
different weights; identify which is reported. The milestone uses the pooled
matched-hour ratio. Persistence has no training-seed uncertainty. Bold numbers
identify the smallest mean, not statistical significance or a test-selected model.

[Lago et al. (2021)](https://doi.org/10.1016/j.apenergy.2021.116983) discuss
weekly-naive rMAE for electricity-price forecasting and problems with percentage
metrics. Archived price revisions and other availability caveats remain unchanged.

## Prospective forecasting as a product experiment

The previous phrase about a new chronological period meant evaluating forecasts
on days whose outcomes have not influenced development. A daily service makes
that concrete and also creates an engineering experiment. No service is deployed
by this report revision.

1. Start with a few resolved markets (e.g. FR, DE_LU and BE), weekly persistence
   and the price-only pooled XGBoost. Add fixed MLP/GAT challengers later.
2. Ingest data before the Brussels-noon cutoff and store raw responses, received
   timestamps, source identifiers and model versions. Write immutable predictions
   before noon; a late prediction remains a recorded deadline failure.
3. Fetch auction results separately after publication. Keep initial observations
   and subsequent revisions distinct; score the forecasts without rewriting them.
4. Report daily and rolling MAE, rMAE, source freshness/completeness, on-time
   forecast rate, latency and fallback frequency. Calculate rolling rMAE from
   error sums over the window, not the mean of daily ratios. If its denominator
   is zero, report it as undefined. Record benchmark fallbacks separately from
   model predictions so reliability does not conceal model failure.
5. Compare a frozen model with any scheduled-retraining challenger prospectively,
   versioning every change. Add temperature only from an observed feed whose
   receipt times we capture; retrospective ERA5 is not a current-temperature feed.

Public data access is feasible. ENTSO-E provides a REST API, with registration
and API access credentials described in its [official token guide](https://transparencyplatform.zendesk.com/hc/en-us/articles/12845911031188-How-to-get-security-token).
Public accessibility does not establish redistribution rights: check each
selected feed's terms before publishing raw data in a public dashboard.

The service must also validate its target resolution. SDAC changed to 15-minute
products for delivery on 1 October 2025 ([European Commission](https://energy.ec.europa.eu/news/eu-electricity-trading-day-ahead-markets-becomes-more-dynamic-2025-10-01_en)).
To retain our hourly task, verify the upstream hourly definition and use a
documented duration-weighted aggregation of complete quarter-hour intervals.
Preserve missing intervals and 23/25-hour DST days; do not silently switch to a
different forecasting target. Native quarter-hour forecasts would be a separate
experiment.
