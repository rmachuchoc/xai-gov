# Demand series for calibration

This directory is empty on purpose, and the emptiness is load-bearing: when no
series is present the calibration falls back to a simulator-generated
reference and labels the result *sim-to-sim*, which establishes no external
validity and is reported as a blocking diagnostic. Dropping a real series here
is what lifts that.

## Why the data is not committed

The M5 competition terms permit use but not redistribution, so the repository
carries the recipe and a checksum rather than the file.

```bash
# 1. accept the terms and download at
#    kaggle.com/competitions/m5-forecasting-accuracy
# 2. unzip it somewhere, then
make m5 M5_SOURCE=~/Downloads/m5
```

That writes one CSV per store-by-category series into this directory, each
carrying its aggregation level, cadence and source hash in its header
comments.

## Why store x category

M5's 30,490 bottom-level series are intermittent — long runs of zeros, a
documented property of the data. The twin's demand process is not
intermittent, so calibrating against one of those would fail for a reason the
study did not set out to test. The store-by-category level is smooth, retains
weekly seasonality, and has genuine lag-1 autocorrelation.

Autocorrelation is the point. The synthetic calibration reproduced the mean
and the dispersion of its reference and did not reproduce its *dynamics*, and
dynamics is what a disruption-response study depends on. The adapter therefore
rejects any series whose lag-1 autocorrelation falls below a floor: a series
with no dynamics cannot calibrate a dynamics statistic, and admitting one
would reproduce the synthetic failure under a real filename.

## The decision that still has to be made

A period of the twin is abstract. Mapping days onto periods is a modelling
choice that *determines* the autocorrelation, so it cannot be made by a
loader and it cannot be made after seeing the calibration result. Decide
whether one period is one day or one week, record it in the sealed plan, and
only then run the confirmatory campaign. The adapter emits daily series by
default and a weekly series only when asked, so the two cadences are separate
files rather than one file that quietly changed meaning.
