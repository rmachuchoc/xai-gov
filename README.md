# xai-gov

Code and study pipeline for the paper **"Avoiding Inadmissible Defaults in
Shielded Constrained POMDPs under Distribution Shift"** (under revision at
*Machine Learning and Knowledge Extraction*, MDPI; manuscript make-4658289).

The question the code answers is narrow. When a shield blocks an unsafe action,
what should it put in its place? A **fallback** shield substitutes a fixed
default, usually "do nothing". A **projection** shield substitutes the nearest
action inside the admissible envelope. The two are often treated as
interchangeable. They are not when doing nothing is itself unsafe, and in
inventory replenishment it often is. Version 2 adds a third option, an
**admissible fallback** (F-backup), and that comparison turned out to decide
the paper's claim.

## What the study found

All numbers come from the confirmatory campaign of v2.0.0: nine arms, 500
seed-paired replicates each (4500 runs) of a serial supply chain with
backorders, 60 periods per run. The outcome is the retail fill rate β, the
share of customer demand served from stock in the period it arrives.

| Base policy | No supervision | Null fallback | Projection | Admissible fallback |
| --- | --- | --- | --- | --- |
| Fixed robust policy, three echelons | 0.745 | 0.067 | 0.621 | 0.749 |
| Tabular Q-learning, three echelons | — | 0.063 | 0.681 | — |
| Fixed robust policy, five echelons | — | 0.195 | 0.651 | — |

Six hypotheses were sealed before the first run, with a threshold of
2k/α = 240 on the e-value. All six were supported:

- **H1, H2.** Projection beats the null fallback under the fixed policy
  (Δβ = +0.554) and under Q-learning (+0.618).
- **H3.** The admissible fallback beats projection (−0.127 for projection).
- **H4.** The mediator without a shield lowers β from 0.745 to 0.067 on the
  same horizon, because every intervention it makes executes as a hold.
- **H5.** The projection gain over the null fallback replicates on five
  echelons (+0.455).
- **H6.** It holds after excluding the start-up periods (+0.708).

Under the decision rules sealed with the plan, this outcome means the paper
claims that the admissibility of the replacement matters, not the minimality
of the correction.

Three things the study does **not** claim:

- **A formal safety guarantee.** The envelope is an operating approximation
  derived from the specification. No arm meets the bounds of two of the three
  safety properties, and the violation frequencies are reported per property
  with exact intervals.
- **A false-alarm guarantee for the change martingale.** Its conditions do not
  hold for the implemented detector. The false-alarm rate is measured instead:
  0 of 200 stationary runs.
- **External validity beyond the calibration gap.** Calibrated against 30 M5
  store-by-category series, the simulator matches the moments of real demand
  but not its persistence (median posterior predictive gap 9.21; 4.13 with an
  AR(1) term).

## Releases

| Tag | Data (Zenodo) | Notes |
| --- | --- | --- |
| `v2.0.0` | [10.5281/zenodo.24148686](https://doi.org/10.5281/zenodo.24148686) | Current. New engine, β, F-backup, aGRAPA, sealed H1–H6, 4500 confirmatory runs |
| `v1.1.0` | [10.5281/zenodo.23484768](https://doi.org/10.5281/zenodo.23484768) | Superseded. Its simulator had the two defects described in `CHANGELOG-v2.0.0.md` |
| `v1.0.1` | [10.5281/zenodo.23091494](https://doi.org/10.5281/zenodo.23091494) | Superseded |

Each tag points to the exact commit that produced its data, and the Zenodo
bundle records that commit in `PROVENANCE.txt`.

**Do not use the numbers of v1.x.** A reviewer found that the v1 engine
counted supplier shipments as customer service (rates above one) and lost
backordered material. `CHANGELOG-v2.0.0.md` describes both defects and
everything that changed.

## Installation

Python 3.12 on Linux. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev,figures]"
make check        # ruff, mypy and 844 tests
```

The network diagram is converted from SVG with `rsvg-convert`. On Ubuntu or
Debian, install it with `sudo apt install librsvg2-bin`.

If ROS or another framework is on your `PYTHONPATH`, the `Makefile` clears it
for every target, so the test run stays isolated.

## Reproducing the study

Every run is deterministic given its seed, so the commands below reproduce the
archived results exactly. They are listed in the order the protocol ran them.
Results go to `outputs/v2/`. The scripts accept `--workers` (at most 6).

```bash
PY=.venv/bin/python

# 1. Pilot on the new engine (its own seed), then simulated power
$PY scripts/run_pilot_v2.py --outputs-root /tmp/pilot_v2
$PY scripts/power_v2.py

# 2. Check the plan against the seal (the plan is already sealed)
$PY scripts/seal_v2.py --check

# 3. Confirmatory campaign: 9 arms x 500 replicates
$PY scripts/run_confirmatory_v2.py --outputs-root outputs/v2
$PY scripts/posthoc_v2.py --outputs-root outputs/v2
$PY scripts/recompute_cs_v2.py --outputs-root outputs/v2

# 4. Exploratory campaigns, detector null line and sweeps
$PY scripts/run_descriptor_exploratory_v2.py --outputs-root outputs/v2
$PY scripts/run_detector_exploratory_v2.py --outputs-root outputs/v2
$PY scripts/run_detector_null.py --outputs-root outputs/v2 \
    --output outputs/v2/analysis/detector_null.json
make sweeps       # writes to outputs/v2 by default

# 5. Theorem A1 conditions, parameter table, worked examples, figures
$PY scripts/verify_theorem_a1.py --output outputs/v2/analysis/theorem_a1.json
$PY scripts/parameter_table_v2.py
$PY scripts/worked_examples_v2.py
make figures-v2
```

The sealed plan is `specs/v2/confirmatory_v2.seal.json`. The confirmatory
runner refuses to start if the plan no longer matches its seal.
`specs/v2/CHRONOLOGY.md` lists, with dates and commits, what was fixed before
the first run, what was inspected before the seal, and the eight amendments
of the v1 plan.

The campaign reports are the sealed record and are never rewritten. The
confidence sequences cited in the paper are in `confidence_sequences_v2.json`
next to each report.

## Real-data calibration (M5)

The M5 series are licensed and are **not** in the repository. Download the
competition files yourself, then:

```bash
make m5 M5_SOURCE=~/Downloads/m5    # writes 30 series to data/demand/
make calibrate                       # weekly cadence, i.i.d. and AR(1) fits
make caltable                        # LaTeX table for the manuscript
```

`data/demand/README.md` explains the aggregation level and the cadence choice.

## Repository layout

```
src/xai_gov/
  simulation/     serial chain with backorders, demand regimes, disruptions
  policies/       heuristic, robust DRO, random, tabular Q-learning
  conformal/      scores, calibrators, adaptive level, martingales, detector
  governance/     belief filter, threshold solver, budget, CPOMDP mediator
  verification/   specification, envelope, shield with its three operators
                  (projection, null fallback, base-stock backup)
  causal/         structural model, recourse, fidelity audit, descriptors
  analysis/       power, preregistration, campaigns, aGRAPA e-values,
                  confidence sequences, sweeps, calibration, figures
configs/          every experiment, arm and campaign as YAML
specs/v2/         sealed plan, pilot, power, parameters, chronology
scripts/          entry points for every step above
tests/            unit and integration tests; the propositions and the
                  material-conservation identity are assertions here
```

Three rules the code enforces, because each was learned from a defect that
completed silently:

1. **Material is conserved in every period.** Initial inventory plus
   production equals stock at the nodes plus material in transit plus sales.
   The v1 engine broke this without any test noticing.
2. **A bet uses only the past.** The aGRAPA fraction for replicate t depends
   on replicates before t, and its scale is fixed by the declared range.
3. **A cache key covers the configuration, not only the criteria.** Otherwise
   a changed arm is compared against a stale copy of itself.

## Use of generative AI

Claude Opus 4 and Claude Opus 5.5 (Anthropic) were used for code development
and debugging, formatting of results, and drafting and editing of text. All
generated code was checked with the test suite. The authors reviewed all
generated material and are responsible for the content of this repository and
of the paper.

## Citation

If you use the code, please cite the archived release:

> Ortiz, D.; Machucho, R.; Hernandez, J.; Garcia, J.; Herrera, H.; Lopez, J.
> *xai-gov study bundle v2.0.0* [Data set and software]. Zenodo, 2026.
> https://doi.org/10.5281/zenodo.24148686

## License

See `LICENSE`. The M5 data remain under their original competition terms.
