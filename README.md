# xai-gov

Code and study pipeline for the paper **"Action Substitution versus Fallback
Shields in Constrained POMDPs under Distribution Shift"** (submitted to
*Machine Learning and Knowledge Extraction*, MDPI).

The question the code answers is narrow. When a shield blocks an unsafe action,
what should it put in its place? A **fallback** shield substitutes a fixed
default, usually "do nothing". A **projection** shield substitutes the nearest
action that satisfies the safety specification. The two are often treated as
interchangeable. They are not when doing nothing is itself unsafe, and in
inventory replenishment during a disruption, it usually is.

## What the study found

All numbers come from 800 seed-paired runs (20 arms × 40 replicates) of a
three-echelon supply-network twin under high-volatility demand.

| Base policy | No shield | Fallback shield | Projection shield |
| --- | --- | --- | --- |
| Fixed robust policy (DRO) | 0.611 | 0.059 | 0.670 |
| Tabular Q-learning, learning online | 0.671 | 0.063 | 0.649 |

Values are mean service levels. The projection–fallback gap is supported for
both policies (e = 1.62 × 10⁶ and 9.35 × 10⁵, against a familywise threshold
of 400). The Q-learning policy did not learn to avoid the fallback within the
60-period horizon.

Of ten preregistered hypotheses, four are supported, one is refuted in the
opposite direction (mediation with a null default *lowered* service by 0.55
instead of raising it), and five are inconclusive. The five inconclusive ones
concern the conformal detector and the explanation descriptors.

Two things the study does **not** claim:

- **Coverage.** The adaptive conformal detector under-flagged in 17 of 20 arms.
  Excluding the forecaster's warm-up from calibration moved the error in the
  direction the warm-up lemma predicts, but not past the threshold. No coverage
  guarantee is reported as holding.
- **External validity beyond the calibration gap.** Calibrated against 30 M5
  store-by-category series, the twin's demand process matches the moments of
  real demand but not its persistence (median posterior predictive gap 9.33;
  4.16 with an AR(1) term).

## Releases

| Tag | Data (Zenodo) | Notes |
| --- | --- | --- |
| `v1.1.0` | [10.5281/zenodo.23484768](https://doi.org/10.5281/zenodo.23484768) | Current. Adds Q-learning, the fallback shield, RQ10–RQ11, a predeclared e-value scale and the 2k/α threshold |
| `v1.0.1` | [10.5281/zenodo.23091494](https://doi.org/10.5281/zenodo.23091494) | Earlier submission; superseded |

Each tag points to the exact commit that produced its data, and the Zenodo
bundle records that commit in `PROVENANCE.txt`.

`CHANGELOG-v1.1.0.md` lists every change in the current release, including two
analysis corrections made before it was published. In short: the e-value
betting fraction used to be scaled by the largest observed difference, which
let each bet depend on data not yet seen. It is now the constant 1, and the
familywise threshold counts two directional tests per hypothesis.

## Installation

Python 3.12 on Linux. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
make check        # ruff, mypy and 631 tests
```

If ROS or another framework is on your `PYTHONPATH`, the `Makefile` clears it
for every target, so the test run stays isolated.

## Reproducing the study

Every run is deterministic given its seed, so the commands below reproduce the
archived results exactly.

```bash
make study        # confirmatory campaign: 20 arms × 40 replicates
make sweeps       # break-even and severity sweeps: 300 runs
```

`make study` refuses to start if the power check fails, seals the analysis plan
by hash before the first run, and reuses cached runs only when their
configuration hash and exclusion-criteria version still match. If you edit the
plan, use `make study-amend`; it records an amendment and keeps the superseded
seal.

Outputs go to `outputs/campaigns/governance_value/`:

- `study_bundle.json`: everything, including the sealed plan and run IDs
- `digest.md`: a readable summary with diagnostics
- `tables.md`: the manuscript tables, generated rather than transcribed
- `cells.csv`: one row per run

The script exits with status 1 when a blocking diagnostic is raised. Two are
expected in this release: the detector's under-flagging, and the
synthetic-reference calibration in phase 5 of the study script. The manuscript
uses the M5 calibration below instead.

## Real-data calibration (M5)

The M5 series are licensed and are **not** in the repository. Download the
competition files yourself, then:

```bash
make m5 M5_SOURCE=~/Downloads/m5    # writes 30 series to data/demand/
make calibrate                       # weekly cadence, i.i.d. and AR(1) fits
make caltable                        # LaTeX table for the manuscript
```

`data/demand/README.md` explains the aggregation level and the cadence choice.

## Figures

The manuscript figures are produced by three scripts, all reading from the
study bundle:

```bash
.venv/bin/python scripts/render_figures_mdpi.py   --dpi 600   # Figures 2–5
.venv/bin/python scripts/render_network_png.py    --dpi 600   # Figure 1 (network)
.venv/bin/python scripts/render_operator_figure.py --dpi 600  # Figure 6 (operators)
```

PNGs land in `outputs/campaigns/governance_value/figures/print/`. Copy them to
the `Definitions/` folder of the LaTeX project. `make figures` still exists but
calls the original renderer, which predates the editorial fixes; use the three
scripts above for the paper.

## Repository layout

```
src/xai_gov/
  simulation/     the twin: network, demand regimes, disruptions, engine
  policies/       heuristic, robust DRO, random, tabular Q-learning
  conformal/      scores, calibrators, adaptive level, martingales, detector
  governance/     belief filter, threshold solver, budget, CPOMDP mediator
  verification/   PCTL specification, model checker, shield, monitors
  causal/         structural model, recourse, fidelity audit, descriptors
  analysis/       power, preregistration, campaigns, e-values, sweeps,
                  sensitivity, calibration, frontier, figures
configs/          every experiment, arm and campaign as YAML
scripts/          study, sweeps, M5, calibration and figure entry points
tests/            unit and integration tests; the theorems are assertions here
```

Three rules the code enforces, because each was learned from a defect that
completed silently:

1. **A cache key covers the configuration, not only the criteria.** Otherwise a
   changed arm is compared against a stale copy of itself.
2. **A sign convention is part of the hypothesis.** One-sided tests run on
   absolute coverage error, not the signed error.
3. **A condition that holds in every arm is a diagnostic, not an exclusion.**
   Excluding on it would delete the study instead of protecting it.

## Use of generative AI

Claude Opus 4 (Anthropic) was used for code development and debugging,
formatting of results, and drafting and editing of text. The authors reviewed
all generated material and are responsible for the content of this repository
and of the paper.

## Citation

If you use the code, please cite the archived release:

> Ortiz, D.; Machucho, R.; Hernandez, J.; Garcia, J.; Herrera, H.; Lopez, J.
> *xai-gov study bundle v1.1.0* [Data set and software]. Zenodo, 2026.
> https://doi.org/10.5281/zenodo.23484768

## License

See `LICENSE`. The M5 data remain under their original competition terms.
