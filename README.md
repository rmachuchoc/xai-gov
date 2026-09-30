# XAI-Gov

Agent-governed digital twin for supply chains: causal explainable AI,
adaptive conformal inference, a constrained-POMDP governance agent under a
verified safety shield, and an auditable traceability contract.

Reference document: *The XAI-Gov Project — research protocol*.

## Requirements

Ubuntu 24.04 LTS, Python 3.12, free software only. The virtual environment
is expected at `.venv` inside the project root.

## Install

```bash
git clone https://github.com/rmachuchoc/xai-gov.git
cd xai-gov
python3.12 -m venv .venv
source .venv/bin/activate
make install          # dev dependencies + editable install
make check            # ruff, mypy, pytest
make doctor           # audit the environment for reproducibility risks
```

Always run tests through `make test`, never bare `pytest`. Pytest
autoloads every plugin registered on the system, and a test result that
depends on unrelated system packages is not reproducible; `make test` sets
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` so collection sees only this project.
On a machine with ROS on `PYTHONPATH`, bare `pytest` fails outright — its
`launch_testing` plugin imports `lark`. `make doctor` reports that and
similar hazards.

If the project lives elsewhere, export `XAI_GOV_ROOT`; nothing in the
code assumes a hard-coded path.

## Current state — complete

All six stages are in place. Runs execute end to end, leave verifiable
artifacts, and can be governed by a constrained-POMDP agent whose
intervention threshold is *derived*, whose explanations are *audited*, and
whose actions pass a shield *synthesized from a formal specification*.

```bash
xai-gov info                                  # resolved paths and settings
xai-gov factors                               # available experimental factors
xai-gov run --experiment configs/experiments/high_vol_delegated.yaml
xai-gov kpis <run-name>                       # the six indicator layers
xai-gov runs                                  # list runs with chain status
xai-gov verify-log <run-name>                 # verify a decision log
```

Or `make run EXP=configs/experiments/high_vol_shielded.yaml`.

**What exists.** A multi-echelon twin (retail → distribution → plant) with
stochastic lead times, an in-transit pipeline, backlog propagation upstream,
capacity limits and four disruption mechanisms. Five demand regimes, three of
which deliberately break exchangeability and declare where. Three policy
families: the classical `heuristic_sQ`, an `opaque_dro` robust policy, and
`random_bounded` as the falsification arm.

The conformal layer: studentized nonconformity over a causal EWMA forecast,
recency-weighted calibration with a reported effective sample size, adaptive
conformal inference (ACI) with a clipped level and saturation diagnostics, a
mixture power martingale for anytime-valid change detection, and optional
conformal risk control calibrated against a safety loss rather than coverage.
The classical split scheme is kept as the control arm for RQ6.

The governance layer: a belief filter over {nominal, drift, disruption}; the
intervention threshold b\* solved by value iteration on the collapsed belief
(with the closed-form myopic threshold computed alongside it as a rail); a
discounted budget constraint priced by a Lagrange multiplier that reports
κ\*, the shadow price of oversight capacity; and H0–H3 modelled as
allocations of decision rights — differing in *when* the organization
exercises what it retained, not merely in how much it grants.

The causal layer: a structural causal model whose mechanisms are the
simulator's own — a test fails if the two drift apart, because a silent
divergence would leave every downstream number well formed while describing a
mechanism the twin does not run. Recourse is filtered to levers that are
admissible, identifiable and affordable, and every descriptor arm passes
through the same explanatory-fidelity audit, including the post-hoc one, which
is implemented in good faith rather than as a straw man.

The verification layer: safety properties in probabilistic temporal logic,
exact reachability checking on a declared finite abstraction, and a shield
*synthesized* from those properties — so adding a property changes the shield
instead of requiring someone to hand-edit a safe mode that then drifts from the
specification. The shield can raise a quantity governance zeroed: cancelling a
replenishment during a stockout is itself a way to violate a property, and an
agent whose only tool is refusal will do it. Runtime monitors report when the
concrete run leaves the model the properties were verified on, because a run
that advertises verified properties after diverging from that model is making a
claim it cannot support.

The oversight layer: escalation as learned delegation against a simulated
supervisor with finite attention, regime-dependent competence and fatigue —
better than the system under disruption, worse under nominal conditions, which
is what makes the allocation non-trivial. A complementarity ledger scores the
team, the human alone and the system alone on every decision, so RQ7 can come
out negative.

The analysis layer: preregistered plans sealed by hash, so an analysis that
does not match its seal is reported as exploratory rather than confirmatory.
Comparisons use e-values and betting martingales, which means a campaign may be
stopped when the compute runs out or extended later without correction — the
fixed-sample statistics the field normally uses forbid exactly the thing
simulation research must do. Sobol indices separate findings about governance
from findings about a declared cost. Calibration against public data yields a
posterior rather than a point estimate, and the sim-to-real gap is measured on
statistics held out of the fit. The Pareto frontier is the deliverable Theorem 3
implies: no arm is best on coverage, budget and return at once, so the artifact
is the set of informed choices and their exchange rates.

```bash
xai-gov seal configs/campaigns/smoke.yaml       # commit the plan first
xai-gov campaign configs/campaigns/smoke.yaml --replicates 5

make study-quick                                # the whole pipeline, fast
make study                                      # the committed study, n = 30
```

`make study` runs the five phases in the order the protocol requires: the
power check first (a campaign that cannot detect the effect it seeks should not
consume a week of compute, so an under-powered study is refused unless
overridden), then the seal, then the campaign, then sensitivity, then
calibration. It writes a study bundle carrying the sealed plan, the per-cell
run identifiers, the evidence for each hypothesis, the frontier, the Sobol
indices and the sim-to-real gap — the object to hand a reviewer.

**What does not exist, and says so.** The `marl_heterogeneous` and `llm_agent`
policy families and the `multi_role` / `federated` governance architectures
fail with a message naming what would be required. Indicator layers write
`{"status": "not_available", "requires": …}` rather than a silent zero — a KPI
file with a fabricated zero corrupts a meta-analysis; one with a stated gap is
merely incomplete.

## Layout

```
configs/          declarative configuration, composed via _include_
  app/            paths, logging, runtime, tracking
  network/        topologies: single_echelon, three_echelon
  scenarios/      demand regimes and disruption schedules
  policies/       one file per policy family
  conformal/      detector schemes: aci_adaptive, split_classical, risk_controlled
  explainability/ descriptor arms: none, post_hoc, causal
  verification/   safety properties and the shield: shield_on, shield_off
  oversight/      deferral policies: none, threshold, learned
  campaigns/      preregistered analysis plans
  analysis/       sensitivity parameter ranges
  governance/     architectures, autonomy regimes, economics and budgets
  experiments/    composed runs; each declares its own master_seed
data/             raw, interim, processed (DVC-tracked)
outputs/runs/     one directory per run; never edited by hand
specs/            PCTL properties and synthesized shields (stage 5)
scripts/          doctor.py (environment audit), retire_legacy.sh
src/xai_gov/      the package, one subpackage per protocol layer
  core/ io/       determinism, hashing, config, decision log, run artifacts
  simulation/     network, state, demand, disruptions, engine
  policies/       base, families, registry
  conformal/      scores, calibrator, adaptive, martingale, risk_control, detector
  causal/         scm, recourse, fidelity, descriptor
  verification/   specification, model_checker, shield, monitors
  oversight/      supervisor, delegation
  analysis/       preregistration, campaign, anytime, sensitivity, calibration,
                  frontier, report
  governance/     agent, belief, thresholds, budget, autonomy, cpomdp, registry
  kpis/           the six indicator layers
  orchestration/  experiment composition and execution
tests/            unit and integration tests mirroring src/
```

## Contracts

Four invariants hold across the codebase and are covered by tests:

1. **Determinism.** One master seed per run; every component derives its
   own stream by label, so adding a component never shifts an existing
   one's random draws.
2. **Traceability.** No decision executes without a sealed record. The log
   is append-only and hash-chained; a run whose chain fails verification
   never receives a manifest.
3. **Configuration outside the code.** No module reads YAML on its own and
   no path is built by string concatenation.
4. **Honest failure.** A contract violation aborts the run and says which
   contract broke. A component that does not exist yet names the stage that
   will deliver it. Nothing degrades silently and nothing reports a
   fabricated value.
5. **Proposal, mediation, enforcement are separate authorities.** A policy
   proposes and can never execute; only governance may alter a proposal;
   only the shield may override governance. The decision record shows all
   three, so responsibility for any executed action is attributable. The
   layering is enforced by tests that import each module first in a cold
   interpreter — an import cycle here would mean the simulation reaching into
   the authorities meant to constrain it.
6. **Decision boundaries are derived, not chosen.** No threshold in the
   governance path is read from a configuration file. b\* comes from value
   iteration on declared economics, so changing the cost of an intervention
   moves the boundary automatically and in the direction Theorem 2 predicts —
   a claim the test suite checks rather than asserts.

## Roadmap

| Stage | Content |
| --- | --- |
| 1 | Reproducibility base and traceability contract *(delivered)* |
| 2 | Multi-echelon twin, policy families, experiment runner, KPI layers *(delivered)* |
| 3 | Adaptive conformal inference, test martingales, governance CPOMDP over beliefs *(delivered)* |
| 4 | Structural causal model and causal recourse, explanatory fidelity *(delivered)* |
| 5 | PCTL specifications, shield synthesis, runtime monitors, learned delegation *(delivered)* |
| 6 | Campaigns, anytime-valid statistics, simulation-based calibration, benchmark *(delivered)* |

## License

MIT.
