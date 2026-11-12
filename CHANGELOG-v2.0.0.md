# v2.0.0

This release rebuilds the simulator and reruns the study from scratch. A reviewer
of the first submission reproduced the released code and found two defects in
the period step, and both were confirmed. Every absolute number published with
v1.x is therefore unreliable, and v2.0.0 should be used in its place. The v1.x
tags and their Zenodo bundles stay online as a record of what was submitted.

## The two defects

- **Service level above one.** The engine added up the units served by every
  node, suppliers included, and divided by retail demand, so supplier backlog
  clearance was counted as customer service and some arms reported rates above
  11.
- **Lost backorders.** When a supplier could not fill an order, it took the
  units out of its own inventory once at order time and again when it later
  served the backlog, and never shipped anything to the node that had ordered.
  An eight-unit backorder simply disappeared.

The tests that confirmed both defects (`tests/integration/test_material_flow.py`)
were committed failing, before the engine changed, and pass now.

## Changed

- **Engine.** The period step is now the standard serial model with
  backorders. Each node receives what is due, observes its demand (customer
  demand at the retailer, last period's order from the node below elsewhere),
  ships what it can into the downstream pipeline and keeps the rest on
  backorder, and only then places its own order. Ordering no longer moves
  material. Conservation of material is asserted in every period.
- **Primary outcome.** The retail fill rate β, the share of customer demand
  served from stock in the period it arrives, replaces the service level. It
  lies in [0, 1] by construction. α (periods without a shortfall) and the mean
  customer backlog are secondary.
- **One horizon.** Every arm runs 60 periods on the same demand paths; the
  unsupervised baseline used to run 40.
- **Envelope.** It is documented as an operating approximation derived from
  the specification, not a verified PCTL guarantee. Each run logs, per
  property, whether it was violated, and the campaign reports exact
  Clopper–Pearson intervals against the bounds 0.05, 0.20 and 0.02, along with
  how often the envelope was infeasible and why.
- **Inference.** The fixed betting fractions gave way to the predictable aGRAPA
  bet (Waudby-Smith and Ramdas, 2024), computed from earlier replicates only and
  clipped to [0, 1/2]. Confidence sequences now come from inverting the same
  two-sided e-process, so they hold at every replicate and agree with the tests
  by construction. The empirical-Bernstein interval is gone.
- **Detector.** Coverage error is measured against the fixed target ε = 0.10,
  not the adapted level, and the trajectory of εₜ is recorded. The conformal
  martingale no longer claims a δ-level false-alarm guarantee; its false-alarm
  rate is measured on stationary runs instead.
- **Theorem A1.** The comparative statics of the optimal threshold now rest on
  explicit conditions on the continuation values, checked numerically on the
  sweep grid (`scripts/verify_theorem_a1.py`).

## Added

- **F-backup**, an admissible fallback: a base-stock controller that sees what
  the base policy sees, with its level fixed in the network configuration and
  its order clipped to the envelope (`configs/verification/shield_backup.yaml`).
  The confirmatory campaign compares it with projection in the arm `backup`;
  `q_backup` and `five_q_backup` configure it over the learned policy and the
  five-echelon chain.
- **A sealed v2 plan** (`specs/v2/confirmatory_v2.seal.json`): six confirmatory
  hypotheses, H1–H6, with an effect of Δβ = 0.05, threshold 2k/α = 240 and base
  seed 20261107, shared by no earlier campaign. The number of replicates, 500,
  comes from simulating the power of the exact procedure on a 30-replicate
  pilot (`scripts/power_v2.py`). The v1.1.0 campaign is treated only as a pilot,
  and none of its cells enters a v2 test.
- **Exploratory campaigns** on the new engine for the descriptor comparison and
  the detector ablations (former RQ3 and RQ6–RQ6d), a stationary null line of
  200 runs per detector, and the break-even and severity sweeps with paired
  bootstrap intervals.
- **Records for the revision**: admissibility over every decision, governance
  rates with their denominators, εₜ clipping and saturation per arm, the
  parameter table generated from the resolved configurations, worked numeric
  examples, and a dated chronology of the v1 amendments and the v2 plan
  (`specs/v2/CHRONOLOGY.md`).
- **Figures for v2** (`make figures-v2`), each with the CSV it was drawn from.

## Results in brief

All six confirmatory hypotheses were supported. Projection beats the null
fallback under both policies, on five echelons and after excluding the start-up
periods, but the admissible fallback beats projection by 0.127 in β (H3). Under
the decision rules sealed before the campaign, the claim becomes that what
matters is that the replacement is admissible, not that the correction is
minimal.

## No longer reported

- The strategic indicator TVC. Its two terms are in different units, so its
  size, and for some arms its sign, depend on the unit chosen for the
  intervention cost. The code still computes it so that v1 bundles can be
  read, but the v2 analysis and the paper rely on the net value and the return
  on governed decisions instead.
