# Chronology of the preregistration (review make-4658289, Reviewer 2, point 4)

This file records, in order, what was fixed and what was seen, so a reader can
tell prospectively specified tests from analyses informed by earlier data. Every
entry can be checked against a file or a commit named beside it. Times are UTC
unless marked otherwise; commit times are local (UTC−6).

## 1. v1 (the campaign reported in the submitted manuscript; now the pilot)

Plan: `configs/campaigns/governance_value.yaml`, sealed in
`outputs/campaigns/governance_value.seal.json`. The seal holds the first plan
and eight amendments; each superseded plan is kept with the time it was
replaced.

| Plan | Hash (12) | Replaced at | Change |
| --- | --- | --- | --- |
| 0 | `4f0eaea2f64b` | 2026-08-21 17:17 | First seal: RQ3, RQ4, RQ6, RQ8 |
| 1 | `8611752620bc` | 2026-08-21 17:22 | RQ6 control and rationale; RQ3 treatment, control, rationale; notes |
| 2 | `9530a5c25304` | 2026-08-24 20:16 | RQ3 treatment and control |
| 3 | `b4b8918fca36` | 2026-08-24 20:28 | RQ6 indicator and rationale |
| 4 | `c112af71e1cb` | 2026-08-24 21:03 | RQ6b added; RQ6 treatment and rationale |
| 5 | `76e61767268c` | 2026-08-24 21:10 | RQ6c added |
| 6 | `c9a11c9851ac` | 2026-09-14 21:40 | RQ6d added |
| 7 | `d53997563077` | 2026-10-19 20:29 | RQ9 added |
| 8 | `cf7e327cfc73` | (current) | RQ10, RQ11 added (commit `d4192e9`, v1.1.0) |

What was informed by earlier results, by the plan's own rationales:

- **RQ6** (from plan 4): its treatment was changed to the reference-scale arm
  "because the first campaign found the co-adapting score normalizing the
  shift away".
- **RQ6b**: "added after the first campaign measured a coverage error of
  −0.158".
- **RQ6c**: "added after RQ6b refuted the self-normalization account".
- **RQ6d**: motivated by "the detector payload of the first campaign".
- RQ3, RQ4, RQ8, RQ9, RQ10, RQ11 cite no earlier result; RQ10 and RQ11 were
  added together with the arms they compare (v1.1.0).

Reused cells: the final v1 report
(`outputs/campaigns/governance_value/campaign_report.json`, generated
2026-10-22 00:40) reused **all 800 cells** from earlier runs
(`campaign_summary.reused_from_previous_run = 800`). The cells of RQ10 and
RQ11 were therefore generated before those hypotheses were sealed, under the
same seeds as every other arm.

Consequence for v2: RQ6–RQ6d are exploratory (decision 5), and the v1 campaign
is treated as a pilot. No v1 cell enters any v2 test.

## 2. v2 (this revision)

### 2.1 Fixed before any v2 code ran

`/home/ruben/Documents/CLAUDE.md` (last written 2026-11-04 16:39 local), from
the review letter (`revision`, 16:03) and the revision plan
(`Plan de revision MAKE - detallado.pdf`, 16:21): the engine model, beta as
the primary outcome, the envelope reported as a specification-derived
approximation with bounds 0.05 / 0.20 / 0.02, F-backup as the admissible
fallback, aGRAPA with lambda in [0, 1/2] and a fixed scale, power ≥ 0.80 at
delta beta = 0.05, **the six confirmatory hypotheses H1–H6 (which arms, H3
two-sided)**, the threshold 2k/alpha = 240, a new seed with no reused cell,
the withdrawal of the delta guarantee, coverage against 0.10, and **the four
decision rules** that map H1 and H3 to the article's claim and title.

### 2.2 Built and seen before the seal (2026-11-04, local time)

| When | What | Seed | What was visible |
| --- | --- | --- | --- |
| 16:56 | Failing tests confirm R1.1, R1.2 on the v1 engine (`ac81d00`) | — | v1 defects only |
| 17:01 | New engine (`537eef7`); sanity probe of 2 stochastic chains | 7, 42 | beta of test chains, no campaign arm |
| 17:10 | Smoke run of 9 shield arms, **one run each** (scratchpad) | 20260703 (the arms' own v1 seed) | beta: F-null 0.06, projection 0.69–0.76, F-backup 0.80–0.81 |
| 17:18 | Envelope smoke run of 4 arms, one run each | 20260703 | violation frequencies, infeasibility |
| 17:22 | Null-line smoke, 10 runs per detector | 20261104 | 0/10 false alarms |
| — | **H4 two-sided and H6 with w = 27 decided** (asked before the pilot) | — | — |
| 17:35 | Campaign v2 file with H1–H6 (`8be582d`); aGRAPA (`20b8efb`) | — | — |
| 17:36 | Pilot smoke, 2 replicates; pilot, **30 replicates** | 20261105 | means and sds of all six paired differences |
| 17:38 | Power simulation from the pilot (`270e6e7`) | 20261106 | required n per hypothesis (pilot means removed by centring) |
| — | **n = 500, base seed 20261107, the replicate as the envelope's unit decided** | — | — |
| 17:46 | Plan checks pass; **seal** (`9d050de`, sealed 23:46:03 UTC) | — | — |

The decisions taken after something was visible are n, the seed and the
envelope's interval unit (after the pilot), and H4's direction and H6's window
(after the one-run smoke of the shield arms, which included neither arm of H4).
None of them selects an outcome: n was chosen from the simulated power at the
pilot's dispersion, the seed is new, and the unit makes the interval exact. The
hypotheses' arms, indicators and H1–H3 directions, and the decision rules, were
fixed in §2.1 before anything was run.

Seeds used before the seal and never reused by the confirmatory campaign:
20260703 (v1 and smoke runs), 20261104 (null line), 20261105 (pilot), 20261106
(power). The seal records them in `design.seeds_not_to_reuse`, and
`analysis/plan_v2.check_plan` refuses a plan whose base seed is among them.

### 2.3 After the seal

| When (UTC) | What | Status |
| --- | --- | --- |
| 23:58:03 → 00:00:23 | Confirmatory campaign, 9 arms × 500, seed 20261107 | confirmatory; plan matches its seal; **0 cells reused** (`reused_from_previous_run = 0`) |
| 00:01:51 → 00:02:03 | Null line, 200 runs per detector, seed 20261104 | descriptive (decision 6) |
| 00:05:50 → 00:11:36 | Sweeps with bootstrap intervals, 100 per level, seed 20261108 | exploratory |
| 00:23:02 → | Post-hoc economics with matched baselines; q_ungoverned and five_ungoverned run at the campaign seed | post hoc, outside the sealed plan |
| 00:24:58 → 00:26:15 | RQ6–RQ6d re-run, 500 per arm, seed 20261107 | exploratory, unsealed |
| 00:29:01 | Theorem A1 numerical check | analytical |
| → 00:35:37 | Admissibility and eps_t clipping summaries from the stored cells | post hoc |

Nothing after the seal changed the plan; `tests/unit/test_plan_v2.py` fails if
the plan in the repository departs from its seal.

## 3. Conditions behind the familywise control

- k = 6 confirmatory hypotheses; each may be claimed in either direction, so
  the union bound over 2k e-values gives the threshold 2k/alpha = 240.
- Each e-value is the predictable aGRAPA wealth on paired differences
  (`analysis/agrapa.py`), a nonnegative supermartingale under its null for
  i.i.d. bounded differences. Rejection at the first crossing is valid by
  Ville's inequality at any stopping time.
- Replicates are independent: replicate r draws its seed from the base seed and
  r alone (`analysis/campaign.plan_campaign`), and arms share it, so each
  difference is within one replicate and the differences are i.i.d. across
  replicates. All arms are deterministic given the seed.
- The scale is the declared range of beta's differences, [−1, 1]; a difference
  outside it would be refused, not clipped.
