# v1.1.0

Adds a learned policy and a true fallback shield so the central comparison can
be stated as fallback versus projection over the same specification, and tested
over a policy that adapts to the mediation it runs under.

## Added

- `QLearningPolicy` (`qlearning_tabular`): tabular Q-learning over
  inventory-position and backlog bands, pre-trained on a single-node surrogate
  without oversight and learned online in the twin. Deterministic from its seed.
- `Shield(mode="fallback")`: the same admissible envelope as the projection
  shield, but an inadmissible action is replaced by the null action instead of
  projected. `fallback_inadmissible` counts activations that leave the action
  outside the envelope. Configured with `verification.shield_mode`.
- Four arms: `fallback_shield` (opaque DRO), `q_ungoverned`, `q_fallback`,
  `q_projection`.
- Two preregistered hypotheses by amendment: RQ10 (projection vs fallback,
  fixed policy) and RQ11 (projection vs fallback, learned policy). The
  familywise threshold rises from 160 to 200.

## Unchanged

- Projection remains the default operator; every v1.0.1 arm, seed and result is
  reproduced exactly. The 640 confirmatory cells are reused on resume because
  their resolved configuration hashes are unchanged.

## Analysis corrections (folded into v1.1.0 before release)
- E-values and confidence sequences used a scale taken from the full sample
  (max |d_i|), so each bet depended on observations not yet made and the
  process was not a martingale. The scale is now the constant 1: every
  indicator lies in [0, 1], so paired differences already lie in [-1, 1].
- The familywise threshold counts two directional tests per hypothesis:
  2k/alpha = 400 for k = 10.
- Supported: RQ8, RQ9, RQ10, RQ11. Refuted: RQ4. Inconclusive: RQ3, RQ6,
  RQ6b, RQ6c, RQ6d. No simulation was re-run; only the analysis was
  recomputed over the same 800 cells.
- Regression tests fail if a data-dependent scale is reintroduced.
