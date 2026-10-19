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
