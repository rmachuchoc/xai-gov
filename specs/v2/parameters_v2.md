# Essential parameters of the v2 simulation

Generated from the resolved configurations; plan seal `3376ba1d76f2633d`.

## Design

| parameter | value |
| --- | --- |
| horizon (periods) | 60 |
| replicates per arm | 500 |
| base seed | 20261107 |
| confirmatory hypotheses k | 6 |
| familywise threshold 2k/alpha | 240 |
| declared effect on beta | 0.05 |
| target power | 0.8 |
| start-up excluded in H6 (periods) | 27 |

## Arms

| arm | network | policy | governance | shield operator | detector scale |
| --- | --- | --- | --- | --- | --- |
| projection | three_echelon | opaque_dro | cpomdp | projection | adaptive |
| fallback | three_echelon | opaque_dro | cpomdp | fallback | adaptive |
| backup | three_echelon | opaque_dro | cpomdp | backup | adaptive |
| q_projection | three_echelon | qlearning_tabular | cpomdp | projection | adaptive |
| q_fallback | three_echelon | qlearning_tabular | cpomdp | fallback | adaptive |
| governed | three_echelon | opaque_dro | cpomdp | none | adaptive |
| ungoverned | three_echelon | opaque_dro | no_governance | none | — |
| five_projection | five_echelon | opaque_dro | cpomdp | projection | adaptive |
| five_fallback | five_echelon | opaque_dro | cpomdp | fallback | adaptive |
| q_ungoverned | three_echelon | qlearning_tabular | no_governance | none | — |
| five_ungoverned | five_echelon | opaque_dro | no_governance | none | — |

## Network: three_echelon

| node | echelon | supplier | initial inventory | capacity | s | Q | S = s+Q (F-backup) | lead time mean | lead time dispersion | holding | backlog | order |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| retail-01 | 0 | dc-01 | 45.0 | 120.0 | 25.0 | 35.0 | 60.0 | 2.0 | 0.25 | 1.0 | 5.0 | 0.5 |
| dc-01 | 1 | plant-01 | 90.0 | 300.0 | 60.0 | 90.0 | 150.0 | 3.0 | 0.25 | 0.6 | 5.0 | 0.5 |
| plant-01 | 2 | __external__ | 180.0 | 600.0 | 120.0 | 200.0 | 320.0 | 4.0 | 0.25 | 0.3 | 5.0 | 0.5 |

## Network: five_echelon

| node | echelon | supplier | initial inventory | capacity | s | Q | S = s+Q (F-backup) | lead time mean | lead time dispersion | holding | backlog | order |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| retail-01 | 0 | regional-01 | 45.0 | 120.0 | 25.0 | 35.0 | 60.0 | 2.0 | 0.25 | 1.0 | 5.0 | 0.5 |
| regional-01 | 1 | dc-01 | 70.0 | 200.0 | 45.0 | 65.0 | 110.0 | 2.0 | 0.25 | 0.8 | 5.0 | 0.5 |
| dc-01 | 2 | national-01 | 90.0 | 300.0 | 60.0 | 90.0 | 150.0 | 3.0 | 0.25 | 0.6 | 5.0 | 0.5 |
| national-01 | 3 | plant-01 | 130.0 | 450.0 | 90.0 | 140.0 | 230.0 | 3.0 | 0.25 | 0.45 | 5.0 | 0.5 |
| plant-01 | 4 | __external__ | 180.0 | 600.0 | 120.0 | 200.0 | 320.0 | 4.0 | 0.25 | 0.3 | 5.0 | 0.5 |

## Demand (high volatility)

```json
{
  "cycle": 5,
  "dispersion": 0.2,
  "mean": 12.0,
  "regime": "high_volatility",
  "volatility_multiplier": 3.0
}
```

## Policy: opaque DRO

```json
{
  "name": "opaque_dro",
  "params": {
    "ambiguity_radius": 0.25,
    "service_quantile": 0.95,
    "window": 8
  }
}
```

## Policy: tabular Q-learning

```json
{
  "name": "qlearning_tabular",
  "params": {
    "alpha": 0.1,
    "demand_mean": 20.0,
    "demand_sd": 6.0,
    "epsilon": 0.05,
    "gamma": 0.95,
    "online": true,
    "pretrain_episodes": 400,
    "pretrain_epsilon": 0.2,
    "pretrain_periods": 60,
    "seed": 20261019
  }
}
```

## Governance (cpomdp_h2)

```json
{
  "architecture": "cpomdp",
  "autonomy_regime": "H2",
  "budget": {
    "allowance": 20.0,
    "discount": 0.95,
    "dual_rate": 0.05
  },
  "economics": {
    "discount": 0.95,
    "disruption_loss": 10.0,
    "intervention_cost": 1.0
  },
  "levels": {
    "escalation_belief": 0.6,
    "quantity_bound": 60.0,
    "safe_mode_belief": 0.9
  },
  "regime_model": {
    "change_likelihood_ratio": 8.0,
    "emission": {
      "disruption": 0.8,
      "drift": 0.35,
      "nominal": 0.05
    },
    "persistence": 0.9
  }
}
```

## Conformal detector

```json
{
  "burn_in": 12,
  "calibrator": {
    "name": "weighted",
    "params": {
      "decay": 0.97,
      "window": 200
    }
  },
  "forecaster": {
    "alpha": 0.3,
    "beta": 0.1
  },
  "level": {
    "params": {
      "gamma": 0.05,
      "max_level": 0.5,
      "min_level": 0.005,
      "target_level": 0.1
    },
    "scheme": "aci"
  },
  "martingale": {
    "enabled": true,
    "params": {
      "delta": 0.01
    }
  },
  "risk_control": {
    "enabled": false
  },
  "score": "studentized_residual",
  "warmup": 15
}
```

## Safety specification and envelope

| id | operator | predicate | bad state | bound | horizon |
| --- | --- | --- | --- | --- | --- |
| SAFE-1 | persistent | backlog_critical | backlog >= 50 | 0.05 | 3 |
| SAFE-2 | bounded_eventually | inventory_critical | inventory <= 5 | 0.2 | 5 |
| SAFE-3 | persistent | stockout | inventory <= 0 | 0.02 | 2 |

| envelope | value |
| --- | --- |
| horizon_lookahead | 1 |
| max_order (upper cap besides capacity) | 200.0 |
| claim | specification-derived operating approximation, not a PCTL guarantee |
| interval | Clopper-Pearson 95% |
| unit | replicate |
