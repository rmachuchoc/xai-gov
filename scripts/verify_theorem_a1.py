#!/usr/bin/env python
"""Numerical verification of Theorem A1 on the sweep grid (decision 8; R2.6).

    python scripts/verify_theorem_a1.py --output outputs/v2/analysis/theorem_a1.json

Uses the belief dynamics and discount the governed arms declare
(configs/governance/cpomdp_h2.yaml): persistence, the nominal and disruption
emission probabilities, and the economics' discount.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from xai_gov.analysis.theorem_a1 import verify
from xai_gov.core.hashing import canonical_json
from xai_gov.governance.thresholds import BeliefDynamics
from xai_gov.io.yaml_loader import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="verify Theorem A1 numerically")
    parser.add_argument("--governance", default="configs/governance/cpomdp_h2.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    governance = load_config(root / args.governance, root=root)["governance"]
    model = governance["regime_model"]
    dynamics = BeliefDynamics(
        persistence=float(model["persistence"]),
        flag_probability_nominal=float(model["emission"]["nominal"]),
        flag_probability_disruption=float(model["emission"]["disruption"]),
    )
    result = verify(dynamics=dynamics, discount=float(governance["economics"]["discount"]))
    result["source"] = args.governance
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(result) + "\n", encoding="utf-8")

    optimal = result["optimal"]
    print(f"  myopic: cost {result['myopic']['nondecreasing_in_cost']}, "
          f"loss {result['myopic']['nonincreasing_in_loss']}")
    for key in ("C1_single_crossing_everywhere", "C2_holds_everywhere", "C3_holds_everywhere",
                "nondecreasing_in_cost_everywhere", "nonincreasing_in_loss_everywhere"):
        print(f"  {key}: {optimal[key]}")
    for row in optimal["by_loss"]:
        if row["loss"] == 10.0:
            print(f"  L=10: b* over c {row['costs']} = {row['optimal_threshold']}")
    print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
