#!/usr/bin/env python
"""Essential simulation parameters of v2, generated from the resolved configs (R2.7).

    python scripts/parameter_table_v2.py

Writes specs/v2/parameters_v2.md and specs/v2/parameters_v2.json. Every value is
read from the configuration each arm actually resolves to, or from the constant
in code that the simulation uses, so the table cannot drift from what ran.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xai_gov.analysis.preregistration import load_seal
from xai_gov.core.hashing import canonical_json
from xai_gov.core.paths import Paths
from xai_gov.core.settings import load_settings
from xai_gov.io.yaml_loader import load_config
from xai_gov.kpis.layers import STARTUP_PERIODS
from xai_gov.orchestration.runner import plan_experiment
from xai_gov.verification.shield import Shield

EXTRA = {"q_ungoverned": "configs/experiments/q_ungoverned.yaml",
         "five_ungoverned": "configs/experiments/five_ungoverned.yaml"}
PREDICATES = {
    "stockout": "inventory <= 0",
    "backlog_critical": "backlog >= 50",
    "inventory_critical": "inventory <= 5",
    "capacity_exhausted": "capacity <= 0",
}


def _row(cells: list[Any]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    return [_row(header), _row(["---"] * len(header)), *(_row(r) for r in rows), ""]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    settings = load_settings(Paths(root=root))
    campaign = load_config(root / "configs/campaigns/confirmatory_v2.yaml", root=root)
    plan, sealed_hash = load_seal(root / "specs/v2/confirmatory_v2.seal.json")
    paths = {str(a): str(p) for a, p in campaign["arms"].items()} | EXTRA
    configs = {arm: plan_experiment(root / path, settings=settings).config
               for arm, path in paths.items()}

    out: list[str] = [
        "# Essential parameters of the v2 simulation", "",
        f"Generated from the resolved configurations; plan seal `{sealed_hash[:16]}`.", "",
    ]
    data: dict[str, Any] = {"seal": sealed_hash}

    design = plan["design"]
    out += ["## Design", ""]
    design_rows = [
        ["horizon (periods)", configs["projection"]["experiment"]["periods"]],
        ["replicates per arm", plan["replicates"]],
        ["base seed", plan["base_seed"]],
        ["confirmatory hypotheses k", design["test"]["hypotheses"]],
        ["familywise threshold 2k/alpha", design["test"]["familywise_threshold"]],
        ["declared effect on beta", design["effect"]],
        ["target power", design["target_power"]],
        ["start-up excluded in H6 (periods)", STARTUP_PERIODS],
    ]
    out += _table(["parameter", "value"], design_rows)
    data["design"] = dict(design_rows)

    out += ["## Arms", ""]
    arm_rows = []
    for arm, config in configs.items():
        verification = config.get("verification") or {}
        shield = verification.get("shield_mode", "projection") if verification.get("shield") \
            else "none"
        governance = (config.get("governance") or {}).get("architecture", "none")
        conformal = (config.get("conformal") or {})
        scale = ((conformal.get("forecaster") or {}).get("scale_mode") or "adaptive") \
            if conformal else "—"
        arm_rows.append([arm, config["network"]["name"], config["policy"]["name"], governance,
                         shield, scale])
    out += _table(["arm", "network", "policy", "governance", "shield operator",
                   "detector scale"], arm_rows)
    data["arms"] = arm_rows

    for network in ("three_echelon", "five_echelon"):
        config = next(c for c in configs.values() if c["network"]["name"] == network)
        defaults = config["network"].get("node_defaults", {})
        out += [f"## Network: {network}", ""]
        rows = []
        for node in config["network"]["nodes"]:
            merged = defaults | node
            rows.append([merged["node_id"], merged["echelon"], merged["supplier_id"],
                         merged["initial_inventory"], merged["capacity"],
                         merged["reorder_point"], merged["order_quantity"],
                         merged["reorder_point"] + merged["order_quantity"],
                         merged["lead_time_mean"], merged.get("lead_time_dispersion", 0.0),
                         merged.get("holding_cost", 1.0), merged.get("backlog_cost", 5.0),
                         merged.get("order_cost", 0.5)])
        out += _table(["node", "echelon", "supplier", "initial inventory", "capacity", "s",
                       "Q", "S = s+Q (F-backup)", "lead time mean", "lead time dispersion",
                       "holding", "backlog", "order"], rows)
        data[network] = rows

    sections = {
        "Demand (high volatility)": configs["projection"]["demand"],
        "Policy: opaque DRO": configs["projection"]["policy"],
        "Policy: tabular Q-learning": configs["q_projection"]["policy"],
        "Governance (cpomdp_h2)": configs["projection"]["governance"],
        "Conformal detector": configs["projection"]["conformal"],
    }
    for title, section in sections.items():
        out += [f"## {title}", "", "```json", json.dumps(section, indent=2, sort_keys=True),
                "```", ""]
        data[title] = section

    verification = configs["projection"]["verification"]
    shield = Shield()
    out += ["## Safety specification and envelope", ""]
    properties = verification["specification"]["properties"]
    out += _table(["id", "operator", "predicate", "bad state", "bound", "horizon"],
                  [[p["id"], p["operator"], p["predicate"], PREDICATES[p["predicate"]],
                    p["bound"], p["horizon"]] for p in properties])
    envelope_rows = [["horizon_lookahead", verification.get("horizon_lookahead", 1)],
                     ["max_order (upper cap besides capacity)", shield.max_order],
                     ["claim", design["envelope"]["claim"]],
                     ["interval", design["envelope"]["interval"]],
                     ["unit", design["envelope"]["unit"]]]
    out += _table(["envelope", "value"], envelope_rows)
    data["specification"] = properties
    data["envelope"] = dict(envelope_rows)

    (root / "specs/v2/parameters_v2.md").write_text("\n".join(out), encoding="utf-8")
    (root / "specs/v2/parameters_v2.json").write_text(canonical_json(data) + "\n",
                                                       encoding="utf-8")
    print("  -> specs/v2/parameters_v2.md, specs/v2/parameters_v2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
