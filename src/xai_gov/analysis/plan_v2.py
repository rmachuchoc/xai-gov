"""Checks a v2 plan must pass before it is sealed, and before it runs.

Sealing fixes the plan by hash; these checks make sure what is fixed is what the
revision committed to. Each failure names the commitment it breaks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xai_gov.analysis.agrapa import familywise_threshold
from xai_gov.core.hashing import content_hash

if TYPE_CHECKING:
    from xai_gov.analysis.preregistration import Preregistration


@dataclass(frozen=True, slots=True)
class PlanCheck:
    name: str
    passed: bool
    detail: str

    def to_payload(self) -> dict[str, Any]:
        return {"check": self.name, "passed": self.passed, "detail": self.detail}


def _report_hash(path: Path) -> str:
    return content_hash(json.loads(path.read_text(encoding="utf-8")))


def check_plan(
    plan: Preregistration, *, root: Path, horizons: dict[str, int]
) -> list[PlanCheck]:
    """Every commitment of the v2 design, checked against the files it names."""
    design = plan.design or {}
    checks: list[PlanCheck] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(PlanCheck(name, bool(passed), detail))

    add("replicates fixed", plan.replicates is not None, f"replicates = {plan.replicates}")
    add("base seed fixed", plan.base_seed is not None, f"base_seed = {plan.base_seed}")
    used = {int(s) for s in design.get("seeds_not_to_reuse", [])}
    add("base seed is new", plan.base_seed is not None and plan.base_seed not in used,
        f"{plan.base_seed} against seeds already used {sorted(used)}")

    test = design.get("test", {})
    expected = familywise_threshold(hypotheses=len(plan.hypotheses), alpha=plan.alpha)
    add("k matches the hypotheses", int(test.get("hypotheses", -1)) == len(plan.hypotheses),
        f"declared k = {test.get('hypotheses')}, hypotheses listed = {len(plan.hypotheses)}")
    add("threshold is 2k/alpha", abs(float(test.get("familywise_threshold", -1)) - expected)
        < 1e-9, f"declared {test.get('familywise_threshold')}, 2k/alpha = {expected}")

    for key in ("pilot", "power"):
        entry = design.get(key, {})
        path = root / str(entry.get("report", ""))
        if not path.is_file():
            add(f"{key} report present", False, f"{path} not found")
            continue
        actual = _report_hash(path)
        add(f"{key} report unchanged", actual == entry.get("report_hash"),
            f"{path.name}: sealed {str(entry.get('report_hash'))[:16]}, file {actual[:16]}")

    power_path = root / str(design.get("power", {}).get("report", ""))
    if power_path.is_file():
        power = json.loads(power_path.read_text(encoding="utf-8"))
        required = power["plan"]["replicates"]
        add("replicates meet the simulated power",
            required is not None and plan.replicates is not None
            and plan.replicates >= required,
            f"plan {plan.replicates} against required {required}")
        add("power simulated for this effect",
            abs(float(power["effect"]) - float(design.get("effect", -1))) < 1e-12,
            f"power effect {power['effect']}, plan effect {design.get('effect')}")
        simulated = {h["hypothesis"] for h in power["hypotheses"]}
        listed = {h.hypothesis_id for h in plan.hypotheses}
        add("power covers every hypothesis", simulated == listed,
            f"simulated {sorted(simulated)}, listed {sorted(listed)}")
        pilot = json.loads((root / design["pilot"]["report"]).read_text(encoding="utf-8"))
        add("pilot seed is not the campaign seed",
            int(pilot["base_seed"]) != plan.base_seed,
            f"pilot {pilot['base_seed']}, campaign {plan.base_seed}")

    distinct = sorted(set(horizons.values()))
    add("one horizon for every arm", len(distinct) == 1, f"horizons {distinct}")
    missing = sorted(set(plan.arms()) - set(horizons))
    add("every hypothesized arm has a file", not missing, f"missing {missing}")
    rules = design.get("decision_rules", [])
    add("decision rules declared", len(rules) >= 4, f"{len(rules)} rules")
    return checks


def failures(checks: list[PlanCheck]) -> list[PlanCheck]:
    return [check for check in checks if not check.passed]


def decision_scenario(hypotheses: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the sealed decision rules to H1 and H3.

    H1 is directional; H3 is two-sided, so its direction is read from the sign
    of the mean difference (projection minus backup) only once H3 is supported.
    """
    by_id = {str(h["id"]): h for h in hypotheses}
    h1, h3 = by_id.get("H1"), by_id.get("H3")
    if h1 is None or h3 is None:
        raise ValueError("the decision rules need H1 and H3")
    if not h1["supported"]:
        return {"rule": 4, "when": "H1 not supported",
                "claim": "the v1 effect was an artefact of the engine defects",
                "title": "decided with the coauthors"}
    if not h3["supported"]:
        return {"rule": 1, "when": "H1 supported and H3 shows no difference",
                "claim": (
                    "what matters is that substitution is admissible, not that it is minimal"
                ),
                "title": "Admissible Substitution, Not Projection…"}
    if float(h3["mean_difference"]) > 0.0:
        return {"rule": 2, "when": "H1 supported and H3 favours projection",
                "claim": "minimal correction adds value beyond admissibility",
                "title": "kept"}
    return {"rule": 3, "when": "H1 supported and H3 favours the backup fallback",
            "claim": "the general lesson is to avoid the inadmissible fallback",
            "title": "reformulated"}
