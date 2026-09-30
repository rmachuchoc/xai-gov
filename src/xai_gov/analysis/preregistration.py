"""Preregistration.

A preregistration is a promise, and this module makes it a machine-checkable
one. Hypotheses, the primary indicator, the comparison arms and the exclusion
criteria are written to a file and sealed with a hash *before* any campaign
runs. The analysis then refuses to proceed against a plan whose hash does not
match the sealed one.

The reason is not suspicion of the researcher. It is that the flexibility to
choose an outcome measure after seeing the data is enough to produce a
significant finding from noise without anyone intending to — and no amount of
care substitutes for making the choice unrevisable.

What the seal permits: adding an analysis, clearly labelled exploratory.
What it forbids: quietly editing the primary hypothesis and reporting the
result as confirmatory. Both paths exist here, and the distinction travels into
the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xai_gov.core.hashing import canonical_json, content_hash


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """One falsifiable claim, with the prediction that could refute it."""

    hypothesis_id: str
    question: str
    indicator: str
    treatment: str
    control: str
    direction: str
    predicted_effect: float
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.direction not in ("greater", "less", "different"):
            raise ValueError(
                f"{self.hypothesis_id}: direction must be 'greater', 'less' or "
                f"'different', not {self.direction!r}"
            )
        if self.treatment == self.control:
            raise ValueError(
                f"{self.hypothesis_id}: treatment and control are the same arm "
                f"({self.treatment!r}); such a comparison cannot fail"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.hypothesis_id,
            "question": self.question,
            "indicator": self.indicator,
            "treatment": self.treatment,
            "control": self.control,
            "direction": self.direction,
            "predicted_effect": self.predicted_effect,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class Preregistration:
    """The sealed analysis plan for a campaign."""

    campaign: str
    hypotheses: tuple[Hypothesis, ...]
    alpha: float = 0.05
    minimum_replicates: int = 30
    exclusion_criteria: tuple[str, ...] = (
        "runs whose decision log fails hash-chain verification",
        "runs whose conformal detector reports saturation",
        "runs that aborted before completing their declared horizon",
    )
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.hypotheses:
            raise ValueError("a preregistration with no hypotheses commits to nothing")
        seen: set[str] = set()
        for hypothesis in self.hypotheses:
            if hypothesis.hypothesis_id in seen:
                raise ValueError(f"duplicate hypothesis id {hypothesis.hypothesis_id!r}")
            seen.add(hypothesis.hypothesis_id)
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if self.minimum_replicates < 2:
            raise ValueError("minimum_replicates must be at least 2")

    @property
    def plan_hash(self) -> str:
        """The seal. Any edit to the plan changes it."""
        return content_hash(self.to_payload())

    def by_id(self, hypothesis_id: str) -> Hypothesis:
        for hypothesis in self.hypotheses:
            if hypothesis.hypothesis_id == hypothesis_id:
                return hypothesis
        raise KeyError(f"no hypothesis {hypothesis_id!r} in this preregistration")

    def arms(self) -> tuple[str, ...]:
        """Every experiment arm the plan refers to."""
        named: set[str] = set()
        for hypothesis in self.hypotheses:
            named.update((hypothesis.treatment, hypothesis.control))
        return tuple(sorted(named))

    def to_payload(self) -> dict[str, Any]:
        return {
            "campaign": self.campaign,
            "alpha": self.alpha,
            "minimum_replicates": self.minimum_replicates,
            "hypotheses": [h.to_payload() for h in self.hypotheses],
            "exclusion_criteria": list(self.exclusion_criteria),
            "notes": self.notes,
        }

    def seal(self, path: Path, *, amend: bool = False) -> str:
        """Write the plan and return its hash.

        Refuses to overwrite unless ``amend`` is set. A preregistration that can
        be silently replaced is not a preregistration, and the failure must be
        loud enough that nobody works around it by accident.

        An amendment is the honest path when a plan has to change: the superseded
        seal is kept beside the new one, so the record shows what was committed
        to first and what replaced it. A plan quietly re-sealed leaves no such
        trace, which is the difference between amending a protocol and rewriting
        history.
        """
        if path.exists() and not amend:
            raise FileExistsError(
                f"{path} already holds a sealed plan; a preregistration that can be "
                "overwritten commits to nothing. Re-seal with amend=True to record "
                "an amendment, which keeps the superseded seal alongside it."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        superseded: list[dict[str, Any]] = []
        if path.exists():
            import json

            previous = json.loads(path.read_text(encoding="utf-8"))
            superseded = list(previous.get("superseded", []))
            superseded.append(
                {
                    "plan_hash": previous.get("plan_hash"),
                    "plan": previous.get("plan"),
                    "amended_at": datetime.now(UTC).isoformat(timespec="seconds"),
                }
            )
        payload = {
            "plan": self.to_payload(),
            "plan_hash": self.plan_hash,
            "sealed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "amendments": len(superseded),
            "superseded": superseded,
        }
        path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        return self.plan_hash


@dataclass(frozen=True, slots=True)
class SealCheck:
    """Whether an analysis is running against the plan it claims."""

    matches: bool
    sealed_hash: str
    current_hash: str
    status: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "matches": self.matches,
            "sealed_hash": self.sealed_hash,
            "current_hash": self.current_hash,
            "status": self.status,
        }


def verify_seal(plan: Preregistration, sealed_hash: str) -> SealCheck:
    """Compare a plan against the hash it was sealed with."""
    current = plan.plan_hash
    matches = current == sealed_hash
    return SealCheck(
        matches=matches,
        sealed_hash=sealed_hash,
        current_hash=current,
        status=(
            "confirmatory: the plan matches its seal"
            if matches
            else (
                "the plan has changed since it was sealed; any result from it is "
                "exploratory and must be reported as such"
            )
        ),
    )


def load_seal(path: Path) -> tuple[dict[str, Any], str]:
    """Read a sealed plan and its hash."""
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["plan"], str(payload["plan_hash"])


def build_preregistration(config: dict[str, Any]) -> Preregistration:
    """Construct a preregistration from configuration."""
    declared = config.get("hypotheses")
    if not isinstance(declared, list) or not declared:
        raise ValueError("a preregistration must declare a non-empty 'hypotheses' list")

    hypotheses: list[Hypothesis] = []
    for index, entry in enumerate(declared):
        if not isinstance(entry, dict):
            raise ValueError(f"hypothesis {index} must be a mapping")
        params = dict(entry)
        identifier = str(params.pop("id", f"H{index + 1}"))
        try:
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=identifier,
                    question=str(params.pop("question", "")),
                    indicator=str(params.pop("indicator")),
                    treatment=str(params.pop("treatment")),
                    control=str(params.pop("control")),
                    direction=str(params.pop("direction", "greater")),
                    predicted_effect=float(params.pop("predicted_effect", 0.0)),
                    rationale=str(params.pop("rationale", "")),
                )
            )
        except KeyError as error:
            raise ValueError(f"{identifier}: missing required key {error}") from error
        if params:
            raise ValueError(f"{identifier}: unknown keys {sorted(params)}")

    kwargs: dict[str, Any] = {}
    for key in ("alpha", "minimum_replicates", "notes"):
        if key in config:
            kwargs[key] = config[key]
    if "exclusion_criteria" in config:
        kwargs["exclusion_criteria"] = tuple(str(c) for c in config["exclusion_criteria"])

    try:
        return Preregistration(
            campaign=str(config.get("campaign", "unnamed")),
            hypotheses=tuple(hypotheses),
            **kwargs,
        )
    except TypeError as error:
        raise ValueError(f"preregistration rejected its parameters: {error}") from error
