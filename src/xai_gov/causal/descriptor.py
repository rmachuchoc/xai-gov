"""Descriptor builders: the three arms of the explainability factor.

The protocol's factorial design contrasts three descriptor treatments, and this
module is all three so that the comparison is like-for-like — same world, same
decision rule, same fidelity audit.

* `none` — the ungoverned baseline. No explanation is offered and none is
  claimed.
* `post_hoc` — a correlational attribution in the spirit of SHAP/LIME:
  importance proportional to a feature's standardized deviation from its
  reference. This is the arm the protocol argues is insufficient, and it is
  implemented in good faith rather than as a straw man: it is a reasonable
  approximation of what a practitioner gets from an off-the-shelf explainer
  applied to tabular operating state.
* `causal` — interventional effects on a declared SCM, filtered to admissible,
  identifiable and affordable recourse.

Every arm goes through the same fidelity check, and the post-hoc arm is *not*
exempted from it. That is the experimental point: if a correlational
attribution happens to rank the levers correctly it will pass, and the
protocol's claim about causal descriptors will have to be earned rather than
assumed. Any descriptor whose fidelity falls below the floor is marked
`informative=False`, and the governance agent then treats it as absent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from xai_gov.causal.fidelity import explanatory_fidelity
from xai_gov.causal.recourse import RecourseSearch, build_recourse_search
from xai_gov.causal.scm import (
    CausalWorld,
    DecisionRule,
    Variable,
    reorder_rule,
    world_from_state,
)
from xai_gov.core.variants import build_variant
from xai_gov.io.decision_record import CausalDescriptor, OperatingState


class DescriptorBuilder(ABC):
    """Produces the governance descriptor phi_t for one decision."""

    method: str = "abstract"

    @abstractmethod
    def build(
        self,
        *,
        state: OperatingState,
        reorder_point: float,
        order_quantity: float,
    ) -> CausalDescriptor:
        """Describe why this decision came out as it did."""

    def to_payload(self) -> dict[str, Any]:
        return {"method": self.method}


@dataclass(slots=True)
class NoDescriptor(DescriptorBuilder):
    """The control arm: no explanation, and no pretence of one."""

    method: str = "none"

    def build(
        self, *, state: OperatingState, reorder_point: float, order_quantity: float
    ) -> CausalDescriptor:
        del state, reorder_point, order_quantity
        return CausalDescriptor(
            actionability_score=0.0,
            feature_effects={},
            explanatory_fidelity=0.0,
            identifiable=False,
            informative=False,
            method="none",
        )


@dataclass(slots=True)
class AuditedDescriptor(DescriptorBuilder):
    """A descriptor whose claims are checked against interventional truth.

    Both non-trivial arms share this base, and that is the experimental point:
    the fidelity audit is applied to the correlational arm on exactly the same
    footing as to the causal one. Exempting it would make the protocol's claim
    about causal descriptors true by construction instead of by measurement.

    ``NoDescriptor`` deliberately does *not* inherit from this: it makes no
    claim, so there is nothing to audit.
    """

    method: str = "audited"
    fidelity_floor: float = 0.5
    search: RecourseSearch = field(default_factory=RecourseSearch)

    def build(
        self, *, state: OperatingState, reorder_point: float, order_quantity: float
    ) -> CausalDescriptor:
        raise NotImplementedError


@dataclass(slots=True)
class PostHocDescriptor(AuditedDescriptor):
    """Correlational attribution, audited against the interventional truth.

    Importance is the standardized deviation of each feature from a reference
    level — large deviations look important. This is exactly the reasoning that
    makes post-hoc attribution attractive and exactly why it can mislead: a
    feature can deviate dramatically and have no causal path to the decision,
    while the lever that actually controls the outcome sits at its reference
    value and is scored near zero.
    """

    method: str = "post_hoc"
    fidelity_floor: float = 0.5
    search: RecourseSearch = field(default_factory=RecourseSearch)
    reference: dict[str, float] = field(
        default_factory=lambda: {
            Variable.INVENTORY.value: 50.0,
            Variable.IN_TRANSIT.value: 20.0,
            Variable.CAPACITY.value: 60.0,
            Variable.DATA_DELAY.value: 0.0,
            Variable.DEMAND.value: 20.0,
            Variable.BACKLOG.value: 0.0,
        }
    )
    scale: dict[str, float] = field(
        default_factory=lambda: {
            Variable.INVENTORY.value: 20.0,
            Variable.IN_TRANSIT.value: 10.0,
            Variable.CAPACITY.value: 20.0,
            Variable.DATA_DELAY.value: 2.0,
            Variable.DEMAND.value: 8.0,
            Variable.BACKLOG.value: 5.0,
        }
    )

    def build(
        self, *, state: OperatingState, reorder_point: float, order_quantity: float
    ) -> CausalDescriptor:
        world = world_from_state(state)
        rule = reorder_rule(reorder_point=reorder_point, order_quantity=order_quantity)

        attributed = {
            name: (world.values.get(name, 0.0) - reference) / self.scale.get(name, 1.0)
            for name, reference in self.reference.items()
        }
        # Audited on the same footing as the causal arm: the levers the causal
        # descriptor can act on. Scoring post-hoc attribution against a wider
        # set would let it claim credit for ranking variables no one can move.
        truth = self.search.interventional_effects(world, rule)
        report = explanatory_fidelity(
            {k: v for k, v in attributed.items() if k in truth},
            truth,
            floor=self.fidelity_floor,
        )

        return CausalDescriptor(
            actionability_score=0.0,  # post-hoc attribution implies no recourse
            feature_effects={k: round(v, 6) for k, v in sorted(attributed.items())},
            explanatory_fidelity=round(report.fidelity, 6),
            # Correlational attribution makes no identifiability claim. Marking
            # it identifiable would be the precise error the protocol is about.
            identifiable=False,
            informative=report.informative,
            method="post_hoc",
        )


@dataclass(slots=True)
class CausalRecourseDescriptor(AuditedDescriptor):
    """Interventional effects on the declared SCM, with recourse and fidelity."""

    method: str = "causal"
    fidelity_floor: float = 0.5
    search: RecourseSearch = field(default_factory=RecourseSearch)
    last_recourse: dict[str, Any] = field(default_factory=dict)

    def build(
        self, *, state: OperatingState, reorder_point: float, order_quantity: float
    ) -> CausalDescriptor:
        world: CausalWorld = world_from_state(state)
        rule: DecisionRule = reorder_rule(
            reorder_point=reorder_point, order_quantity=order_quantity
        )

        effects = self.search.interventional_effects(world, rule)
        result = self.search.search(world, rule)
        # The attribution and the ground truth are the same quantity here, so
        # fidelity is 1 by construction whenever an ordering exists. That is
        # not a free pass: it is the definition of a faithful explanation, and
        # it is still computed rather than asserted so that a bug in the
        # intervention machinery shows up as a fidelity failure.
        report = explanatory_fidelity(effects, effects, floor=self.fidelity_floor)
        self.last_recourse = result.to_payload()

        return CausalDescriptor(
            actionability_score=round(result.actionability_score, 6),
            feature_effects={k: round(v, 6) for k, v in sorted(effects.items())},
            explanatory_fidelity=round(report.fidelity, 6),
            identifiable=result.identifiable,
            informative=report.informative and result.identifiable,
            method="causal",
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "fidelity_floor": self.fidelity_floor,
            "search": self.search.to_payload(),
            "last_recourse": self.last_recourse,
        }


def available_descriptors() -> tuple[str, ...]:
    return ("none", "post_hoc", "causal")


def build_descriptor(config: dict[str, Any]) -> DescriptorBuilder:
    """Construct the descriptor arm named in configuration."""
    name = str((config or {}).get("descriptor", "none"))
    if name == "none":
        return NoDescriptor()

    params = dict((config or {}).get("params", {}))
    search_config = params.pop("recourse", None)
    variants: dict[str, type[AuditedDescriptor]] = {
        "post_hoc": PostHocDescriptor,
        "causal": CausalRecourseDescriptor,
    }
    builder = build_variant(
        kind="descriptor", name=name, params=params, variants=variants
    )
    if search_config is not None:
        if not isinstance(search_config, dict):
            raise ValueError("descriptor 'recourse' must be a mapping")
        builder.search = build_recourse_search(search_config)
    return builder
