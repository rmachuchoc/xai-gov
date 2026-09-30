"""The governance agent as a constrained POMDP.

This is the agent the protocol formalizes as
⟨S, A, O, T, Ω, R, C, B, γ⟩. What distinguishes it from the rule-based
mediation it replaces is not that it has more branches, but that its single
decision boundary is *derived*:

* the state is a belief over the latent risk regime, filtered from the
  conformal signal rather than read off a counter;
* the boundary b* comes from value iteration on declared economics, so
  changing the cost of an intervention moves the threshold automatically and
  in the direction Theorem 2 predicts;
* the budget constraint is priced by a Lagrange multiplier, so the agent
  intervenes less as oversight capacity becomes scarce, and reports κ* — what
  that scarcity cost.

The action chosen at a given belief is then a question of *rights*, not of
risk: the autonomy regime decides whether an unauthorized action is escalated
(H0: the organization holds the decision) or vetoed (H1: the range was agreed
in advance), and whether autonomy may be reduced at all (H2 only).

One ordering commitment: the agent decides on the belief *after* folding in
this period's signal, but the threshold was derived before the run began.
Deriving the threshold from data the agent has already acted on would fit the
boundary to its own decisions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from xai_gov.causal.descriptor import DescriptorBuilder, NoDescriptor
from xai_gov.conformal.detector import DetectorBank
from xai_gov.core.logging import get_logger
from xai_gov.core.seeds import derive_seed
from xai_gov.governance.agent import (
    NEUTRAL_DESCRIPTOR,
    GovernanceAgent,
    GovernanceVerdict,
)
from xai_gov.governance.autonomy import AutonomyRegime, AutonomyState
from xai_gov.governance.belief import BeliefFilter, RegimeModel
from xai_gov.governance.budget import GovernanceBudget
from xai_gov.governance.thresholds import ThresholdEconomics, ThresholdSolution, solve_threshold
from xai_gov.io.decision_record import (
    ActionKind,
    GovernanceAction,
    OperatingState,
    RiskRegime,
)
from xai_gov.oversight.delegation import (
    ComplementarityLedger,
    DeferralPolicy,
    NoDeferral,
)
from xai_gov.oversight.supervisor import SimulatedSupervisor
from xai_gov.verification.monitors import MonitorLike, NoMonitor
from xai_gov.verification.shield import NoShield, ShieldLike

if TYPE_CHECKING:
    from xai_gov.policies.base import PolicyProposal
    from xai_gov.simulation.network import NodeSpec

_LOG = get_logger("governance.cpomdp")


@dataclass
class CpomdpGovernanceAgent(GovernanceAgent):
    """Belief-based governance under a budget constraint and a derived boundary."""

    agent_id: str = "cpomdp"
    version: str = "1.0.0"
    autonomy_regime: str = "H2"

    detectors: DetectorBank = field(default_factory=lambda: DetectorBank({}))
    descriptor_builder: DescriptorBuilder = field(default_factory=NoDescriptor)
    shield: ShieldLike = field(default_factory=NoShield)
    monitor: MonitorLike = field(default_factory=NoMonitor)
    deferral: DeferralPolicy = field(default_factory=NoDeferral)
    supervisor: SimulatedSupervisor | None = None
    regime_model: RegimeModel = field(default_factory=RegimeModel)
    economics: ThresholdEconomics = field(default_factory=ThresholdEconomics)
    budget: GovernanceBudget = field(default_factory=GovernanceBudget)
    quantity_bound: float = 60.0
    safe_mode_belief: float = 0.9
    escalation_belief: float = 0.6
    #: How much of the disruption loss an affordable recourse removes. A
    #: situation an operator can repair at admissible cost is less costly to
    #: leave alone than one they cannot, and by Theorem 2's monotonicity a
    #: lower effective loss raises the intervention threshold. Off by default:
    #: a descriptor-blind agent is the control arm, so it must be what a bare
    #: construction gives. The causal experiment switches the channel on.
    recourse_loss_relief: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.escalation_belief <= self.safe_mode_belief <= 1.0:
            raise ValueError(
                "beliefs must satisfy 0 < escalation_belief <= safe_mode_belief <= 1"
            )
        if not 0.0 <= self.recourse_loss_relief < 1.0:
            raise ValueError(
                "recourse_loss_relief must lie in [0, 1); relieving the whole loss "
                "would make intervention never worthwhile"
            )
        self._autonomy = AutonomyState(initial=AutonomyRegime(self.autonomy_regime))
        self._filters: dict[str, BeliefFilter] = {}
        self._period = 0
        self.ledger = ComplementarityLedger()
        # A separate stream for the counterfactual human, so scoring the
        # baseline never perturbs the sequence the real supervisor draws from.
        self._counterfactual = random.Random(
            derive_seed(self.supervisor.seed if self.supervisor else 0, "counterfactual")
        )
        self._solution: ThresholdSolution = solve_threshold(
            self.economics,
            persistence=self.regime_model.persistence,
            flag_probability_nominal=self.regime_model.flag_probability(RiskRegime.NOMINAL),
            flag_probability_disruption=self.regime_model.flag_probability(
                RiskRegime.DISRUPTION
            ),
        )
        # A second threshold, derived the same way from the loss an affordable
        # recourse leaves behind. Solved once at construction rather than per
        # decision: the economics do not change during a run, and re-solving
        # inside the loop would put a value iteration in the hot path for a
        # number that never moves.
        if self.recourse_loss_relief > 0.0:
            relieved = ThresholdEconomics(
                intervention_cost=self.economics.intervention_cost,
                disruption_loss=self.economics.disruption_loss
                * (1.0 - self.recourse_loss_relief),
                discount=self.economics.discount,
                false_intervention_cost=self.economics.false_intervention_cost,
            )
            self._repairable_solution: ThresholdSolution | None = solve_threshold(
                relieved,
                persistence=self.regime_model.persistence,
                flag_probability_nominal=self.regime_model.flag_probability(
                    RiskRegime.NOMINAL
                ),
                flag_probability_disruption=self.regime_model.flag_probability(
                    RiskRegime.DISRUPTION
                ),
            )
        else:
            self._repairable_solution = None
        self.repairable_decisions = 0
        _LOG.info(
            "intervention threshold derived",
            extra={
                "threshold": round(self._solution.threshold, 4),
                "myopic": round(self._solution.myopic, 4),
                "converged": self._solution.converged,
                "regime": self._autonomy.current.value,
            },
        )

    # -- accessors --------------------------------------------------------
    @property
    def threshold(self) -> float:
        """b*, the derived intervention boundary."""
        return self._solution.threshold

    def threshold_for(self, descriptor: Any) -> tuple[float, bool]:
        """The boundary in force for this decision, and whether recourse moved it.

        This is where an explanation becomes a decision rather than a record.
        An uninformative descriptor — one whose explanatory fidelity failed its
        audit — changes nothing, which is the point of auditing it: a
        low-fidelity explanation must not be able to talk the agent out of
        intervening.
        """
        if self._repairable_solution is None or not descriptor.informative:
            return self._solution.threshold, False
        recourse = getattr(self.descriptor_builder, "last_recourse", None) or {}
        if not recourse.get("cheapest_recourse"):
            return self._solution.threshold, False
        return self._repairable_solution.threshold, True

    @property
    def solution(self) -> ThresholdSolution:
        return self._solution

    def belief_for(self, node_id: str) -> BeliefFilter:
        if node_id not in self._filters:
            self._filters[node_id] = BeliefFilter(model=self.regime_model)
        return self._filters[node_id]

    # -- mediation --------------------------------------------------------
    def mediate(
        self,
        *,
        spec: NodeSpec,
        state: OperatingState,
        proposal: PolicyProposal,
    ) -> GovernanceVerdict:
        self._period = state.period
        self.budget.tick()
        # Monitors watch every period, governed or not. Instrumenting only the
        # shielded arm would make the two arms unobservable in the same terms,
        # and the price of verified safety is a difference between them.
        self.monitor.observe(state)

        signal = self.detectors.for_node(spec.node_id).observe(state.demand_observed)
        belief_filter = self.belief_for(spec.node_id)
        belief = belief_filter.update(signal)
        disruption = belief_filter.disruption

        # The descriptor is built from the proposal's own terms, so the
        # explanation describes the decision that was actually made rather than
        # a generic one. An uninformative descriptor is carried as such: the
        # agent must be able to see that it is acting without explanation.
        descriptor = self.descriptor_builder.build(
            state=state,
            reorder_point=spec.reorder_point,
            order_quantity=proposal.quantity,
        )

        rights = self._autonomy.rights
        base_cost = self.economics.intervention_cost
        priced_cost = self.budget.effective_cost(base_cost)

        # 1. Rights come first. An action the regime does not authorize is not
        #    weighed against risk at all: the system simply lacks the standing
        #    to take it, whatever the belief says.
        if not rights.permits(
            proposal.action, quantity=proposal.quantity, bound=self.quantity_bound
        ):
            action = rights.default_action_when_not_permitted()
            return self._intervene(
                action=action,
                state=state,
                proposal=proposal,
                belief=belief,
                signal=signal,
                cost=base_cost,
                rationale=(
                    f"{self._autonomy.current.value} does not authorize "
                    f"{proposal.action.value} of {proposal.quantity:.2f} "
                    f"(bound {self.quantity_bound:.2f})"
                ),
                descriptor=descriptor,
                escalated=action is GovernanceAction.ESCALATE,
            )

        # 2. Below the derived boundary the proposal stands. Note what is not
        #    happening here: no threshold was read from a config file.
        threshold, repairable = self.threshold_for(descriptor)
        if repairable:
            self.repairable_decisions += 1
        if disruption < threshold:
            return self._approve(
                state=state,
                proposal=proposal,
                descriptor=descriptor,
                signal=signal,
                belief=belief,
                rationale=(
                    f"belief in disruption {disruption:.3f} below derived "
                    f"threshold {threshold:.3f}"
                    + (
                        "; raised because the descriptor found affordable recourse"
                        if repairable
                        else ""
                    )
                ),
            )

        # 3. The boundary is crossed, but intervention must be affordable. An
        #    exhausted budget is recorded as a refusal, not as an approval:
        #    the distinction is the whole point of tracking a constraint.
        if not self.budget.can_afford(priced_cost):
            self.budget.refuse(base_cost)
            self.budget.update_multiplier(self._period)
            return self._approve(
                state=state,
                proposal=proposal,
                descriptor=descriptor,
                signal=signal,
                belief=belief,
                rationale=(
                    f"intervention warranted at belief {disruption:.3f} but the "
                    f"governance budget is exhausted "
                    f"({self.budget.discounted_spent:.2f}/{self.budget.allowance:.2f}); "
                    "executed unmediated and recorded as a refusal"
                ),
            )

        # 4. Which intervention, by severity of belief and by rights held.
        if disruption >= self.safe_mode_belief:
            return self._intervene(
                action=GovernanceAction.ACTIVATE_SAFE_MODE,
                state=state,
                proposal=proposal,
                belief=belief,
                signal=signal,
                cost=base_cost,
                rationale=(
                    f"belief in disruption {disruption:.3f} at or above safe-mode "
                    f"level {self.safe_mode_belief:.2f}"
                ),
                descriptor=descriptor,
                safe_mode=True,
            )

        if disruption >= self.escalation_belief and rights.may_escalate:
            # Whether to spend a human's attention is itself a decision. The
            # deferral policy weighs the supervisor's expected advantage in
            # this regime against what the attention costs; a fixed threshold
            # spends it identically whether or not doing so helps.
            regime = belief_filter.most_likely
            attention = (
                self.supervisor.attention_remaining if self.supervisor is not None else 0.0
            )
            deferral = self.deferral.should_defer(
                regime=regime,
                disruption_belief=disruption,
                attention_remaining=attention,
            )
            if deferral.defer and self.supervisor is not None:
                return self._escalate_to_human(
                    state=state,
                    proposal=proposal,
                    belief=belief,
                    signal=signal,
                    descriptor=descriptor,
                    regime=regime,
                    cost=base_cost,
                    deferral_reason=deferral.reason,
                )
            # The policy declined to spend attention here. That is a decision
            # with consequences, so it is recorded as one rather than silently
            # becoming an ordinary veto.
            lever = self._cheapest_lever(descriptor)
            return self._intervene(
                action=GovernanceAction.ESCALATE,
                state=state,
                proposal=proposal,
                belief=belief,
                signal=signal,
                cost=base_cost,
                rationale=(
                    f"belief in disruption {disruption:.3f} at or above escalation "
                    f"level {self.escalation_belief:.2f}; referred to human "
                    f"oversight{lever}"
                ),
                descriptor=descriptor,
                escalated=True,
            )

        if rights.may_reduce_autonomy and self._autonomy.reduce():
            return self._intervene(
                action=GovernanceAction.REDUCE_AUTONOMY,
                state=state,
                proposal=proposal,
                belief=belief,
                signal=signal,
                cost=base_cost,
                rationale=(
                    f"belief in disruption {disruption:.3f} crossed the derived "
                    f"threshold {self.threshold:.3f}; autonomy reduced to "
                    f"{self._autonomy.current.value}"
                ),
                descriptor=descriptor,
            )

        return self._intervene(
            action=GovernanceAction.VETO,
            state=state,
            proposal=proposal,
            belief=belief,
            signal=signal,
            cost=base_cost,
            rationale=(
                f"belief in disruption {disruption:.3f} crossed the derived "
                f"threshold {self.threshold:.3f}; proposal vetoed"
            ),
            descriptor=descriptor,
        )

    def _cheapest_lever(self, descriptor: Any) -> str:
        """Name the cheapest recourse, when the descriptor found one."""
        recourse = getattr(self.descriptor_builder, "last_recourse", None)
        if not descriptor.informative or not recourse:
            return ""
        cheapest = recourse.get("cheapest_recourse")
        if not cheapest:
            return "; no affordable recourse was found"
        levers = ", ".join(
            f"{name} to {value:.1f}" for name, value in cheapest["interventions"].items()
        )
        return f"; cheapest recourse is {levers} at cost {cheapest['cost']:.1f}"

    # -- helpers ----------------------------------------------------------
    def _approve(
        self,
        *,
        state: OperatingState,
        proposal: PolicyProposal,
        descriptor: Any,
        signal: Any,
        belief: dict[str, float],
        rationale: str,
    ) -> GovernanceVerdict:
        """Let the proposal stand, subject to the shield.

        An approval passes through the shield exactly as an intervention does.
        Governance approving an action is not a safety judgement — it is a
        judgement about risk of *disruption* — and a shield that only inspected
        the decisions governance already distrusted would never see the case it
        exists for: a confidently-approved action that violates a property.
        """
        action, quantity, outcome = self.shield.filter(
            state=state, action=proposal.action, quantity=proposal.quantity
        )
        self._record_outcome(
            regime=RiskRegime.NOMINAL,
            truly_unsafe=not self._is_safe(state, proposal.quantity),
            team_correct=self._is_safe(state, quantity),
        )
        return GovernanceVerdict(
            action=GovernanceAction.APPROVE,
            final_action=action,
            final_quantity=quantity,
            rationale=rationale + (f"; {self._shield_note(outcome)}" if outcome.activated else ""),
            descriptor=descriptor,
            conformal=signal,
            belief=belief,
            intervention_cost=0.0,
            escalated=False,
            safe_mode=False,
            shield=outcome,
        )

    def _escalate_to_human(
        self,
        *,
        state: OperatingState,
        proposal: PolicyProposal,
        belief: dict[str, float],
        signal: Any,
        descriptor: Any,
        regime: RiskRegime,
        cost: float,
        deferral_reason: str,
    ) -> GovernanceVerdict:
        """Spend a unit of human attention, and record what it bought.

        The ledger entry is the point of this method. It records what the team
        did *and* what each actor would have done alone, because a team that
        beats the system might simply be the human doing all the work, and
        strict complementarity cannot be read off team performance.
        """
        assert self.supervisor is not None
        # Ground truth, available to the simulator only. It never reaches the
        # agent; it exists so the study can score the team.
        truly_unsafe = not self._is_safe(state, proposal.quantity)
        judgement = self.supervisor.review(regime=regime, truly_unsafe=truly_unsafe)
        self.deferral.observe_outcome(
            regime=regime, deferred=judgement.available, correct=judgement.correct
        )
        self._record_outcome(
            regime=regime,
            truly_unsafe=truly_unsafe,
            team_correct=judgement.correct,
            human_correct=judgement.correct,
        )

        quantity = proposal.quantity if judgement.approved else 0.0
        action = proposal.action if judgement.approved else ActionKind.HOLD
        action, quantity, outcome = self.shield.filter(
            state=state, action=action, quantity=quantity, escalated=True
        )

        self.budget.charge(cost, self._period)
        self.budget.update_multiplier(self._period)
        availability = (
            "supervisor reviewed" if judgement.available else "supervisor unavailable"
        )
        return GovernanceVerdict(
            action=GovernanceAction.ESCALATE,
            final_action=action,
            final_quantity=quantity,
            rationale=(
                f"deferred to human oversight ({deferral_reason}); {availability}: "
                f"{judgement.reason}{self._cheapest_lever(descriptor)}"
                + (f"; {self._shield_note(outcome)}" if outcome.activated else "")
            ),
            descriptor=descriptor,
            conformal=signal,
            belief=belief,
            intervention_cost=cost,
            escalated=True,
            safe_mode=False,
            shield=outcome,
        )

    def _record_outcome(
        self,
        *,
        regime: RiskRegime,
        truly_unsafe: bool,
        team_correct: bool,
        human_correct: bool | None = None,
    ) -> None:
        """Score the team and both actors alone, on every decision.

        Recording only the deferred decisions would make the complementarity
        test vacuous: on those the team *is* the human, so the two columns
        would be identical by construction and the comparison would answer
        nothing. The counterfactual human is therefore simulated on every
        decision — and deliberately without charging the attention budget,
        because a hypothetical review costs the organization nothing. That is
        an advantage granted to the human baseline, which makes the
        complementarity finding harder to obtain rather than easier.

        The system alone is scored as approving whatever it proposed: without
        oversight there is nothing to catch an unsafe action, so it is correct
        exactly when the action was safe.
        """
        if self.supervisor is None:
            return
        if human_correct is None:
            accuracy = self.supervisor.accuracy.get(regime.value, 0.5)
            human_correct = bool(self._counterfactual.random() < accuracy)
        self.ledger.record(
            regime=regime,
            team=team_correct,
            human_alone=human_correct,
            system_alone=not truly_unsafe,
        )

    def _is_safe(self, state: OperatingState, quantity: float) -> bool:
        """Whether an action is admissible under the specification.

        Used only to score the simulated supervisor. With no shield configured
        there is no specification to judge against, so everything counts as
        safe and the complementarity ledger records a degenerate case rather
        than inventing a ground truth.
        """
        envelope = getattr(self.shield, "envelope", None)
        if envelope is None:
            return True
        return bool(envelope(state).contains(quantity))

    def _shield_note(self, outcome: Any) -> str:
        return (
            f"shield substituted {outcome.substituted_quantity:.1f} "
            f"under {outcome.property_id}"
        )

    def _intervene(
        self,
        *,
        action: GovernanceAction,
        state: OperatingState,
        proposal: PolicyProposal,
        belief: dict[str, float],
        signal: Any,
        cost: float,
        rationale: str,
        descriptor: Any = NEUTRAL_DESCRIPTOR,
        escalated: bool = False,
        safe_mode: bool = False,
    ) -> GovernanceVerdict:
        """Charge the budget, then let the shield choose what executes.

        An intervention proposes holding, but holding is not automatically
        safe: cancelling a replenishment during a stockout is a way to violate
        a safety property, and an agent whose only tool is refusal will do it.
        The shield may therefore raise the executed quantity above what
        governance asked for. That is the point of synthesizing it from the
        specification rather than writing a safe mode by hand.
        """
        self.budget.charge(cost, self._period)
        self.budget.update_multiplier(self._period)
        final_action, final_quantity, outcome = self.shield.filter(
            state=state, action=ActionKind.HOLD, quantity=0.0, escalated=escalated
        )
        return GovernanceVerdict(
            action=action,
            final_action=final_action,
            final_quantity=final_quantity,
            rationale=rationale
            + (f"; {self._shield_note(outcome)}" if outcome.activated else ""),
            descriptor=descriptor,
            conformal=signal,
            belief=belief,
            intervention_cost=cost,
            escalated=escalated,
            safe_mode=safe_mode,
            shield=outcome,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.agent_id,
            "version": self.version,
            "autonomy": self._autonomy.to_payload(),
            "threshold": self._solution.to_payload(),
            "repairable_threshold": (
                self._repairable_solution.to_payload()
                if self._repairable_solution is not None
                else None
            ),
            "recourse_loss_relief": self.recourse_loss_relief,
            "repairable_decisions": self.repairable_decisions,
            "budget": self.budget.to_payload(),
            "regime_model": self.regime_model.to_payload(),
            "beliefs": {
                node_id: filt.to_payload() for node_id, filt in sorted(self._filters.items())
            },
            "detectors": self.detectors.to_payload(),
            "descriptor": self.descriptor_builder.to_payload(),
            "shield": self.shield.to_payload(),
            "monitor": self.monitor.to_payload(),
            "oversight": {
                "deferral": self.deferral.to_payload(),
                "supervisor": (
                    self.supervisor.to_payload() if self.supervisor is not None else None
                ),
                "complementarity": self.ledger.to_payload(),
            },
            "levels": {
                "escalation_belief": self.escalation_belief,
                "safe_mode_belief": self.safe_mode_belief,
                "quantity_bound": self.quantity_bound,
            },
        }
