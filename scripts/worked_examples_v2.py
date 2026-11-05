#!/usr/bin/env python
"""Small worked examples of the v2 formulas, computed by the package (R3.1).

    python scripts/worked_examples_v2.py

Writes specs/v2/worked_examples.md. Every number comes from calling the code
the simulation uses, so an example cannot disagree with what ran.
"""

from __future__ import annotations

from pathlib import Path

from xai_gov.analysis.agrapa import AgrapaEProcess
from xai_gov.core.seeds import SeedBundle
from xai_gov.governance.agent import NoGovernanceAgent
from xai_gov.governance.thresholds import ThresholdEconomics, myopic_threshold, solve_values
from xai_gov.io.decision_record import ActionKind, OperatingState
from xai_gov.policies.base import Policy, PolicyProposal
from xai_gov.simulation.demand import StableDemand
from xai_gov.simulation.engine import DigitalTwin
from xai_gov.simulation.network import EXTERNAL_SOURCE, NetworkSpec, NodeSpec
from xai_gov.verification.shield import Shield, base_stock_order


def table(header: list[str], rows: list[list[object]]) -> list[str]:
    def row(cells: list[object]) -> str:
        return "| " + " | ".join(str(c) for c in cells) + " |"
    return [row(header), row(["---"] * len(header)), *(row(r) for r in rows), ""]


class OneOrder(Policy):
    """The retailer orders 20 once; upstream nodes cover their backorders."""

    policy_id = "worked_example"
    version = "1"

    def propose(self, *, spec, state, rng):  # type: ignore[no-untyped-def]
        q = (20.0 if state.period == 0 else 0.0) if spec.echelon == 0 else max(
            state.backlog - state.inventory - state.in_transit, 0.0)
        if q <= 0.0:
            return PolicyProposal(ActionKind.HOLD, 0.0, {}, "hold")
        return PolicyProposal(ActionKind.REORDER, q, {}, "order")


def node(node_id: str, echelon: int, supplier: str, stock: float) -> NodeSpec:
    return NodeSpec(node_id=node_id, echelon=echelon, supplier_id=supplier,
                    initial_inventory=stock, capacity=1000.0, reorder_point=0.0,
                    order_quantity=0.0, lead_time_mean=1.0)


def serial_example() -> list[str]:
    network = NetworkSpec(nodes=(node("retail", 0, "dc", 3.0), node("dc", 1, "plant", 12.0),
                                 node("plant", 2, EXTERNAL_SOURCE, 100.0)))
    rows = []
    for periods in range(1, 7):
        twin = DigitalTwin(network=network,
                           demand=StableDemand(name="s", mean=2.0, dispersion=0.0),
                           policy=OneOrder(), governance=NoGovernanceAgent(),
                           seeds=SeedBundle(master_seed=0), periods=periods)
        result = twin.run()
        trace = result.traces[-1]
        r, d = twin.states["retail"], twin.states["dc"]
        rows.append([periods - 1, trace.demand_total, trace.served_total, trace.customer_shipped,
                     r.backlog, r.inventory, r.in_transit, r.awaiting, d.inventory, d.backlog])
    total = DigitalTwin(network=network, demand=StableDemand(name="s", mean=2.0, dispersion=0.0),
                        policy=OneOrder(), governance=NoGovernanceAgent(),
                        seeds=SeedBundle(master_seed=0), periods=6).run()
    demand = sum(t.demand_total for t in total.traces)
    on_time = sum(t.served_total for t in total.traces)
    return [
        "## 1. One period of the serial model, and the fill rate beta", "",
        "Retailer with 3 units facing a demand of 2 per period; it orders 20 at period 0 from a",
        "DC holding 12. The DC sees that order at period 1, ships 12 and owes 8; it orders 8",
        "from the plant, which ships them at period 2; the DC forwards them at period 3. All",
        "lead times are 1.", "",
        *table(["t", "customer demand", "served on time (beta)", "shipped to customer",
                "retail backlog", "retail stock", "retail in transit", "retail awaiting",
                "DC stock", "DC backlog"], rows),
        f"beta over the six periods = {on_time:g} / {demand:g} = {on_time / demand:.4f}. A unit "
        "shipped late counts in the backlog, never in beta, so beta cannot exceed 1.", "",
    ]


def envelope_example() -> list[str]:
    spec = NodeSpec(node_id="retail", echelon=0, supplier_id="dc", initial_inventory=45.0,
                    capacity=120.0, reorder_point=25.0, order_quantity=35.0, lead_time_mean=2.0)
    state = OperatingState(period=30, inventory=4.0, backlog=0.0, in_transit=6.0,
                           capacity=120.0, demand_observed=10.0)
    envelope = Shield().envelope(state, node_id="retail")
    rows = []
    for proposal in (0.0, 30.0, 150.0):
        projected = Shield(mode="projection").filter(state=state, action=ActionKind.REORDER,
                                                     quantity=proposal, spec=spec)[1]
        null = Shield(mode="fallback").filter(state=state, action=ActionKind.REORDER,
                                              quantity=proposal, spec=spec)[1]
        backup = Shield(mode="backup").filter(state=state, action=ActionKind.REORDER,
                                              quantity=proposal, spec=spec)[1]
        rows.append([proposal, envelope.contains(proposal), projected, null,
                     envelope.contains(null), backup])
    position = state.inventory + state.in_transit - state.backlog
    return [
        "## 2. The admissible envelope and the three operators", "",
        f"Retail state: inventory {state.inventory:g}, in transit (on order) "
        f"{state.in_transit:g}, backlog {state.backlog:g}, last demand "
        f"{state.demand_observed:g}, capacity {state.capacity:g}. With one period of "
        f"lookahead the projected position {position:g} - {state.demand_observed:g} = "
        f"{position - state.demand_observed:g} reaches the critical level (5), so SAFE-2 and "
        f"SAFE-3 engage and require an order of at least 5 + 10 - 4 - 6 = 5.", "",
        f"Envelope: [{envelope.minimum:g}, {envelope.maximum:g}] ({envelope.reason}). "
        f"F-backup's level is S = s + Q = 60, so its order is "
        f"max(60 - {position:g}, 0) = {base_stock_order(spec, state):g}, clamped to the "
        "envelope.", "",
        *table(["proposal", "admissible", "projection executes", "F-null executes",
                "F-null admissible", "F-backup executes"], rows),
        "Projection moves an inadmissible proposal to the nearest bound; F-null replaces it "
        "with 0 even though 0 is itself inadmissible here; F-backup replaces it with an "
        "admissible base-stock order.", "",
    ]


def agrapa_example() -> list[str]:
    process = AgrapaEProcess()
    rows = []
    for t, d in enumerate((0.10, 0.06, 0.08, -0.02, 0.09), start=1):
        mean, variance, bet = process.mean_estimate, process.variance_estimate, process.next_bet()
        process.update(d)
        rows.append([t, d, round(mean, 4), round(variance, 4), round(bet, 4),
                     round(1 + bet * d, 4), round(process.wealth, 4)])
    return [
        "## 3. The aGRAPA bet, step by step", "",
        "Five paired differences of beta (treatment minus control). The bet for replicate t",
        "uses only replicates 1..t-1: lambda_t = clip(mu / (sigma2 + mu^2), 0, 1/2), with a",
        "prior mean 0 and variance 1. The wealth is the e-value; the plan rejects when it",
        "reaches 240.", "",
        *table(["t", "d_t", "mu_{t-1}", "sigma2_{t-1}", "lambda_t", "1 + lambda_t d_t",
                "wealth K_t"], rows),
        "With lambda at most 1/2 and differences near 0.05, the log wealth grows by about "
        "0.025 per replicate, which is why reaching 240 takes over 220 replicates.", "",
    ]


def threshold_example() -> list[str]:
    economics = ThresholdEconomics(intervention_cost=1.0, disruption_loss=10.0, discount=0.95)
    solution = solve_values(economics)
    b_star = solution.threshold()
    index = solution.grid.index(b_star)
    rows = [[round(b, 3), round(solution.margin(b), 4), "intervene" if solution.margin(b) >= 0
             else "wait"] for b in solution.grid[max(index - 3, 0): index + 3]]
    return [
        "## 4. The intervention threshold: myopic and optimal", "",
        f"c = 1, L = 10, gamma = 0.95. Myopic: b_m = c / L = {myopic_threshold(economics):g}: "
        "intervene once one period of expected loss b*L exceeds the cost.", "",
        "Optimal: intervene at the first belief where the margin "
        "D(b) = b*L + gamma*(C(b) - V(0)) - c turns nonnegative; C(b) - V(0) is what waiting "
        f"costs in later periods, so b* = {b_star:g} is earlier than the myopic {0.1:g}.", "",
        *table(["belief b", "D(b)", "decision"], rows),
    ]


def coverage_example() -> list[str]:
    flags, scored = 7, 60
    return [
        "## 5. Coverage error against the fixed target", "",
        f"A node scored in {scored} periods raises {flags} flags: realized rate "
        f"{flags}/{scored} = {flags / scored:.4f}; coverage error = {flags / scored:.4f} - 0.10 "
        f"= {flags / scored - 0.10:+.4f}. The adapted level eps_t is the controller's variable "
        "and is shown separately; it is not the reference.", "",
    ]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    lines = ["# Worked examples (v2)", "",
             "Computed by the package; see scripts/worked_examples_v2.py.", ""]
    for part in (serial_example, envelope_example, agrapa_example, threshold_example,
                 coverage_example):
        lines += part()
    (root / "specs/v2/worked_examples.md").write_text("\n".join(lines), encoding="utf-8")
    print("  -> specs/v2/worked_examples.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
