#!/usr/bin/env python
"""Run the complete study: power check, campaign, sensitivity, calibration.

This is the script that produces everything a manuscript needs, in the order
the protocol requires it. Each phase writes its own artifact, and the phases
are separable so a re-run after a change does not have to redo the parts that
did not change.

    python scripts/run_study.py --replicates 30

Order matters and is not arbitrary:

1. **Power** first, because a campaign that cannot detect the effect it is
   looking for should not consume a week of compute. The check runs before
   anything executes and refuses an under-powered study unless overridden.
2. **Seal** next, so the hypotheses are fixed before any number exists.
3. **Campaign** — the runs themselves, all arms at all replicates.
4. **Sensitivity** on the declared economics, to separate findings about
   governance from findings about a chosen cost.
5. **Calibration** against a reference series, reporting the sim-to-real gap
   every conclusion is then qualified by.

The study bundle written at the end is the object a reviewer should be handed:
it carries the sealed plan, the per-cell run identifiers, the evidence for each
hypothesis, the frontier, the sensitivity indices and the calibration gap.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from xai_gov.analysis.artifacts import diagnose, write_artifacts
from xai_gov.analysis.calibration import (
    DemandSummary,
    abc_rejection,
    default_priors,
    posterior_predictive_check,
    prior_widths,
)
from xai_gov.analysis.power import analyse_power
from xai_gov.analysis.preregistration import build_preregistration, load_seal
from xai_gov.analysis.report import CAMPAIGN_REPORT_NAME, run_campaign
from xai_gov.analysis.sensitivity import build_parameter_space, sobol_analysis
from xai_gov.core.hashing import canonical_json
from xai_gov.core.logging import configure_logging, get_logger
from xai_gov.core.paths import Paths
from xai_gov.core.settings import load_settings
from xai_gov.io.writers import KPIS_NAME, read_json
from xai_gov.io.yaml_loader import load_config

_LOG = get_logger("study")

STUDY_BUNDLE_NAME = "study_bundle.json"


def phase(index: int, name: str) -> None:
    print(f"\n[{index}/5] {name}", flush=True)


def run_power_check(replicates: int, *, effect_size: float) -> dict[str, Any]:
    """Refuse to burn compute on a study that cannot see its own effect."""
    analysis = analyse_power(planned=replicates, effect_size=effect_size)
    print(f"  effect size sought      {analysis.effect_size}")
    print(f"  required (fixed n)      {analysis.required_paired}")
    print(f"  required (anytime)      {analysis.required_sequential}")
    print(f"  planned                 {analysis.planned}")
    print(f"  minimum detectable      {analysis.detectable_effect:.3f}")
    if analysis.adequate:
        print("  verdict                 adequate")
    else:
        print(
            f"  verdict                 UNDER-POWERED: "
            f"{analysis.planned} < {analysis.required_sequential}"
        )
    return analysis.to_payload()


def run_sensitivity(config: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """Sobol indices over the declared economics.

    The model is a governance-response proxy rather than a full simulation run.
    That is a deliberate trade: the full twin at every Saltelli point would
    cost thousands of runs, and the proxy is exact in the part that matters —
    it composes the same threshold solver, the same budget arithmetic and the
    same escalation levels the agent uses.

    Every declared parameter must enter the model. An earlier version passed
    four parameters to a function that read two, and reported the other two as
    non-influential: an exact zero that described the wiring, not the system.
    """
    from xai_gov.governance.thresholds import ThresholdEconomics, solve_threshold

    space = build_parameter_space(config.get("sensitivity"))
    base_samples = int(config.get("sensitivity", {}).get("base_samples", 64))

    def model(parameters: dict[str, float]) -> float:
        cost = parameters.get("intervention_cost", 1.0)
        loss = parameters.get("disruption_loss", 10.0)
        allowance = parameters.get("allowance", 20.0)
        escalation = parameters.get("escalation_belief", 0.6)
        economics = ThresholdEconomics(intervention_cost=cost, disruption_loss=loss)
        threshold = solve_threshold(economics, grid_size=101).threshold

        # The outcome is the net value the parameters imply, not the raw
        # intervention rate. The rate alone is dominated by whichever term
        # binds first — usually the budget — so a decomposition of its variance
        # reports that one term and hides the rest. Value is where both
        # channels actually meet: the threshold decides *when* to intervene and
        # the budget decides *whether it can be paid for*.
        grid = [index / 200.0 for index in range(201)]
        crossings = [belief for belief in grid if belief >= threshold]
        if not crossings:
            return 0.0

        escalated_share = len([b for b in crossings if b >= escalation]) / len(crossings)
        unit_cost = cost * (1.0 + escalated_share)
        funded = min(len(crossings), allowance / unit_cost if unit_cost > 0 else 0.0)

        # Benefit: loss avoided on the decisions that were funded, weighted by
        # the belief at which each was taken. Cost: what funding them consumed.
        avoided = sum(crossings[: int(funded)]) * loss / len(grid)
        spent = funded * unit_cost / len(grid)
        return float(avoided - spent)

    report = sobol_analysis(model, space.parameters, base_samples=base_samples, seed=seed)
    for index in report.indices:
        mark = "influential" if index.influential else "negligible"
        print(
            f"  {index.name:<20} S1={index.first_order:.3f} "
            f"ST={index.total_effect:.3f}  {mark}"
        )
    print(f"  robust to parameter choice: {report.robust}")
    return report.to_payload()


def run_calibration(
    *, seed: int, proposals: int, root: Path, column: str | None = None
) -> dict[str, Any]:
    """Calibrate the demand process and report the sim-to-real gap.

    Reads a real series when one is present under ``data/demand/`` and falls
    back to a labelled synthetic one otherwise. The distinction reaches the
    artifact: a gap computed against simulator output is a sim-to-sim gap and
    establishes nothing about external validity, so the run record says which
    one it holds rather than leaving a reader to assume.
    """
    from xai_gov.analysis.calibration import CALIBRATION_STATISTICS, bootstrap_scales
    from xai_gov.analysis.data import resolve_series

    summarize = DemandSummary()
    priors = default_priors()
    reference = resolve_series(root, column=column)
    observed = summarize(list(reference.values))

    # Each statistic scaled by its own sampling variability rather than by its
    # magnitude. Autocorrelation of a near-independent series sits close to
    # zero, so dividing by magnitude turned a noise term into the dominant
    # component of the distance and left both parameters unidentified.
    scales = bootstrap_scales(list(reference.values), summarize, seed=seed)

    def simulator(parameters: dict[str, float], sim_seed: int) -> list[float]:
        rng = np.random.default_rng(sim_seed)
        return [
            float(max(0.0, rng.normal(parameters["base_demand"], parameters["noise_scale"])))
            for _ in range(len(reference.values))
        ]

    posterior = abc_rejection(
        simulator=simulator,
        summarize=summarize,
        observed=observed,
        priors=priors,
        proposals=proposals,
        quantile=0.05,
        seed=seed,
        scales=scales,
    )
    check = posterior_predictive_check(
        simulator=simulator,
        summarize=summarize,
        observed=observed,
        posterior=posterior,
        calibrated_statistics=CALIBRATION_STATISTICS,
        draws=60,
        seed=seed + 1,
        scales=scales,
    )
    identified = posterior.identified(prior_widths(priors))

    print(f"  series                 {reference.source}")
    if reference.synthetic:
        print("  WARNING                synthetic reference: the gap below is sim-to-sim")
    elif reference.intermittent:
        print(
            f"  WARNING                {reference.zero_fraction:.0%} zero periods: "
            "intermittent demand is not what this twin models"
        )
    for name, value in posterior.mean().items():
        verdict = "identified" if identified.get(name) else "weakly constrained"
        print(f"  {name:<20} {value:.3f}  ({verdict})")
    print(f"  sim-to-real gap        {check.sim_to_real_gap:.4f}")
    print(f"  overfitted to fit set  {check.overfitted}")

    return {
        "series": reference.to_payload(),
        "observed": {k: round(v, 6) for k, v in observed.items()},
        "scales": {k: round(v, 6) for k, v in sorted(scales.items())},
        "posterior": posterior.to_payload(),
        "identified": identified,
        "predictive_check": check.to_payload(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run the complete XAI-Gov study")
    parser.add_argument(
        "--campaign",
        default="configs/campaigns/governance_value.yaml",
        help="campaign configuration file",
    )
    parser.add_argument("--replicates", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=20260703)
    parser.add_argument("--effect-size", type=float, default=0.5)
    parser.add_argument(
        "--calibration-proposals",
        type=int,
        default=600,
        help="ABC proposals; lower for a quick pass",
    )
    parser.add_argument(
        "--calibration-column",
        default=None,
        help=(
            "column to read from data/demand/demand.csv; defaults to the last "
            "numeric column, and the choice is recorded in the artifact"
        ),
    )
    parser.add_argument(
        "--allow-underpowered",
        action="store_true",
        help="proceed even when the power check fails",
    )
    parser.add_argument(
        "--skip-calibration", action="store_true", help="skip the calibration phase"
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help=(
            "console log level (default WARNING). The per-run detail always "
            "reaches the log file; the console is kept quiet so the phase "
            "summaries are readable."
        ),
    )
    parser.add_argument(
        "--amend-seal",
        action="store_true",
        help=(
            "re-seal a changed plan as an amendment, keeping the superseded seal "
            "beside it; without this a changed plan runs as exploratory"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "processes to run cells across (default 1). Cells are independent "
            "by construction, so this is safe; the run directory allocator "
            "claims names atomically."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "re-run cells that already completed. By default a campaign skips "
            "cells whose artifacts still verify, so a crash does not discard "
            "the work before it."
        ),
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    settings = load_settings(Paths(root=root))
    # The console is quiet by default and the file keeps everything. A study
    # that prints one line per shielded decision buries its own phase summaries
    # under thousands of lines, and the detail is what a log file is for.
    configure_logging(
        level=args.log_level,
        console=True,
        jsonl=True,
        log_dir=settings.paths.logs,
        filename="study.log",
        force=True,
    )

    started = datetime.now(UTC)
    print(f"XAI-Gov study  ({root})")
    print(f"started        {started.isoformat(timespec='seconds')}")

    phase(1, "Power analysis")
    power = run_power_check(args.replicates, effect_size=args.effect_size)
    if not power["adequate"] and not args.allow_underpowered:
        print(
            "\nrefusing to run an under-powered study. Raise --replicates to at "
            f"least {power['required_anytime_valid']}, or pass "
            "--allow-underpowered to proceed and have the report say so.",
            file=sys.stderr,
        )
        return 2

    campaign_path = settings.paths.resolve(args.campaign)
    config = load_config(campaign_path, root=root)
    arms_config = config.get("arms")
    if not isinstance(arms_config, dict) or not arms_config:
        print("the campaign declares no arms", file=sys.stderr)
        return 2
    arms = {str(k): settings.paths.resolve(str(v)) for k, v in arms_config.items()}
    plan = build_preregistration({k: v for k, v in config.items() if k != "arms"})

    phase(2, "Sealing the plan")
    seal_path = settings.outputs_root / "campaigns" / f"{plan.campaign}.seal.json"
    if seal_path.exists():
        _, sealed_hash = load_seal(seal_path)
        matched = sealed_hash == plan.plan_hash
        print(f"  existing seal          {sealed_hash[:16]}")
        print(f"  matches current plan   {matched}")
        if not matched and args.amend_seal:
            sealed_hash = plan.seal(seal_path, amend=True)
            print(f"  amended to             {sealed_hash[:16]}")
            print("  the superseded seal is kept in the same file")
        elif not matched:
            print("  the campaign will be reported as exploratory")
            print("  pass --amend-seal to record an amendment instead")
    else:
        sealed_hash = plan.seal(seal_path)
        print(f"  sealed                 {sealed_hash[:16]}")
    print(f"  hypotheses             {len(plan.hypotheses)}")
    print(f"  arms named by plan     {len(plan.arms())} of {len(arms)} in the campaign")
    print(f"    {', '.join(plan.arms())}")
    unnamed = sorted(set(arms) - set(plan.arms()))
    if unnamed:
        # Arms that run but that no hypothesis references. Legitimate — they
        # feed the frontier and the strategic indicators — but a line labelled
        # only "arms" invites a reader to cite the smaller number as the
        # campaign's size.
        print(f"  run but not hypothesized  {', '.join(unnamed)}")

    phase(3, f"Campaign: {len(arms)} arms x {args.replicates} replicates")
    report = run_campaign(
        plan=plan,
        arms=arms,
        settings=settings,
        project_root=root,
        replicates=args.replicates,
        base_seed=args.base_seed,
        sealed_hash=sealed_hash,
        workers=args.workers,
        resume=not args.no_resume,
    )
    campaign_payload = report.to_payload()
    summary = campaign_payload["campaign_summary"]
    print(f"  cells                  {summary['cells']}")
    print(f"  included               {summary['included']}")
    print(f"  excluded               {summary['excluded']}")
    if summary.get("reused_from_previous_run"):
        print(f"  reused                 {summary['reused_from_previous_run']}")
    multiplicity = campaign_payload.get("multiplicity", {})
    print(
        f"  thresholds             per-comparison "
        f"{multiplicity.get('per_comparison_threshold')}, familywise "
        f"{multiplicity.get('familywise_threshold')} "
        f"({multiplicity.get('hypotheses')} hypotheses)"
    )
    for result in campaign_payload["hypotheses"]:
        evidence = result.get("evidence") or {}
        if result["supported"]:
            mark = "SUPPORTED"
        elif evidence.get("refuted"):
            mark = "REFUTED"
        else:
            mark = "inconclusive"
        reverse = evidence.get("reverse_e_value")
        print(
            f"  {result['id']:<8} {mark:<12} n={result['pairs']:<3} "
            f"mean={result['mean_difference']:+.4f} "
            f"e+={evidence.get('forward_e_value', 0.0):.3g} "
            f"e-={'n/a' if reverse is None else f'{reverse:.3g}'}"
        )
    frontier = campaign_payload["frontier"]
    print(f"  frontier arms          {', '.join(frontier['frontier_arms']) or 'none'}")
    print(f"  Theorem 3 consistent   {frontier['theorem_3_consistent']}")

    # Indicators on which the arms are indistinguishable: any comparison there
    # is a comparison of seed noise, whatever a pairwise test reported.
    inert = campaign_payload.get("hierarchical", {}).get(
        "indistinguishable_indicators", []
    )
    if inert:
        print(f"  seed-dominated         {len(inert)} indicators")
    strategic = campaign_payload.get("strategic", {})
    if strategic.get("status") == "available":
        for arm, value in sorted(strategic["arms"].items()):
            vog = value["value_of_governance"]
            tvc = value["tvc"]
            print(
                f"  {arm:<20} VoG={vog:+.4f} "
                f"TVC={'undefined' if tvc is None else f'{tvc:+.4f}'}"
            )

    campaign_dir = settings.outputs_root / "campaigns" / plan.campaign
    report.write(campaign_dir / CAMPAIGN_REPORT_NAME)

    # The KPI tree of every cell, keyed by run, so the flat table can be built
    # without re-reading the run directories later.
    kpis_by_run: dict[str, dict[str, Any]] = {}
    for cell in campaign_payload["cells"]:
        run_dir = settings.runs_root / cell["run_name"]
        kpis_path = run_dir / KPIS_NAME
        if kpis_path.exists():
            kpis_by_run[cell["run_name"]] = read_json(kpis_path)

    phase(4, "Global sensitivity")
    sensitivity_config = load_config(
        settings.paths.resolve("configs/analysis/sensitivity.yaml"), root=root
    )
    sensitivity = run_sensitivity(sensitivity_config, seed=args.base_seed)

    calibration: dict[str, Any] | None = None
    if args.skip_calibration:
        phase(5, "Calibration (skipped)")
    else:
        phase(5, "Simulator calibration")
        calibration = run_calibration(
            seed=args.base_seed,
            proposals=args.calibration_proposals,
            root=root,
            column=args.calibration_column,
        )

    # Named before the bundle so the qualification reads as one sentence, and
    # so a synthetic reference cannot be described as a sim-to-real gap.
    reference_kind = (
        "synthetic"
        if (calibration or {}).get("series", {}).get("synthetic")
        else "real"
    )
    bundle = {
        "study": plan.campaign,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "replicates": args.replicates,
        "base_seed": args.base_seed,
        "power": power,
        "campaign": campaign_payload,
        "sensitivity": sensitivity,
        "calibration": calibration,
        # Read this before any number above it.
        "reporting_status": campaign_payload["reporting_status"],
        "qualifications": [
            (
                f"Every conclusion holds at the fidelity level established by the "
                f"calibration phase against a {reference_kind} reference series; "
                f"the gap is reported there."
                if calibration
                else "Calibration was skipped: no gap is established, so no "
                "conclusion transfers beyond the simulator."
            ),
            (
                "Hypotheses reported as insufficient are untested at this sample "
                "size, not refuted. The minimum detectable effect is in the power "
                "block."
            ),
            (
                "Sensitivity indices say whether a finding is about governance or "
                "about a declared cost; check 'robust_to_parameter_choice'."
            ),
        ],
    }

    bundle_path = campaign_dir / STUDY_BUNDLE_NAME
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(canonical_json(bundle) + "\n", encoding="utf-8")

    written = write_artifacts(
        bundle=bundle,
        phases={
            "01_power": power,
            "02_seal": {
                "plan": plan.to_payload(),
                "plan_hash": plan.plan_hash,
                "sealed_hash": sealed_hash,
                "seal_file": str(seal_path),
            },
            "03_campaign": campaign_payload,
            "04_sensitivity": sensitivity,
            "05_calibration": calibration or {"skipped": True},
        },
        kpis_by_run=kpis_by_run,
        directory=campaign_dir,
        project_root=root,
    )

    findings = diagnose(bundle)
    blocking = [f for f in findings if f.severity == "blocking"]

    print(f"\nstudy bundle   {bundle_path}")
    print(f"digest         {written['digest.md']}")
    print(f"tables         {written['tables.md']}")
    print(f"cell table     {written['cells.csv']}")
    figures = sorted(name for name in written if name.endswith(".svg"))
    if figures:
        print(f"figures        {len(figures)} in {campaign_dir / 'figures'}")
    print(f"diagnostics    {written['diagnostics.json']}")
    print(f"run log        {settings.paths.logs / 'study.log'}")
    print(f"status         {bundle['reporting_status']}")

    if findings:
        print(f"\ndiagnostics    {len(blocking)} blocking, "
              f"{len(findings) - len(blocking)} needing attention")
        for finding in findings:
            print(f"  [{finding.severity}] {finding.code}: {finding.detail}")
    # A blocking finding means a conclusion drawn from this run would be wrong,
    # so the exit status says so rather than leaving it to whoever reads the
    # scrollback.
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
