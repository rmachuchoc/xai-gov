"""Command-line entry point.

Commands:

    xai-gov info                      resolved paths, settings, versions
    xai-gov config <path>             print a fully resolved configuration
    xai-gov factors                   demand regimes, policies, governance arms
    xai-gov run --experiment <path>   execute one experiment
    xai-gov kpis <run>                print the indicator layers of a run
    xai-gov runs                      list runs with their chain status
    xai-gov verify-log <run|file>     verify a hash-chained decision log
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from xai_gov import __version__
from xai_gov.core.logging import configure_logging, get_logger
from xai_gov.core.settings import Settings, load_settings
from xai_gov.io.hashchain import ChainVerificationError, verify_file
from xai_gov.io.writers import DECISION_LOG_NAME, KPIS_NAME, MANIFEST_NAME, read_json
from xai_gov.io.yaml_loader import ConfigError, load_config

_LOG = get_logger("cli")


def _bootstrap() -> Settings:
    settings = load_settings()
    configure_logging(
        level=settings.logging.level,
        console=settings.logging.console,
        jsonl=settings.logging.jsonl,
        log_dir=settings.paths.logs,
        filename=settings.logging.filename,
    )
    return settings


# -- commands -------------------------------------------------------------
def cmd_info(_: argparse.Namespace, settings: Settings) -> int:
    print(f"xai-gov {__version__}   python {sys.version.split()[0]}")
    print(f"root      {settings.paths.root}")
    print(f"configs   {settings.paths.configs}")
    print(f"outputs   {settings.outputs_root}")
    print(f"runs      {settings.runs_root}")
    print(f"logs      {settings.paths.logs}")
    runtime = settings.runtime
    print(
        f"runtime   strict={runtime.strict} workers={runtime.max_workers} "
        f"events_csv={runtime.write_events_csv} decision_log={runtime.write_decision_log}"
    )
    tracking = settings.tracking
    print(f"tracking  enabled={tracking.enabled} backend={tracking.backend}")
    return 0


def cmd_config(args: argparse.Namespace, settings: Settings) -> int:
    from xai_gov.core.hashing import canonical_json, content_hash

    target = settings.paths.resolve(args.path)
    resolved = load_config(target, root=settings.paths.root)
    print(canonical_json(resolved))
    print(f"# config_hash {content_hash(resolved)}", file=sys.stderr)
    return 0


def cmd_verify_log(args: argparse.Namespace, settings: Settings) -> int:
    target = Path(args.target)
    if not target.is_absolute():
        candidate = settings.runs_root / args.target
        target = candidate if candidate.exists() else settings.paths.resolve(args.target)
    if target.is_dir():
        target = target / DECISION_LOG_NAME
    if not target.is_file():
        print(f"no decision log at {target}", file=sys.stderr)
        return 2
    try:
        count = verify_file(target)
    except ChainVerificationError as error:
        print(f"FAIL  {target}\n      {error}", file=sys.stderr)
        return 1
    print(f"OK    {target}  ({count} entries verified)")
    return 0


def cmd_runs(_: argparse.Namespace, settings: Settings) -> int:
    if not settings.runs_root.is_dir():
        print("no runs yet")
        return 0
    rows: list[tuple[str, str, str, str]] = []
    for directory in sorted(settings.runs_root.iterdir()):
        if not directory.is_dir():
            continue
        manifest_path = directory / MANIFEST_NAME
        if not manifest_path.is_file():
            rows.append((directory.name, "-", "no manifest", "-"))
            continue
        manifest = read_json(manifest_path)
        rows.append(
            (
                directory.name,
                str(manifest.get("decisions", "-")),
                str(manifest.get("status", "-")),
                str(manifest.get("chain_head", ""))[:12],
            )
        )
    if not rows:
        print("no runs yet")
        return 0
    width = max(len(row[0]) for row in rows)
    print(f"{'run'.ljust(width)}  {'dec.':>5}  status                chain")
    for name, decisions, status, head in rows:
        print(f"{name.ljust(width)}  {decisions:>5}  {status[:20]:<20}  {head}")
    return 0


def cmd_factors(_: argparse.Namespace, __: Settings) -> int:
    from xai_gov.causal.descriptor import available_descriptors
    from xai_gov.conformal.scores import available_scores
    from xai_gov.governance.autonomy import available_regimes as available_autonomy
    from xai_gov.governance.registry import available_architectures
    from xai_gov.oversight.delegation import available_deferral_policies
    from xai_gov.policies.registry import available_policies
    from xai_gov.simulation.demand import available_regimes
    from xai_gov.verification.specification import available_predicates

    print("demand regimes        " + ", ".join(available_regimes()))
    print("policies             " + ", ".join(available_policies()))
    print("governance arms      " + ", ".join(available_architectures()))
    print("autonomy regimes     " + ", ".join(available_autonomy()))
    print("conformal schemes    aci, fixed")
    print("nonconformity scores " + ", ".join(available_scores()))
    print("descriptors          " + ", ".join(available_descriptors()))
    print("deferral policies    " + ", ".join(available_deferral_policies()))
    print("safety predicates    " + ", ".join(available_predicates()))
    print("shield               on, off")
    print()
    print("scheduled for later stages:")
    print("  policies           marl_heterogeneous, llm_agent (stage 6)")
    print("  governance         federated (stage 6)")
    return 0


def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    from xai_gov.orchestration.runner import plan_experiment, run_experiment

    target = settings.paths.resolve(args.experiment)
    # Planning and execution are reported separately. A rejected
    # configuration and a run that broke mid-flight are different failures,
    # and conflating them sends the reader to the wrong file.
    try:
        plan_experiment(target, settings=settings)
    except (ValueError, KeyError) as error:
        print(f"invalid experiment configuration: {error}", file=sys.stderr)
        return 2
    try:
        manifest = run_experiment(target, settings=settings)
    except NotImplementedError as error:
        print(f"not available yet: {error}", file=sys.stderr)
        return 3
    except (ValueError, KeyError) as error:
        print(f"run failed: {error}", file=sys.stderr)
        print("  the run was aborted; its manifest carries an 'aborted' status", file=sys.stderr)
        return 1

    print(f"run       {manifest['run_name']}")
    print(f"status    {manifest['status']}")
    print(f"decisions {manifest['decisions']}")
    print(f"chain     {manifest['chain_head'][:16]}  ({manifest['chain_entries']} entries)")
    print(f"config    {manifest['config_hash'][:16]}")
    print(f"artifacts {settings.runs_root / manifest['run_name']}")
    return 0 if manifest["status"] == "completed" else 1


def cmd_kpis(args: argparse.Namespace, settings: Settings) -> int:
    directory = settings.runs_root / args.run
    if not directory.is_dir():
        directory = settings.paths.resolve(args.run)
    path = directory / KPIS_NAME
    if not path.is_file():
        print(f"no KPI file at {path}", file=sys.stderr)
        return 2
    layers = read_json(path)
    for name, layer in layers.items():
        if not isinstance(layer, dict):
            continue
        status = str(layer.get("status", "unknown"))
        print(f"\n{name}  [{status}]")
        if status == "not_available":
            print(f"  requires: {layer.get('requires')}")
            continue
        for key, value in layer.items():
            if key in ("status", "requires"):
                continue
            print(f"  {key:<28} {value}")
    return 0


def _load_campaign(path: Path, settings: Settings) -> tuple[Any, dict[str, Path]]:
    """Read a campaign file into a plan and its arm paths."""
    from xai_gov.analysis.preregistration import build_preregistration

    config = load_config(path, root=settings.paths.root)
    arms_config = config.get("arms")
    if not isinstance(arms_config, dict) or not arms_config:
        raise ValueError("a campaign must declare a non-empty 'arms' mapping")
    arms = {
        str(name): settings.paths.resolve(str(target))
        for name, target in arms_config.items()
    }
    plan_config = {k: v for k, v in config.items() if k != "arms"}
    return build_preregistration(plan_config), arms


def _seal_path(settings: Settings, campaign: str) -> Path:
    return settings.outputs_root / "campaigns" / f"{campaign}.seal.json"


def cmd_seal(args: argparse.Namespace, settings: Settings) -> int:
    """Seal a plan before any run touches it."""
    target = settings.paths.resolve(args.campaign)
    try:
        plan, arms = _load_campaign(target, settings)
    except (ValueError, KeyError) as error:
        print(f"invalid campaign: {error}", file=sys.stderr)
        return 2

    missing = sorted(set(plan.arms()) - set(arms))
    if missing:
        print(f"the plan names arms with no experiment file: {missing}", file=sys.stderr)
        return 2

    path = _seal_path(settings, plan.campaign)
    try:
        digest = plan.seal(path)
    except FileExistsError as error:
        print(str(error), file=sys.stderr)
        return 1

    print(f"campaign   {plan.campaign}")
    print(f"hypotheses {len(plan.hypotheses)}")
    print(f"arms       {', '.join(plan.arms())}")
    print(f"seal       {digest}")
    print(f"written    {path}")
    return 0


def cmd_campaign(args: argparse.Namespace, settings: Settings) -> int:
    """Execute a campaign and write its report."""
    from xai_gov.analysis.preregistration import load_seal
    from xai_gov.analysis.report import CAMPAIGN_REPORT_NAME, run_campaign

    target = settings.paths.resolve(args.campaign)
    try:
        plan, arms = _load_campaign(target, settings)
    except (ValueError, KeyError) as error:
        print(f"invalid campaign: {error}", file=sys.stderr)
        return 2

    seal_file = _seal_path(settings, plan.campaign)
    sealed_hash: str | None = None
    if seal_file.exists():
        _, sealed_hash = load_seal(seal_file)
    else:
        # Not an error: an exploratory campaign is legitimate. But it must be
        # labelled as one, here and in the report, rather than presented as a
        # confirmatory result nobody committed to in advance.
        print(
            f"no seal at {seal_file}; this campaign will be reported as exploratory",
            file=sys.stderr,
        )

    try:
        report = run_campaign(
            plan=plan,
            arms=arms,
            settings=settings,
            project_root=settings.paths.root,
            replicates=args.replicates,
            base_seed=args.base_seed,
            sealed_hash=sealed_hash,
        )
    except (ValueError, KeyError) as error:
        print(f"campaign failed: {error}", file=sys.stderr)
        return 1

    path = report.write(
        settings.outputs_root / "campaigns" / plan.campaign / CAMPAIGN_REPORT_NAME
    )
    payload = report.to_payload()
    summary = payload["campaign_summary"]

    print(f"campaign   {plan.campaign}")
    print(f"status     {payload['reporting_status']}")
    print(f"cells      {summary['included']} included, {summary['excluded']} excluded")
    for result in payload["hypotheses"]:
        mark = "supported" if result["supported"] else "insufficient"
        print(
            f"  {result['id']:<8} {mark:<12} "
            f"n={result['pairs']:<3} mean={result['mean_difference']:+.4f}"
        )
    frontier = payload["frontier"]
    if frontier["frontier_arms"]:
        print(f"frontier   {', '.join(frontier['frontier_arms'])}")
    print(f"report     {path}")
    return 0


# -- parser ---------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xai-gov",
        description="Agent-governed digital twin for supply chains.",
    )
    parser.add_argument("--version", action="version", version=f"xai-gov {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="show resolved paths and settings").set_defaults(func=cmd_info)

    config_parser = sub.add_parser("config", help="print a fully resolved configuration")
    config_parser.add_argument("path", help="path to a YAML configuration file")
    config_parser.set_defaults(func=cmd_config)

    verify_parser = sub.add_parser("verify-log", help="verify a hash-chained decision log")
    verify_parser.add_argument("target", help="run name, run directory, or JSONL file")
    verify_parser.set_defaults(func=cmd_verify_log)

    sub.add_parser("runs", help="list runs and their chain status").set_defaults(func=cmd_runs)

    sub.add_parser("factors", help="list available experimental factors").set_defaults(
        func=cmd_factors
    )

    run_parser = sub.add_parser("run", help="execute an experiment")
    run_parser.add_argument("--experiment", required=True, help="experiment configuration file")
    run_parser.set_defaults(func=cmd_run)

    kpis_parser = sub.add_parser("kpis", help="print the indicator layers of a run")
    kpis_parser.add_argument("run", help="run name or run directory")
    kpis_parser.set_defaults(func=cmd_kpis)

    seal_parser = sub.add_parser(
        "seal", help="seal a campaign plan before running it"
    )
    seal_parser.add_argument("campaign", help="campaign configuration file")
    seal_parser.set_defaults(func=cmd_seal)

    campaign_parser = sub.add_parser("campaign", help="execute a preregistered campaign")
    campaign_parser.add_argument("campaign", help="campaign configuration file")
    campaign_parser.add_argument(
        "--replicates", type=int, default=5, help="replicates per arm (default 5)"
    )
    campaign_parser.add_argument(
        "--base-seed", type=int, default=20260703, help="seed the replicates derive from"
    )
    campaign_parser.set_defaults(func=cmd_campaign)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = _bootstrap()
        return int(args.func(args, settings))
    except ConfigError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
