#!/usr/bin/env python
"""Calibrate against every real series and report the distribution of gaps.

Calibrating against one series raises a question the result cannot answer:
which one, and why that one. Choosing by best fit would be fitting the choice
to the outcome, which is the error this project has been careful about
elsewhere. Choosing arbitrarily leaves a reviewer to wonder whether another
series would have said something different.

Running all of them removes the choice. It turns "the gap is X against one
series" into "the gap is X across thirty real store-by-category series," which
is a harder claim to dismiss, and it costs thirty calibrations where each takes
seconds.

    python scripts/calibrate_all.py --cadence weekly

The cadence is a required argument with no default. It determines the
autocorrelation, which is the statistic being calibrated, so a script that
picked one silently would be making the study's modelling decision on its own.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

from xai_gov.analysis.calibration import (
    CALIBRATION_STATISTICS,
    DemandSummary,
    Prior,
    abc_rejection,
    bootstrap_scales,
    default_priors,
    posterior_predictive_check,
    prior_widths,
)
from xai_gov.analysis.data import load_series
from xai_gov.core.hashing import canonical_json


def weekly(values: list[float]) -> list[float]:
    """Sum daily values into whole weeks, discarding a partial tail.

    A trailing partial week enters the series as a low outlier and reads as a
    demand collapse, which the calibration would then try to reproduce.
    """
    return [sum(values[i : i + 7]) for i in range(0, len(values) - 6, 7)]


def normalize(values: list[float], *, target_mean: float) -> list[float]:
    """Rescale a series to the twin's operating magnitude, preserving shape.

    The twin's units are abstract: its inventory, capacity and reorder point
    sit on a scale of roughly twenty units per period, while real weekly
    store-by-category demand runs from about 1,400 to 27,500 depending on how
    many SKUs the aggregation happened to sweep up. Calibrating the absolute
    level would therefore be calibrating an artifact of the aggregation, and
    it presses the posterior against whatever prior bound it meets.

    What must transfer is the *shape* --- the coefficient of variation and the
    autocorrelation --- because those are the properties the twin's mechanism
    either can or cannot reproduce. Two series with means of 27,495 and 1,364
    and identical dynamics should calibrate to the same twin, and after
    rescaling they do.

    The rescaling is affine and mean-preserving in shape: it leaves cv and
    every autocorrelation unchanged, so no statistic the calibration compares
    is altered except the location it was never about.
    """
    mean = statistics.fmean(values)
    if mean <= 0.0:
        raise ValueError("cannot rescale a series whose mean is not positive")
    factor = target_mean / mean
    return [v * factor for v in values]


def at_prior_bound(
    posterior_mean: dict[str, float], priors: Any, *, margin: float = 0.05
) -> list[str]:
    """Parameters whose posterior mean sits against a prior boundary.

    A posterior pinned to the edge of its prior is narrow because the prior
    stopped it, not because the data constrained it, so the identifiability
    check reports it as identified and means the opposite. Naming the
    parameters lets a reader tell a calibration from a boundary artifact.
    """
    pinned: list[str] = []
    for prior in priors:
        value = posterior_mean.get(prior.name)
        if value is None:
            continue
        span = prior.high - prior.low
        if value >= prior.high - margin * span or value <= prior.low + margin * span:
            pinned.append(prior.name)
    return pinned


def calibrate_one(
    values: list[float], *, seed: int, proposals: int, process: str, drop: tuple[str, ...] = ()
) -> dict[str, Any]:
    """One series, the same procedure the study's phase 5 runs.

    ``process`` selects the generating mechanism under test. The twin's own is
    ``iid`` --- a level plus independent noise --- and ``ar1`` adds a
    persistence parameter. Testing the second here, before changing the
    simulator, is the cheap way to find out whether the missing dynamics are
    the reason every series overfits.

    ``drop`` removes statistics from the distance without removing them from
    the predictive check. The ABC distance is a mean over the statistics it
    compares, so a statistic that is a deterministic function of others
    double-counts whatever they measure: `cv = std/mean` adds no information
    and shifts the distance toward the moments, which is how a dynamics
    parameter ends up unidentified while the moments are pinned. Dropping it
    from the acceptance decision is a change to the metric, not to the model.
    """
    base_summary = DemandSummary()

    def summarize(series: list[float]) -> dict[str, float]:
        stats = base_summary(series)
        return {k: v for k, v in stats.items() if k not in drop}

    priors = list(default_priors())
    if process == "ar1":
        # Bounded away from 1: a unit root is not a stationary demand process,
        # and a prior that admits one lets the sampler wander into series whose
        # summary statistics do not converge.
        priors.append(Prior(name="persistence", low=0.0, high=0.97))
    observed = summarize(values)
    scales = bootstrap_scales(values, summarize, seed=seed)

    def simulator(parameters: dict[str, float], sim_seed: int) -> list[float]:
        rng = np.random.default_rng(sim_seed)
        base = parameters["base_demand"]
        noise = parameters["noise_scale"]
        if process == "iid":
            return [
                float(max(0.0, rng.normal(base, noise))) for _ in range(len(values))
            ]
        # AR(1) about the base level. The innovation variance is scaled by
        # sqrt(1 - rho^2) so that the *stationary* standard deviation is
        # `noise` regardless of persistence: without that, raising persistence
        # would inflate the dispersion too, and the calibration could not tell
        # which parameter the data was asking it to move.
        rho = parameters.get("persistence", 0.0)
        innovation = noise * (1.0 - rho * rho) ** 0.5
        out: list[float] = []
        deviation = 0.0
        for _ in range(len(values)):
            deviation = rho * deviation + float(rng.normal(0.0, innovation))
            out.append(float(max(0.0, base + deviation)))
        return out

    posterior = abc_rejection(
        simulator=simulator, summarize=summarize, observed=observed, priors=priors,
        proposals=proposals, quantile=0.05, seed=seed, scales=scales,
    )
    check = posterior_predictive_check(
        simulator=simulator, summarize=summarize, observed=observed,
        posterior=posterior, calibrated_statistics=CALIBRATION_STATISTICS,
        draws=60, seed=seed + 1, scales=scales,
    )
    identified = posterior.identified(prior_widths(priors))
    means = {k: round(v, 6) for k, v in posterior.mean().items()}
    pinned = at_prior_bound(means, priors)
    # Draws per dimension. A posterior can be wide for two opposite reasons and
    # the width test cannot tell them apart: the data may genuinely fail to
    # constrain the parameter, or the accepted sample may be too thin to
    # resolve the dimension it lives in. They call for opposite responses ---
    # respecify the model, or raise the proposal budget --- so the ratio is
    # reported rather than left for a reader to infer from the parameter count.
    per_dimension = posterior.accepted / max(len(priors), 1)
    return {
        "process": process,
        "dropped_statistics": list(drop),
        "observed": {k: round(v, 6) for k, v in observed.items()},
        "posterior_mean": means,
        "identified": identified,
        "at_prior_bound": pinned,
        "accepted_draws": posterior.accepted,
        "draws_per_parameter": round(per_dimension, 2),
        "all_identified": all(identified.values()) and not pinned,
        "sim_to_real_gap": check.sim_to_real_gap,
        "overfitted": check.overfitted,
        "predictive_check": check.to_payload(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cadence",
        required=True,
        choices=("daily", "weekly"),
        help=(
            "required and with no default: the cadence determines the "
            "autocorrelation being calibrated, so it is a preregistered "
            "modelling choice rather than a script's convenience"
        ),
    )
    parser.add_argument("--dir", type=Path, default=Path("data/demand"))
    parser.add_argument("--out", type=Path, default=Path("outputs/calibration"))
    parser.add_argument("--proposals", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument(
        "--periods",
        type=int,
        default=0,
        help="truncate each series to this many periods (0 keeps all)",
    )
    parser.add_argument(
        "--target-mean",
        type=float,
        default=22.0,
        help=(
            "rescale each series to this mean before calibrating. The twin's "
            "units are abstract and real volumes vary twentyfold with the "
            "aggregation, so the absolute level is not what transfers; shape "
            "is. Pass 0 to calibrate raw levels and expect the posterior to "
            "press against the prior bound."
        ),
    )
    parser.add_argument(
        "--process",
        default="iid",
        choices=("iid", "ar1", "both"),
        help=(
            "generating process to calibrate. 'iid' is the twin's own level "
            "plus independent noise; 'ar1' adds a persistence parameter. "
            "'both' reports the pair, which is what tells you whether missing "
            "dynamics explain an overfit rather than leaving it to a guess."
        ),
    )
    parser.add_argument(
        "--drop-redundant",
        action="store_true",
        help=(
            "exclude cv from the ABC distance. cv is std/mean, a deterministic "
            "function of two statistics already compared, so it double-counts "
            "the moments and dilutes the single statistic that carries "
            "dynamics. Use this to test whether an unidentified persistence is "
            "a property of the metric rather than of the model."
        ),
    )
    args = parser.parse_args(argv)

    paths = sorted(args.dir.glob("m5_*.csv"))
    if not paths:
        print(
            f"no M5 series in {args.dir}; run `make m5 M5_SOURCE=...` first",
            file=sys.stderr,
        )
        return 1

    drop = ("cv",) if args.drop_redundant else ()
    processes = ("iid", "ar1") if args.process == "both" else (args.process,)
    print(f"calibrating {len(paths)} series at {args.cadence} cadence")
    if args.target_mean > 0:
        print(
            f"series rescaled to mean {args.target_mean:g}; shape is preserved and "
            "the absolute level is not calibrated"
        )
    print(f"process    {', '.join(processes)}")
    if drop:
        print(
            f"distance   excludes {', '.join(drop)} --- redundant given the "
            "statistics already compared"
        )
    header = f"{'series':<22}"
    for process in processes:
        header += f"{process + ' gap':>12}{process + ' loose':>14}"
    print(header)
    print("-" * len(header))

    results: dict[str, Any] = {}
    for path in paths:
        values = list(load_series(path).values)
        if args.cadence == "weekly":
            values = weekly(values)
        if args.periods:
            values = values[: args.periods]
        if args.target_mean > 0:
            values = normalize(values, target_mean=args.target_mean)
        name = path.stem.replace("m5_", "").replace("_daily", "")

        per_process: dict[str, Any] = {}
        line = f"{name:<22}"
        for process in processes:
            outcome = calibrate_one(
                values, seed=args.seed, proposals=args.proposals, process=process,
                drop=drop,
            )
            per_process[process] = outcome
            flag = "ok" if not outcome["overfitted"] else "OVERFIT"
            if not outcome["all_identified"]:
                # Name the parameter, not just the failure. "unident" tells a
                # reader that something is wide; which parameter is wide is
                # what says whether the model or the distance metric is at
                # fault, and those call for different fixes.
                loose = [k for k, ok in outcome["identified"].items() if not ok]
                loose += [k for k in outcome["at_prior_bound"] if k not in loose]
                flag = ",".join(k[:4] for k in loose) or "unident"
            line += f"{outcome['sim_to_real_gap']:>12.4f}{flag:>14}"
        print(line)
        results[name] = {
            "periods": len(values),
            "cadence": args.cadence,
            "processes": per_process,
        }

    print("-" * len(header))
    per_process_summary: dict[str, Any] = {}
    for process in processes:
        gaps = sorted(r["processes"][process]["sim_to_real_gap"] for r in results.values())
        identified = sum(
            1 for r in results.values() if r["processes"][process]["all_identified"]
        )
        overfitted = sum(
            1 for r in results.values() if r["processes"][process]["overfitted"]
        )
        pinned = sum(
            1 for r in results.values() if r["processes"][process]["at_prior_bound"]
        )
        median = statistics.median(gaps)
        print(
            f"{process:<10} gap min {gaps[0]:.4f}  median {median:.4f}  max {gaps[-1]:.4f}"
            f"   identified {identified}/{len(gaps)}   overfit {overfitted}/{len(gaps)}"
        )
        draws = statistics.median(
            r["processes"][process]["draws_per_parameter"] for r in results.values()
        )
        # Which parameter is wide, across series. A single parameter loose
        # everywhere points at the distance metric or the model; a different
        # one each time points at the sample.
        loose_counts: dict[str, int] = {}
        for r in results.values():
            outcome = r["processes"][process]
            for key, ok in outcome["identified"].items():
                if not ok:
                    loose_counts[key] = loose_counts.get(key, 0) + 1
            for key in outcome["at_prior_bound"]:
                loose_counts[key] = loose_counts.get(key, 0) + 1
        if loose_counts:
            ranked = sorted(loose_counts.items(), key=lambda kv: -kv[1])
            detail = ", ".join(f"{k} in {v}/{len(gaps)}" for k, v in ranked)
            print(f"{'':10} wide: {detail}")
            per_process_summary.setdefault("_loose", {})[process] = dict(ranked)
        if identified < len(gaps) and pinned == 0 and draws < 30:
            print(
                f"{'':10} {draws:.0f} accepted draws per parameter --- too thin to"
                " resolve the posterior."
            )
            print(
                f"{'':10} Raise --proposals before concluding the parameters are"
                " unidentifiable."
            )
        per_process_summary[process] = {
            "gap": {
                "min": round(gaps[0], 6),
                "median": round(median, 6),
                "max": round(gaps[-1], 6),
            },
            "identified_series": identified,
            "overfitted_series": overfitted,
            "at_prior_bound_series": pinned,
            "median_draws_per_parameter": round(
                statistics.median(
                    r["processes"][process]["draws_per_parameter"]
                    for r in results.values()
                ),
                2,
            ),
        }

    if len(processes) == 2:
        iid = per_process_summary["iid"]["gap"]["median"]
        ar1 = per_process_summary["ar1"]["gap"]["median"]
        ar1_overfit = per_process_summary["ar1"]["overfitted_series"]
        print()
        if ar1 < iid * 0.6 and ar1_overfit < len(results) // 2:
            print(
                f"Persistence closes most of the gap ({iid:.2f} to {ar1:.2f} median) and\n"
                f"leaves {ar1_overfit} of {len(results)} series overfitted. The twin's\n"
                "level-plus-noise demand process is the deficiency, and adding an\n"
                "AR(1) regime to the simulator is justified by this comparison rather\n"
                "than assumed."
            )
            ar1_identified = per_process_summary["ar1"]["identified_series"]
            if ar1_identified < len(results):
                print(
                    f"\nNote the trade: iid is identified and wrong, ar1 fits and is not\n"
                    f"identified ({ar1_identified} of {len(results)}). Adding a parameter\n"
                    "bought fit and cost precision. Whether the cost is structural or an\n"
                    "artifact of the proposal budget is answerable --- rerun with more\n"
                    "proposals and see whether the posteriors narrow."
                )
        elif ar1 < iid:
            print(
                f"Persistence narrows the gap ({iid:.2f} to {ar1:.2f} median) without\n"
                f"clearing it: {ar1_overfit} of {len(results)} series still overfit. Some\n"
                "of the missing dynamics is first-order persistence and some is not;\n"
                "inspect which held-out statistic still fails before changing the twin."
            )
        else:
            print(
                f"Persistence does not help ({iid:.2f} to {ar1:.2f} median). The\n"
                "deficiency is not first-order autocorrelation, and adding an AR(1)\n"
                "regime to the twin would have been a guess."
            )

    summary = {
        "cadence": args.cadence,
        "target_mean": args.target_mean,
        "dropped_statistics": list(drop),
        "series_count": len(results),
        "proposals": args.proposals,
        "seed": args.seed,
        "by_process": per_process_summary,
        "series": results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / (
        f"calibration_{args.cadence}_{args.process}"
        f"{'_nocv' if drop else ''}.json"
    )
    path.write_text(canonical_json(summary) + "\n", encoding="utf-8")
    print(f"\nreport {path}")
    print(
        "Record the cadence and the generating process in the sealed plan "
        "before the confirmatory campaign reads this."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
