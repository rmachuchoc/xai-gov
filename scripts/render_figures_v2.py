#!/usr/bin/env python
"""Figures of the v2 results, read from outputs/v2 (step 5 of the revision).

    python scripts/render_figures_v2.py --outputs-root outputs/v2

Writes a PNG at 600 dpi and a CSV table view of every figure to
<outputs-root>/figures/. Nothing is hard-coded: each value is read from the
campaign reports, the stored cells, the sweeps, the power simulation and the
null line. The manuscript is not touched.

Colour: one hue for single-measure figures; the reference palette's first slots
for the eps_t nodes (three, validated all-pairs) and the power curves (six,
validated adjacent), each line direct-labelled. Grey for reference lines.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

try:
    import matplotlib
except ImportError:  # pragma: no cover - environment guidance
    print("matplotlib is required: pip install -e '.[figures]'", file=sys.stderr)
    raise SystemExit(2) from None

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from xai_gov.analysis.posthoc_v2 import load_cells

TEXT_WIDTH_IN = 170.0 / 25.4
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"
REFERENCE = "#8a8984"
DPI = 600
#: Every label drawn inside a plot sits on a patch of the surface colour, so no
#: gridline, reference line or curve runs through its text.
LABEL_BOX = {"facecolor": "#fcfcfb", "edgecolor": "none", "pad": 1.2}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
ONE = SERIES[0]

CONFIRMATORY = "campaigns/confirmatory_v2/confirmatory_v2_report.json"
DETECTOR = "campaigns/detector_exploratory_v2/detector_exploratory_v2_report.json"
ARM_ORDER = ["ungoverned", "governed", "fallback", "projection", "backup",
             "q_fallback", "q_projection", "five_fallback", "five_projection"]


def style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.axisbelow": True,
    })


def save(fig: Any, directory: Path, name: str, header: list[str], rows: list[list[Any]]) -> None:
    fig.savefig(directory / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    with (directory / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def sci(value: float, digits: int = 1) -> str:
    """A number in scientific notation written as m × 10^n, never m e n."""
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10**exponent
    if abs(exponent) < 3:
        return f"{value:.{max(digits, 2)}f}"
    return rf"${mantissa:.{digits}f}\times10^{{{exponent}}}$"


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    half = 1.96 * statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
    return mean, mean - half, mean + half


def dot_whisker(ax: Any, labels: list[str], means: list[float], lows: list[float],
                highs: list[float]) -> None:
    y = np.arange(len(labels))[::-1]
    ax.hlines(y, lows, highs, color=ONE, linewidth=1.6)
    ax.plot(means, y, "o", color=ONE, markersize=4.5, markeredgecolor=SURFACE,
            markeredgewidth=1.0, zorder=3)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)


# -- figures --------------------------------------------------------------------
def fig_beta(cells: tuple[Any, ...], out: Path) -> None:
    rows = []
    for arm in ARM_ORDER:
        values = [r.indicator("operational.fill_rate") for r in cells
                  if r.included and r.cell.arm == arm]
        values = [v for v in values if v is not None]
        if values:
            m, lo, hi = mean_ci(values)
            rows.append([arm, len(values), round(m, 4), round(lo, 4), round(hi, 4)])
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN * 0.62, 2.8))
    dot_whisker(ax, [r[0] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
                [r[4] for r in rows])
    for i, r in enumerate(rows[::-1]):
        ax.annotate(f"{r[2]:.3f}", (r[4], i), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK_2, bbox=LABEL_BOX, zorder=4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fill rate β at the retailer (mean, 95% CI over 500 replicates)")
    ax.set_title("Fill rate by arm", loc="left")
    save(fig, out, "fig_v2_beta_by_arm", ["arm", "runs", "mean", "ci_low", "ci_high"], rows)


def fig_evidence(report: dict[str, Any], out: Path) -> None:
    rows = []
    for h in report["hypotheses"]:
        cs = h["confidence_sequence"]
        rows.append([h["id"], f"{h['treatment']} − {h['control']}", h["direction"],
                     round(h["mean_difference"], 4), round(cs["lower"], 4),
                     round(cs["upper"], 4), h["evidence"]["forward_e_value"], h["verdict"]])
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN * 0.75, 2.6))
    labels = [f"{r[0]}  {r[1]}" for r in rows]
    dot_whisker(ax, labels, [r[3] for r in rows], [r[4] for r in rows], [r[5] for r in rows])
    ax.axvline(0, color=REFERENCE, linewidth=0.8)
    for i, r in enumerate(rows[::-1]):
        ax.annotate(f"e = {sci(r[6])}", (max(r[5], 0.0), i), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=7, color=INK_2,
                    bbox=LABEL_BOX, zorder=4)
    ax.set_xlim(-0.9, 1.05)
    ax.set_xlabel("paired difference in β (mean, 95% confidence sequence); threshold e ≥ 240")
    ax.set_title("Confirmatory hypotheses H1–H6", loc="left")
    save(fig, out, "fig_v2_evidence", ["hypothesis", "comparison", "direction",
                                         "mean_difference", "cs_low", "cs_high", "e_value",
                                         "verdict"], rows)


def fig_epsilon(cells: tuple[Any, ...], out: Path, arm: str = "projection") -> None:
    by_node: dict[str, dict[int, list[float]]] = {}
    for r in cells:
        if not r.included or r.cell.arm != arm:
            continue
        trajectory = ((r.kpis.get("guarantees") or {}).get("epsilon") or {}).get("trajectory")
        for node, points in (trajectory or {}).items():
            for period, level in points:
                by_node.setdefault(node, {}).setdefault(int(period), []).append(float(level))
    nodes = sorted(by_node.items())
    fig, axes = plt.subplots(1, len(nodes), figsize=(TEXT_WIDTH_IN, 2.3), sharey=True)
    rows = []
    for ax, (node, periods) in zip(np.atleast_1d(axes), nodes, strict=True):
        t = sorted(periods)
        median = [float(np.median(periods[p])) for p in t]
        low = [float(np.quantile(periods[p], 0.1)) for p in t]
        high = [float(np.quantile(periods[p], 0.9)) for p in t]
        ax.fill_between(t, low, high, color=ONE, alpha=0.18, linewidth=0)
        ax.plot(t, median, color=ONE, linewidth=1.5)
        ax.axhline(0.10, color=REFERENCE, linewidth=0.8)
        ax.set_title(node, loc="left")
        ax.set_xlabel("period")
        rows += [[node, p, round(m, 5), round(lo, 5), round(hi, 5)]
                 for p, m, lo, hi in zip(t, median, low, high, strict=True)]
    first = np.atleast_1d(axes)[0]
    first.set_ylabel("adapted level εₜ")
    fig.suptitle(f"Adapted level εₜ by node ({arm}; median and 10–90% over runs; "
                 "grey line: target ε = 0.10)",
                 x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    save(fig, out, "fig_v2_epsilon_trajectory", ["node", "period", "median", "p10", "p90"], rows)


def fig_envelope(report: dict[str, Any], out: Path) -> None:
    envelope = report["envelope"]
    arms = [a for a in ARM_ORDER if a in envelope and envelope[a].get("status") == "available"]
    properties = sorted(envelope[arms[0]]["properties"])
    fig, axes = plt.subplots(1, len(properties), figsize=(TEXT_WIDTH_IN, 3.0), sharey=True,
                             gridspec_kw={"wspace": 0.25})
    rows = []
    for ax, prop in zip(axes, properties, strict=True):
        entries = [envelope[a]["properties"][prop] for a in arms]
        dot_whisker(ax, arms, [e["frequency"] for e in entries], [e["ci_low"] for e in entries],
                    [e["ci_high"] for e in entries])
        bound = entries[0]["bound"]
        ax.axvline(bound, color=REFERENCE, linewidth=0.8)
        ax.set_xlim(0, 1)
        ax.set_title(f"{prop} · bound {bound:g}", loc="left")
        rows += [[a, prop, e["frequency"], e["ci_low"], e["ci_high"], bound, e["verdict"]]
                 for a, e in zip(arms, entries, strict=True)]
    axes[1].set_xlabel("violation frequency (one trial per replicate; Clopper–Pearson 95%)")
    fig.suptitle("Safety properties by arm (grey line: the property's bound)",
                 x=0.01, y=1.04, ha="left", fontsize=9)
    save(fig, out, "fig_v2_envelope", ["arm", "property", "frequency", "ci_low", "ci_high",
                                         "bound", "verdict"], rows)


def fig_sweeps(sweeps: dict[str, Any], out: Path) -> None:
    names = [n for n in ("break_even", "severity") if n in sweeps["sweeps"]]
    fig, axes = plt.subplots(1, len(names), figsize=(TEXT_WIDTH_IN, 2.4))
    axes = np.atleast_1d(axes)
    rows = []
    for ax, name in zip(axes, names, strict=True):
        sweep = sweeps["sweeps"][name]
        levels = [lv for lv in sweep["levels"] if lv["ci95"]]
        x = np.arange(len(levels))
        means = [lv["difference"] for lv in levels]
        lows = [lv["ci95"][0] for lv in levels]
        highs = [lv["ci95"][1] for lv in levels]
        ax.vlines(x, lows, highs, color=ONE, linewidth=1.6)
        ax.plot(x, means, "o", color=ONE, markersize=4.5, markeredgecolor=SURFACE,
                markeredgewidth=1.0, zorder=3)
        ax.axhline(0, color=REFERENCE, linewidth=0.8)
        ax.set_xticks(x, [lv["label"].split("=")[-1] for lv in levels])
        ax.set_xlabel(sweep["parameter"])
        ax.set_title({"break_even": "Unit intervention cost",
                      "severity": "Disruption severity"}[name], loc="left")
        ax.grid(axis="x", visible=False)
        rows += [[name, lv["label"], lv["difference"], lv["ci95"][0], lv["ci95"][1]]
                 for lv in levels]
    axes[0].set_ylabel("net value difference")
    comparison = sweeps["sweeps"][names[0]]["comparison"]
    fig.suptitle(f"Projection shield minus mediation alone ({comparison}); "
                 "mean and 95% bootstrap CI",
                 x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    save(fig, out, "fig_v2_sweeps", ["sweep", "level", "difference", "ci_low", "ci_high"], rows)


def fig_power(power: dict[str, Any], out: Path, sealed_n: int) -> None:
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN * 0.75, 2.6))
    rows = []
    for color, h in zip(SERIES, power["hypotheses"], strict=False):
        cases = [c for g in h["generators"].values() for c in g["by_shift"]]
        n_values = sorted(int(n) for n in cases[0]["curve"])
        worst = [min(c["curve"][str(n)] for c in cases) for n in n_values]
        ax.plot(n_values, worst, color=color, linewidth=1.5, label=h["hypothesis"])
        rows += [[h["hypothesis"], n, p] for n, p in zip(n_values, worst, strict=True)]
    ax.axhline(0.80, color=REFERENCE, linewidth=0.8)
    ax.axvline(sealed_n, color=REFERENCE, linewidth=0.8)
    ax.annotate(f"sealed n = {sealed_n}", (sealed_n, 0.06), xytext=(5, 0),
                textcoords="offset points", fontsize=7, color=INK_2, va="center",
                bbox=LABEL_BOX, zorder=4)
    ax.annotate("power 0.80", (10, 0.80), xytext=(0, 6), textcoords="offset points",
                fontsize=7, color=INK_2, va="bottom", bbox=LABEL_BOX, zorder=4)
    ax.set_xlim(0, 700)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("replicates n")
    ax.set_ylabel("simulated power")
    ax.set_title("Power of aGRAPA at threshold 240, Δβ = 0.05 (worst generator and direction)",
                 loc="left", fontsize=8)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1)
    save(fig, out, "fig_v2_power", ["hypothesis", "n", "power"], rows)


def fig_detector(cells: tuple[Any, ...], null: dict[str, Any] | None, out: Path) -> None:
    order = ["shielded", "contaminated", "long_memory", "reference_scale", "governed_split"]
    rows = []
    for arm in order:
        values = [r.indicator("guarantees.coverage_error") for r in cells
                  if r.included and r.cell.arm == arm]
        values = [v for v in values if v is not None]
        if values:
            m, lo, hi = mean_ci(values)
            rows.append([arm, len(values), round(m, 4), round(lo, 4), round(hi, 4)])
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN * 0.62, 2.2))
    dot_whisker(ax, [r[0] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
                [r[4] for r in rows])
    ax.axvline(0, color=REFERENCE, linewidth=0.8)
    ax.set_xlabel("coverage error: realized flag rate − 0.10 (mean, 95% CI)")
    title = "Detector calibration against the fixed target (exploratory)"
    if null:
        alarms = [f"{a}: {e['false_alarms_demand_facing']['runs_with_alarm']}/"
                  f"{e['false_alarms_demand_facing']['runs']}" for a, e in null["arms"].items()]
        ax.annotate("null line false alarms  " + ", ".join(alarms), (0.0, -0.32),
                    xycoords="axes fraction", fontsize=7, color=INK_2)
    ax.set_title(title, loc="left")
    save(fig, out, "fig_v2_detector_coverage", ["arm", "runs", "mean", "ci_low", "ci_high"],
         rows)


def fig_network(root: Path, out: Path) -> None:
    """Figure 1: the serial chain as simulated, drawn from its configuration."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    from xai_gov.io.yaml_loader import load_config

    config = load_config(root / "configs/network/three_echelon.yaml", root=root)["network"]
    defaults = config.get("node_defaults", {})
    nodes = sorted((defaults | n for n in config["nodes"]), key=lambda n: -n["echelon"])
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.5))
    ax.set_xlim(0, 16.8)
    ax.set_ylim(0.6, 4.3)
    ax.axis("off")
    width, height, y0 = 2.6, 1.9, 1.5
    xs = [2.6, 7.3, 12.0]
    upper, lower = y0 + 1.25, y0 + 0.65

    def label(x: float, y: float, text: str, va: str) -> None:
        ax.text(x, y, text, ha="center", va=va, fontsize=7, color=INK_2, linespacing=1.3)

    def arrow(x0: float, y: float, x1: float, *, dashed: bool, color: str) -> None:
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=9,
                                     linewidth=1.3, color=color,
                                     linestyle=(0, (4, 3)) if dashed else "-"))

    for x, node in zip(xs, nodes, strict=True):
        ax.add_patch(FancyBboxPatch((x, y0), width, height,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor="#eef3fb", edgecolor=ONE, linewidth=1.0))
        ax.text(x + width / 2, y0 + height - 0.35, node["node_id"], ha="center", va="center",
                fontsize=8.5, color=INK)
        ax.text(x + width / 2, y0 + 0.75,
                f"stock {node['initial_inventory']:g}\ncapacity {node['capacity']:g}\n"
                f"s {node['reorder_point']:g} · Q {node['order_quantity']:g}",
                ha="center", va="center", fontsize=7, color=INK_2, linespacing=1.35)

    gap = 0.12
    for xa, xb, b in zip(xs, xs[1:], nodes[1:], strict=False):
        middle = (xa + width + xb) / 2
        arrow(xa + width + gap, upper, xb - gap, dashed=False, color=ONE)
        label(middle, upper + 0.15, f"shipments\nL = {b['lead_time_mean']:g}", "bottom")
        arrow(xb - gap, lower, xa + width + gap, dashed=True, color=REFERENCE)
        label(middle, lower - 0.15, "orders", "top")
    plant = nodes[0]
    arrow(0.4, upper, xs[0] - gap, dashed=False, color=ONE)
    label(1.45, upper + 0.15, f"production\nL = {plant['lead_time_mean']:g}", "bottom")
    end_x = xs[-1] + width
    arrow(end_x + gap, upper, 16.6, dashed=False, color=ONE)
    label((end_x + 16.6) / 2 + 0.15, upper + 0.15, "sales", "bottom")
    arrow(16.6, lower, end_x + gap, dashed=True, color=REFERENCE)
    label((end_x + 16.6) / 2 + 0.15, lower - 0.15, "demand", "top")
    label(xs[-1] + width / 2, y0 + height + 0.2,
          "demand volatility enters at the retailer", "bottom")
    ax.text(0.4, 0.85, "Solid arrows: material. Dashed arrows: orders and demand; a supplier "
            "sees an order one period after it is placed.\nOrders that cannot be filled stay "
            "as backorders and ship when stock arrives.",
            ha="left", va="center", fontsize=7, color=INK_2, linespacing=1.4)
    fig.savefig(out / "fig_v2_network.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_calibration(root: Path, out: Path) -> None:
    """Posterior predictive distance on held-out statistics, 30 M5 series per process.

    Reads the v1 calibration reports (the calibration involves the demand
    process only, so the engine change does not affect it). The median is the
    standard one, the mean of the two middle values for an even count.
    """
    reports = {
        "iid": root / "outputs/calibration/calibration_weekly_both.json",
        "ar1": root / "outputs/calibration/calibration_weekly_ar1.json",
    }
    labels = {"iid": "Level + i.i.d. noise", "ar1": "Level + AR(1) noise"}
    rows: list[list[Any]] = []
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN * 0.8, 2.3))
    rng = np.random.default_rng(0)
    for y, process in enumerate(("ar1", "iid")):
        series = json.loads(reports[process].read_text(encoding="utf-8"))["series"]
        checks = [v["processes"][process] for v in series.values() if process in v["processes"]]
        gaps = [float(c["sim_to_real_gap"]) for c in checks]
        jitter = rng.uniform(-0.18, 0.18, size=len(gaps))
        ax.plot(gaps, y + jitter, "o", color=ONE, alpha=0.75, markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=0.6)
        median = float(np.median(gaps))
        ax.vlines(median, y - 0.3, y + 0.3, color=INK, linewidth=1.6, zorder=3)
        ax.annotate(f"median {median:.2f}", (median, y + 0.32), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7,
                    color=INK, bbox=LABEL_BOX, zorder=4)
        overfit = sum(bool(c["overfitted"]) for c in checks)
        identified = sum(bool(c["all_identified"]) for c in checks)
        rows += [[process, name, round(g, 4), bool(c["overfitted"]), bool(c["all_identified"])]
                 for name, g, c in zip(series, gaps, checks, strict=True)]
        labels[process] += (f"\noverfit {overfit}/{len(gaps)} · "
                            f"identified {identified}/{len(gaps)}")
    ax.set_yticks([0, 1], [labels["ar1"], labels["iid"]])
    ax.set_ylim(-0.75, 1.75)
    ax.set_xlim(0, 12)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("posterior predictive distance on held-out statistics (larger is worse)")
    ax.set_title("Calibration against 30 M5 store-by-category series (weekly)", loc="left")
    fig.savefig(out / "fig_v2_calibration.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    with (out / "fig_v2_calibration.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["process", "series", "gap", "overfitted", "identified"])
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="render the v2 figures")
    parser.add_argument("--outputs-root", required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    outputs = Path(args.outputs_root).resolve()
    figures = outputs / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    style()

    report = json.loads((outputs / CONFIRMATORY).read_text(encoding="utf-8"))
    cells = load_cells(outputs / CONFIRMATORY, outputs / "runs")
    fig_network(root, figures)
    fig_calibration(root, figures)
    fig_beta(cells, figures)
    fig_evidence(report, figures)
    fig_epsilon(cells, figures)
    fig_envelope(report, figures)
    fig_power(json.loads((root / "specs/v2/power_v2.json").read_text(encoding="utf-8")),
              figures, int(report["plan"]["replicates"]))
    sweeps_path = outputs / "sweeps/sweeps.json"
    if sweeps_path.is_file():
        fig_sweeps(json.loads(sweeps_path.read_text(encoding="utf-8")), figures)
    if (outputs / DETECTOR).is_file():
        null_path = outputs / "analysis/detector_null.json"
        null = json.loads(null_path.read_text(encoding="utf-8")) if null_path.is_file() else None
        fig_detector(load_cells(outputs / DETECTOR, outputs / "runs"), null, figures)
    print(f"  -> {figures}")
    for path in sorted(figures.glob("fig_v2_*.png")):
        print(f"     {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
