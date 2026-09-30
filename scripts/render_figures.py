#!/usr/bin/env python
"""Render the study's figures as print-resolution PNGs.

The campaign writes its figures as SVG, which is right for the repository —
text that diffs against its previous version, so a reviewer can see what a
re-run changed — and wrong for a publisher that wants raster at 300 dpi. This
script closes that gap without introducing a second source of truth: it reads
the same study bundle the SVGs are built from, so a figure here and a figure
there cannot disagree.

    python scripts/render_figures.py                     # 300 dpi, default
    python scripts/render_figures.py --dpi 600           # for line art
    python scripts/render_figures.py --format pdf        # vector for LaTeX

Two decisions worth stating.

**Matplotlib rather than an SVG rasterizer.** Converting the existing SVGs
would inherit their layout, which was designed for a web page at an arbitrary
width. A journal figure has a fixed column width in millimetres and a minimum
legible type size at that width, and those constraints change the layout rather
than scaling it. So the figures are re-authored here against print geometry.

**Nothing is hard-coded.** Every value comes from the bundle. A figure that
carries a number the campaign no longer produces is worse than a missing
figure, because it looks current.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import matplotlib
except ImportError:  # pragma: no cover - environment guidance
    print(
        "matplotlib is required for figure rendering and is not a project "
        "dependency:\n  pip install matplotlib",
        file=sys.stderr,
    )
    raise SystemExit(2) from None

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# -- print geometry -------------------------------------------------------
#: MDPI single-column text width is 170 mm. Figures are authored at that width
#: so they are placed at 1:1 and never scaled by the typesetter — scaling is
#: what makes type in one figure disagree with type in the next.
TEXT_WIDTH_MM = 170.0
MM_PER_INCH = 25.4
TEXT_WIDTH_IN = TEXT_WIDTH_MM / MM_PER_INCH

#: Type sizes in points, chosen so that nothing falls below 7 pt at the placed
#: width. Below that, axis labels stop being legible in print regardless of
#: resolution — dpi fixes sharpness, not size.
BASE_PT = 9.0
SMALL_PT = 8.0
TINY_PT = 7.0

#: Palette shared with the report so a reader moving between them is not
#: relearning the colors.
INK = "#292b31"
BODY = "#3f424d"
MUTED = "#75798c"
FAINT = "#9397ab"
RULE = "#e4e7f5"
AXIS = "#b2b6ca"
ACCENT = "#796cbf"
ACCENT_DEEP = "#5d5294"
ACCENT_LIGHT = "#b5abfc"
PAPER = "#ffffff"


def configure_style() -> None:
    """Apply the shared typographic settings.

    Set once, globally, rather than per-figure: a per-figure override is how
    two panels of the same paper end up with different label sizes.
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": BASE_PT,
            "axes.labelsize": SMALL_PT,
            "axes.titlesize": BASE_PT,
            "xtick.labelsize": TINY_PT,
            "ytick.labelsize": TINY_PT,
            "legend.fontsize": TINY_PT,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": BODY,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "axes.grid": False,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            # Grayscale-safe hatching for the one figure that needs a second
            # series distinguished without relying on hue.
            "hatch.linewidth": 0.5,
        }
    )


# -- bundle access --------------------------------------------------------
def dig(tree: Any, path: str) -> float | None:
    """Read a dotted path, returning None rather than a fabricated zero."""
    node = tree
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    if isinstance(node, bool) or not isinstance(node, int | float):
        return None
    return float(node)


@dataclass(frozen=True, slots=True)
class Study:
    """The bundle, the per-arm aggregates, and the calibration reports."""

    bundle: dict[str, Any]
    aggregates: dict[str, Any]
    #: Per-process calibration summaries, keyed by generating process. Empty
    #: when no calibration has run, which is the state a figure must be able
    #: to skip on rather than render as zeros.
    calibration: dict[str, Any] = field(default_factory=dict)

    @property
    def campaign(self) -> dict[str, Any]:
        return self.bundle.get("campaign", {})

    @property
    def hypotheses(self) -> list[dict[str, Any]]:
        return list(self.campaign.get("hypotheses", []))

    @property
    def alpha(self) -> float:
        return float(self.campaign.get("plan", {}).get("alpha", 0.05))

    def arm(self, arm: str, indicator: str, field: str = "mean") -> float | None:
        entry = self.aggregates.get(arm, {}).get(indicator)
        if isinstance(entry, dict) and isinstance(entry.get(field), int | float):
            return float(entry[field])
        return None

    def arms_with(self, indicator: str) -> list[str]:
        return sorted(
            arm
            for arm in self.aggregates
            if self.arm(arm, indicator) is not None
        )


def load_calibration(directory: Path) -> dict[str, Any]:
    """Collect per-process calibration summaries from a directory of reports.

    A later report supersedes an earlier one for the same process, because a
    rerun is a rerun at a larger proposal budget. Merging the other way would
    let a thinner run overwrite a better one. Reports are matched by glob
    rather than named, so adding a cadence or a process variant needs no edit
    here.
    """
    if not directory.exists():
        return {}
    merged: dict[str, Any] = {}
    for path in (directory / n for n in ("calibration_weekly_both.json", "calibration_weekly_ar1.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        series = payload.get("series", {})
        for process, summary in (payload.get("by_process") or {}).items():
            if process.startswith("_"):
                continue
            gaps = sorted(
                s["processes"][process]["sim_to_real_gap"]
                for s in series.values()
                if process in s.get("processes", {})
            )
            if not gaps:
                continue
            merged[process] = {
                **summary,
                "gaps": gaps,
                "series_count": len(gaps),
                "cadence": payload.get("cadence"),
                "proposals": payload.get("proposals"),
            }
    return merged


def load_study(directory: Path) -> Study:
    """Read the bundle and aggregates from a campaign directory."""
    bundle_path = directory / "study_bundle.json"
    aggregates_path = directory / "arm_aggregates.json"
    missing = [p for p in (bundle_path, aggregates_path) if not p.exists()]
    if missing:
        raise SystemExit(
            "cannot render figures: missing "
            + ", ".join(str(p) for p in missing)
            + "\nRun `make study` first, or pass --campaign-dir."
        )
    return Study(
        bundle=json.loads(bundle_path.read_text(encoding="utf-8")),
        aggregates=json.loads(aggregates_path.read_text(encoding="utf-8")),
    )


# -- figures --------------------------------------------------------------
def figure_central(study: Study) -> plt.Figure | None:
    """Service level and intervention rate across the three architectures.

    Grouped bars because the point is not either series alone but their
    relation: the shielded arm attains more service *while intervening more*,
    and a single-series chart cannot show that.
    """
    wanted = [
        ("ungoverned", "No oversight"),
        ("governed", "Veto only"),
        ("shielded", "Veto + substitution"),
    ]
    arms = [(a, label) for a, label in wanted if a in study.aggregates]
    if not arms:
        return None

    service = [study.arm(a, "operational.service_level") or 0.0 for a, _ in arms]
    service_sd = [study.arm(a, "operational.service_level", "sd") or 0.0 for a, _ in arms]
    intervention = [study.arm(a, "governance.intervention_rate") or 0.0 for a, _ in arms]

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, TEXT_WIDTH_IN * 0.40))
    positions = range(len(arms))
    width = 0.34

    ax.bar(
        [p - width / 2 for p in positions],
        service,
        width,
        yerr=service_sd,
        color=ACCENT,
        error_kw={"ecolor": ACCENT_DEEP, "elinewidth": 0.7, "capsize": 2.5},
        label="Service level",
    )
    ax.bar(
        [p + width / 2 for p in positions],
        intervention,
        width,
        color=ACCENT_LIGHT,
        # Hatching so the two series remain distinguishable if the journal
        # prints in grayscale.
        hatch="///",
        edgecolor=PAPER,
        linewidth=0.0,
        label="Intervention rate",
    )

    for position, value, sd in zip(positions, service, service_sd, strict=True):
        ax.text(
            position - width / 2,
            value + sd + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=TINY_PT,
            color=INK,
            fontweight="medium",
        )
    for position, value in zip(positions, intervention, strict=True):
        ax.text(
            position + width / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=TINY_PT,
            color=MUTED,
        )

    ax.set_xticks(list(positions))
    ax.set_xticklabels([label for _, label in arms], fontsize=SMALL_PT, color=BODY)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, max(max(service), max(intervention)) * 1.32)
    ax.yaxis.grid(True, color=RULE, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=2)
    fig.tight_layout()
    return fig


def figure_coverage(study: Study) -> plt.Figure | None:
    """Realized flag rate against the adapted nominal level, per arm.

    Dumbbells rather than bars of the error: the gap is the quantity, and
    showing both endpoints reveals *which* endpoint moved across
    configurations — the observation that redirected our diagnosis.
    """
    arms = [
        a
        for a in study.arms_with("guarantees.ood_rate")
        if study.arm(a, "guarantees.mean_nominal_level") is not None
    ]
    if not arms:
        return None

    arms.sort(key=lambda a: study.arm(a, "guarantees.mean_nominal_level") or 0.0)
    realized = [study.arm(a, "guarantees.ood_rate") or 0.0 for a in arms]
    nominal = [study.arm(a, "guarantees.mean_nominal_level") or 0.0 for a in arms]

    height = max(TEXT_WIDTH_IN * 0.30, 0.26 * len(arms) + 0.9)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, height))

    for index, (low, high) in enumerate(zip(realized, nominal, strict=True)):
        ax.plot([low, high], [index, index], color=AXIS, linewidth=1.1, zorder=1)
    ax.scatter(realized, range(len(arms)), s=26, color=ACCENT, zorder=3,
               label="Realized flag rate")
    ax.scatter(nominal, range(len(arms)), s=26, facecolors=PAPER,
               edgecolors=ACCENT_DEEP, linewidths=1.1, zorder=3,
               label="Nominal level")

    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([a.replace("_", " ") for a in arms], fontsize=TINY_PT, color=BODY)
    ax.set_xlabel("Rate  (gap = coverage error)")
    ax.set_xlim(left=0)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=RULE, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    return fig


def figure_value(study: Study) -> plt.Figure | None:
    """Value of governance against total value creation.

    Both axes signed, and the zero line on the value-of-information axis drawn
    explicitly: the finding is which side of it each arm falls on, and an axis
    fitted to the data's own range would hide that.

    Arms are labelled by *cluster* rather than individually. In this study the
    thirteen arms collapse onto two points — the blocking-only group and the
    substituting group — because organizational value depends on the oversight
    architecture and not on the descriptor or conformal variations. Twelve
    overlapping labels at two coordinates is not a denser figure, it is an
    illegible one, and the collapse is itself the result worth showing.
    """
    strategic = study.campaign.get("strategic", {})
    arms = strategic.get("arms")
    if not isinstance(arms, dict) or not arms:
        return None

    points = [
        (name, float(v["value_of_governance"]), float(v["tvc"]))
        for name, v in arms.items()
        if isinstance(v.get("value_of_governance"), int | float)
        and isinstance(v.get("tvc"), int | float)
    ]
    if not points:
        return None

    # Group arms that are visually coincident, not merely numerically equal.
    # Rounding to a fixed number of decimals fails here: two arms can differ
    # in the fourth decimal and still land on the same pixel, which produced
    # separate labels drawn on top of each other. Proximity is therefore
    # measured relative to the plotted range.
    span_x = max(max(p[1] for p in points) - min(p[1] for p in points), 1e-9)
    span_y = max(max(p[2] for p in points) - min(p[2] for p in points), 1e-9)
    tolerance = 0.04  # fraction of the plotted range that counts as one point

    clusters: list[dict[str, Any]] = []
    for name, vog, tvc in sorted(points):
        for cluster in clusters:
            near_x = abs(vog - cluster["vog"]) / span_x <= tolerance
            near_y = abs(tvc - cluster["tvc"]) / span_y <= tolerance
            if near_x and near_y:
                cluster["members"].append(name)
                # Keep the centroid so the marker sits among its members
                # rather than on whichever one happened to arrive first.
                count = len(cluster["members"])
                cluster["vog"] += (vog - cluster["vog"]) / count
                cluster["tvc"] += (tvc - cluster["tvc"]) / count
                break
        else:
            clusters.append({"vog": vog, "tvc": tvc, "members": [name]})

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, TEXT_WIDTH_IN * 0.46))

    for cluster in sorted(clusters, key=lambda c: c["vog"]):
        vog, tvc, members = cluster["vog"], cluster["tvc"], cluster["members"]
        # Marker area grows with cluster size, so a reader sees that one point
        # carries several arms rather than reading it as a single run.
        ax.scatter(
            vog,
            tvc,
            s=44 + 14 * (len(members) - 1),
            alpha=0.85,
            zorder=3,
            color=ACCENT_DEEP if vog > 0 else ACCENT,
        )
        label = (
            members[0].replace("_", " ")
            if len(members) == 1
            else f"{len(members)} arms"
        )
        # Offset away from the vertical zero line, so a cluster sitting near
        # it does not have its label run across the line.
        to_the_left = vog > 0 and vog < span_x * 0.25
        ax.annotate(
            label,
            (vog, tvc),
            textcoords="offset points",
            xytext=(-10 if to_the_left else 10, 5),
            ha="right" if to_the_left else "left",
            fontsize=TINY_PT,
            color=BODY,
            fontweight="medium" if len(members) > 1 else "normal",
        )

    # Zero must be in range on the horizontal axis: which side of it an arm
    # sits on is the finding. Padding is computed from the data rather than
    # left to the autoscaler, which would clip a point sitting on the line.
    xs = [c["vog"] for c in clusters]
    ys = [c["tvc"] for c in clusters]
    x_low, x_high = min(min(xs), 0.0), max(max(xs), 0.0)
    x_pad = max((x_high - x_low) * 0.16, 0.05)
    y_pad = max((max(ys) - min(ys)) * 0.22, 0.02)
    ax.set_xlim(x_low - x_pad, x_high + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

    ax.axvline(0.0, color=AXIS, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    # Axes-fraction coordinates, not data coordinates: reading the limit before
    # the data is plotted returns matplotlib's default 0-1 range and forces the
    # axis to span from the data to 1.0, which produced a figure several times
    # taller than its content.
    ax.text(
        0.0,
        1.0,
        " VoG = 0",
        transform=ax.get_xaxis_transform(),
        fontsize=TINY_PT,
        color=FAINT,
        va="top",
        ha="left",
    )

    ax.set_xlabel("Value of governance (signed)")
    ax.set_ylabel("Total value creation")
    ax.grid(True, color=RULE, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [
        Patch(facecolor=ACCENT_DEEP, label="Positive value of information"),
        Patch(facecolor=ACCENT, label="Negative value of information"),
    ]
    # Lower left is the one quadrant no cluster occupies in this study, and
    # the legend is placed there rather than by "best", which chose a spot
    # that collided with a label.
    ax.legend(handles=handles, loc="lower left", frameon=False)
    fig.tight_layout()
    return fig


def figure_evidence(study: Study) -> plt.Figure | None:
    """Bidirectional e-values on a logarithmic scale.

    Log scale because the values span nine orders of magnitude; on a linear
    axis every inconclusive result renders as an indistinguishable stub beside
    the two that decided.

    Both directions are plotted. A small forward e-value means the bet lost,
    which is not evidence that the effect runs the other way — only the reverse
    e-value establishes that, and a figure showing one without the other
    invites exactly the misreading this study had to correct.

    Keys are read from the payload the report actually emits
    (``evidence.forward_e_value`` / ``evidence.reverse_e_value``). An earlier
    version read a schema of its own invention and skipped on every run with
    "the bundle carries no data for it" — truthful about where it looked and
    false about the campaign, which is the worst shape a diagnostic can take.
    """
    usable = [
        h
        for h in study.hypotheses
        if isinstance((h.get("evidence") or {}).get("forward_e_value"), int | float)
        and h["evidence"]["forward_e_value"] > 0
    ]
    if not usable:
        return None

    def forward(h: dict[str, Any]) -> float:
        return float(h["evidence"]["forward_e_value"])

    def reverse(h: dict[str, Any]) -> float | None:
        value = (h.get("evidence") or {}).get("reverse_e_value")
        return float(value) if isinstance(value, int | float) and value > 0 else None

    usable.sort(key=lambda h: max(forward(h), reverse(h) or 0.0), reverse=True)
    # Read from the payload rather than recomputed: the threshold the figure
    # draws must be the one the verdicts were taken against.
    declared = [
        h["evidence"]["familywise_threshold"]
        for h in usable
        if isinstance((h.get("evidence") or {}).get("familywise_threshold"), int | float)
    ]
    threshold = float(declared[0]) if declared else len(usable) / study.alpha

    height = max(TEXT_WIDTH_IN * 0.34, 0.30 * len(usable) + 1.0)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, height))
    positions = range(len(usable))
    offset = 0.19

    for index, hypothesis in enumerate(usable):
        for value, sign, color in (
            (forward(hypothesis), -1, ACCENT_DEEP),
            (reverse(hypothesis), +1, ACCENT_LIGHT),
        ):
            if value is None:
                continue
            y = index + sign * offset
            ax.plot([1.0, value], [y, y], color=color, linewidth=3.4,
                    solid_capstyle="butt", zorder=2)
            ax.text(
                value * (1.35 if value >= 1 else 0.74),
                y,
                f"{value:.3g}" if value < 1000 else f"{value:.2e}",
                fontsize=TINY_PT,
                va="center",
                ha="left" if value >= 1 else "right",
                color=INK if value >= threshold else MUTED,
            )

    ax.axvline(1.0, color=AXIS, linewidth=0.9, zorder=1)
    ax.axvline(threshold, color=ACCENT, linewidth=0.8,
               linestyle=(0, (3, 3)), zorder=1)
    ax.text(threshold, -0.75, f" familywise threshold = {threshold:.0f}",
            fontsize=TINY_PT, color=ACCENT_DEEP, va="bottom")

    ax.set_xscale("log")
    ax.set_yticks(list(positions))
    ax.set_yticklabels(
        [f"{h['id']} · {h['indicator'].split('.')[-1].replace('_', ' ')}" for h in usable],
        fontsize=TINY_PT,
        color=BODY,
    )
    ax.set_xlabel("e-value (log scale)")
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=RULE, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [
        Line2D([], [], color=ACCENT_DEEP, linewidth=3.4, label="Preregistered direction"),
        Line2D([], [], color=ACCENT_LIGHT, linewidth=3.4, label="Opposite direction"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    fig.tight_layout()
    return fig


def figure_calibration(study: Study) -> plt.Figure | None:
    """Gap distributions by generating process, one strip per process.

    A strip rather than a box plot. With thirty points per process the
    individual values are legible, and a box would hide that the two
    distributions barely overlap --- which is the comparison the figure exists
    to make. The median is marked because it is the number the text quotes.

    The overfit count sits under each label rather than in a legend, because
    the tension between the two rows is the finding: the twin's own process is
    identified and overfits everywhere, while the process that halves the gap
    leaves its persistence parameter unidentified.
    """
    processes = [p for p in ("iid", "ar1") if p in study.calibration]
    if not processes:
        return None

    labels = {"iid": "Level + i.i.d. noise\n(the twin's own)", "ar1": "Level + AR(1) noise"}
    fig, ax = plt.subplots(
        figsize=(TEXT_WIDTH_IN, 0.9 + 0.95 * len(processes)), constrained_layout=True
    )

    for index, process in enumerate(processes):
        entry = study.calibration[process]
        gaps = [float(g) for g in entry.get("gaps", [])]
        if not gaps:
            continue
        colour = ACCENT if process == "iid" else ACCENT_DEEP
        # Jitter is deterministic: a figure whose points move between renders
        # cannot be diffed against its previous version.
        offsets = [
            0.16 * ((k % 5) - 2) / 2.0 for k in range(len(gaps))
        ]
        ax.scatter(
            gaps,
            [index + o for o in offsets],
            s=26,
            color=colour,
            alpha=0.55,
            edgecolors="none",
            zorder=3,
        )
        median = sorted(gaps)[len(gaps) // 2]
        ax.plot(
            [median, median], [index - 0.26, index + 0.26],
            color=INK, linewidth=2.0, zorder=4,
        )
        ax.text(
            median, index - 0.32, f"{median:.2f}",
            fontsize=SMALL_PT, color=INK, ha="center", va="bottom", fontweight="medium",
        )
        overfit = entry.get("overfitted_series")
        total = entry.get("series_count")
        identified = entry.get("identified_series")
        if overfit is not None and total:
            ax.text(
                -0.012, index + 0.30,
                f"{overfit}/{total} overfit · {identified}/{total} identified",
                transform=ax.get_yaxis_transform(),
                fontsize=TINY_PT, color=MUTED, ha="right", va="top",
            )

    ax.set_yticks(range(len(processes)))
    ax.set_yticklabels(
        [labels[p] for p in processes], fontsize=SMALL_PT, color=BODY
    )
    ax.set_ylim(len(processes) - 0.55, -0.62)
    ax.set_xlim(left=0.0)
    ax.set_xlabel(
        "Posterior predictive distance on held-out statistics  →  worse fit"
    )
    ax.xaxis.grid(True, color=RULE, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    return fig


#: What each figure reads, quoted verbatim in the skip message. A renderer that
#: reports only that it found nothing cannot be told apart from a campaign that
#: produced nothing.
_EXPECTED_INPUTS = {
    "fig1-central": "arm_aggregates.operational.service_level for the three arms",
    "fig2-coverage": "arm_aggregates.guarantees.ood_rate and mean_nominal_level",
    "fig3-value": "campaign.strategic.arms[*].value_of_governance and tvc",
    "fig4-evidence": "campaign.hypotheses[*].evidence.forward_e_value",
    "fig5-calibration": "calibration_*.json by_process[*].gap and overfitted_series",
}

FIGURES = {
    "fig1-central": figure_central,
    "fig2-coverage": figure_coverage,
    "fig3-value": figure_value,
    "fig4-evidence": figure_evidence,
    "fig5-calibration": figure_calibration,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="render the study's figures at print resolution"
    )
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=Path("outputs/campaigns/governance_value"),
        help="directory holding study_bundle.json and arm_aggregates.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: <campaign-dir>/figures/print)",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("outputs/calibration"),
        help="directory holding the calibration_*.json reports",
    )
    parser.add_argument("--dpi", type=int, default=300, help="raster resolution (default 300)")
    parser.add_argument(
        "--format",
        default="png",
        choices=("png", "pdf", "tiff", "eps"),
        help="output format; pdf is vector and ignores --dpi for geometry",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=sorted(FIGURES),
        help="render only the named figure (repeatable)",
    )
    args = parser.parse_args(argv)

    if args.dpi < 300 and args.format in ("png", "tiff"):
        print(
            f"warning: {args.dpi} dpi is below the 300 dpi most publishers "
            "require for raster figures",
            file=sys.stderr,
        )

    configure_style()
    study = load_study(args.campaign_dir)
    study.calibration.update(load_calibration(args.calibration_dir))
    out = args.out or (args.campaign_dir / "figures" / "print")
    out.mkdir(parents=True, exist_ok=True)

    selected = args.only or sorted(FIGURES)
    written: list[Path] = []
    skipped: list[str] = []

    for name in selected:
        figure = FIGURES[name](study)
        if figure is None:
            # Absent rather than empty: a blank figure implies the campaign
            # produced zeros, which is a different claim from producing nothing.
            skipped.append(name)
            continue
        path = out / f"{name}.{args.format}"
        figure.savefig(
            path,
            dpi=args.dpi,
            format=args.format,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(figure)
        written.append(path)

    width_px = round(TEXT_WIDTH_IN * args.dpi)
    print(f"campaign  {args.campaign_dir}")
    print(
        f"format    {args.format} at {args.dpi} dpi "
        f"({TEXT_WIDTH_MM:.0f} mm, {width_px} px wide)"
    )
    for path in written:
        print(f"  wrote   {path}")
    for name in skipped:
        # Name the key, not just the outcome. "No data for it" is true of where
        # the renderer looked and can be false of the campaign, and that gap is
        # what let a schema mismatch survive three consecutive runs looking
        # like an empty result.
        print(
            f"  skipped {name}: its inputs were absent from the bundle "
            f"(expected {_EXPECTED_INPUTS.get(name, 'see the figure function')})"
        )
    if not written:
        print("no figures rendered", file=sys.stderr)
        return 1
    print(f"\nCopy into the LaTeX project:\n  cp {out}/*.{args.format} <overleaf>/Definitions/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
