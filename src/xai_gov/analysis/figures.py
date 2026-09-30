"""Figures generated from the study bundle.

Every number in the four figures of the first report was transcribed by hand
from the digest. That is exactly the step at which a unit mismatch or an
inverted sign enters a manuscript, and four of the nine catalogued defects
were found in transcribed prose rather than in code. This module removes the
step: the figures read the bundle directly, so they regenerate when a campaign
is re-run and cannot disagree with it.

SVG rather than a plotting library, for three reasons that matter here. The
output is text, so a figure diffs against its previous version and a reviewer
can see what a re-run changed. It carries no dependency, so reproducing the
report needs nothing beyond the repository. And it scales without loss, which
a raster export does not.

Two conventions run through all of them. A quantity the bundle does not carry
is omitted rather than defaulted — a figure with a fabricated zero is worse
than a figure with a gap, because the gap is visible. And every figure states
its own units and direction in the caption, since a chart whose axis nobody
can interpret is decoration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Palette, matching the report. Kept here rather than passed in: a figure that
#: can be recoloured per call will eventually be recoloured inconsistently.
INK = "#292b31"
BODY = "#3f424d"
MUTED = "#75798c"
FAINT = "#9397ab"
RULE = "#e4e7f5"
AXIS = "#b2b6ca"
ACCENT = "#796cbf"
ACCENT_DEEP = "#5d5294"
ACCENT_LIGHT = "#b5abfc"
PAPER = "#f3f5fe"


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _text(
    x: float,
    y: float,
    content: str,
    *,
    size: float = 11,
    fill: str = BODY,
    anchor: str = "start",
    weight: str = "400",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter,system-ui,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{_escape(content)}</text>'
    )


def _svg(width: float, height: float, body: str, *, label: str) -> str:
    return (
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" '
        f'style="width:100%;height:auto" role="img" '
        f'aria-label="{_escape(label)}">{body}</svg>'
    )


def _dig(tree: Any, path: str) -> float | None:
    node = tree
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    if isinstance(node, bool) or not isinstance(node, int | float):
        return None
    return float(node)


def network_diagram(
    nodes: list[dict[str, Any]], *, width: float = 660.0, title: str | None = None
) -> str:
    """The physical network: echelons, material and order flows, entry points.

    Generated from the network configuration rather than drawn, so the figure
    documents the parameterization instead of illustrating it, and so a
    five-echelon run produces its own diagram from the same code. A hand-drawn
    diagram is a second description of the network that can disagree with the
    first.

    Read left to right: demand enters at the downstream node, material flows
    toward it, and orders propagate away from it. Drawing both directions
    matters because the paper's central mechanism is upstream starvation — a
    single-arrow diagram would make the propagation path invisible.
    """
    if not nodes:
        return ""

    ordered = sorted(nodes, key=lambda n: int(n.get("echelon", 0)), reverse=True)
    top = 52.0 if title else 34.0
    cell_h, gap = 76.0, 26.0
    left, right = 30.0, width - 30.0
    cell_w = (right - left - gap * (len(ordered) - 1)) / len(ordered)
    parts: list[str] = []

    if title:
        parts.append(_text(left, 22, title, size=12, fill=MUTED, weight="500"))

    for index, node in enumerate(ordered):
        x = left + index * (cell_w + gap)
        downstream = index == len(ordered) - 1
        parts.append(
            f'<rect x="{x:.1f}" y="{top:.1f}" width="{cell_w:.1f}" height="{cell_h}" '
            f'rx="6" fill="{PAPER}" stroke="{ACCENT if downstream else AXIS}" '
            f'stroke-width="{1.8 if downstream else 1}"/>'
        )
        centre = x + cell_w / 2
        parts.append(
            _text(
                centre, top + 20, str(node.get("node_id", "?")), size=12.5,
                fill=INK, anchor="middle", weight="500",
            )
        )
        # The parameters that define the cell, so the figure is documentation.
        for row, (label, key) in enumerate(
            (("inv", "initial_inventory"), ("cap", "capacity"), ("s", "reorder_point"))
        ):
            value = node.get(key)
            if value is None:
                continue
            parts.append(
                _text(
                    centre, top + 37 + row * 13,
                    f"{label} {float(value):.0f}", size=10, fill=MUTED, anchor="middle",
                )
            )

        if index == len(ordered) - 1:
            continue
        # Material flows downstream (toward the customer), orders upstream.
        gap_left, gap_right = x + cell_w, x + cell_w + gap
        mid = top + cell_h / 2
        parts.append(
            f'<path d="M{gap_left + 3:.1f} {mid - 9:.1f} L{gap_right - 6:.1f} '
            f'{mid - 9:.1f}" stroke="{ACCENT}" stroke-width="1.5" fill="none" '
            f'marker-end="url(#arrow-fwd)"/>'
        )
        parts.append(
            f'<path d="M{gap_right - 3:.1f} {mid + 9:.1f} L{gap_left + 6:.1f} '
            f'{mid + 9:.1f}" stroke="{AXIS}" stroke-width="1.2" fill="none" '
            f'stroke-dasharray="3 2" marker-end="url(#arrow-back)"/>'
        )
        lead = ordered[index + 1].get("lead_time_mean")
        lead_sd = ordered[index + 1].get("lead_time_dispersion")
        if lead is not None:
            label = f"L {float(lead):.0f}"
            if lead_sd is not None:
                label += f"\u00b1{float(lead_sd):.1f}"
            parts.append(
                _text(
                    (gap_left + gap_right) / 2, mid - 15, label, size=9.5,
                    fill=FAINT, anchor="middle",
                )
            )

    # Exogenous demand enters at the downstream node; disruptions enter at any.
    last_x = left + (len(ordered) - 1) * (cell_w + gap)
    demand_x = last_x + cell_w / 2
    foot = top + cell_h
    parts.append(
        f'<path d="M{demand_x:.1f} {foot + 30:.1f} L{demand_x:.1f} {foot + 7:.1f}" '
        f'stroke="{ACCENT_DEEP}" stroke-width="1.5" fill="none" '
        f'marker-end="url(#arrow-fwd)"/>'
    )
    parts.append(
        _text(demand_x, foot + 44, "demand", size=10.5, fill=ACCENT_DEEP, anchor="middle")
    )
    parts.append(
        f'<path d="M{left + cell_w / 2:.1f} {top - 26:.1f} '
        f'L{left + cell_w / 2:.1f} {top - 6:.1f}" stroke="{MUTED}" '
        f'stroke-width="1.2" stroke-dasharray="3 2" fill="none" '
        f'marker-end="url(#arrow-fwd)"/>'
    )
    parts.append(
        _text(left + cell_w / 2, top - 32, "disruptions", size=10, fill=MUTED, anchor="middle")
    )

    legend = foot + 62
    parts.append(
        f'<line x1="{left}" y1="{legend - 4:.1f}" x2="{left + 22}" '
        f'y2="{legend - 4:.1f}" stroke="{ACCENT}" stroke-width="1.5"/>'
    )
    parts.append(_text(left + 28, legend, "material", size=10.5, fill=MUTED))
    parts.append(
        f'<line x1="{left + 100}" y1="{legend - 4:.1f}" x2="{left + 122}" '
        f'y2="{legend - 4:.1f}" stroke="{AXIS}" stroke-width="1.2" '
        f'stroke-dasharray="3 2"/>'
    )
    parts.append(_text(left + 128, legend, "orders", size=10.5, fill=MUTED))
    parts.append(
        _text(
            left + 200, legend,
            "inv initial inventory · cap capacity · s reorder point · L lead time",
            size=10, fill=FAINT,
        )
    )

    defs = (
        '<defs>'
        f'<marker id="arrow-fwd" viewBox="0 0 8 8" refX="6" refY="4" '
        f'markerWidth="5" markerHeight="5" orient="auto">'
        f'<path d="M0 1 L6 4 L0 7 z" fill="{ACCENT}"/></marker>'
        f'<marker id="arrow-back" viewBox="0 0 8 8" refX="6" refY="4" '
        f'markerWidth="5" markerHeight="5" orient="auto">'
        f'<path d="M0 1 L6 4 L0 7 z" fill="{AXIS}"/></marker>'
        '</defs>'
    )
    return _svg(
        width, legend + 14, defs + "".join(parts),
        label="physical supply network: echelons, material and order flows",
    )


def load_network(path: Path) -> list[dict[str, Any]]:
    """Read the node list out of a network configuration, defaults merged.

    Parsed with the project's own loader so the diagram and the simulator read
    the same file through the same code; a second parser is a second thing that
    can disagree about what the network is.

    ``node_defaults`` are merged in for the same reason the simulator merges
    them: a parameter declared once for every node is still that node's
    parameter, and a diagram that read only the per-node block would omit
    exactly the values the configuration factored out.
    """
    from xai_gov.io.yaml_loader import load_config

    config = load_config(path, root=path.parent.parent.parent)
    network = config.get("network") or {}
    nodes = network.get("nodes")
    if not isinstance(nodes, list):
        return []
    defaults = network.get("node_defaults") or {}
    if not isinstance(defaults, dict):
        defaults = {}
    return [
        {**defaults, **node} for node in nodes if isinstance(node, dict)
    ]


@dataclass(frozen=True, slots=True)
class ArmSeries:
    """One indicator across arms, ready to plot."""

    indicator: str
    values: dict[str, float]

    @property
    def maximum(self) -> float:
        return max(self.values.values(), default=0.0)

    @property
    def minimum(self) -> float:
        return min(self.values.values(), default=0.0)


def series_from_aggregates(
    aggregates: dict[str, Any], indicator: str, *, field: str = "mean"
) -> ArmSeries:
    """Pull one indicator out of the per-arm aggregates.

    Arms lacking the indicator are absent from the result rather than present
    with a zero: the whole point of the KPI layers refusing to fabricate a zero
    is undone if the figures put one back.
    """
    values: dict[str, float] = {}
    for arm, indicators in aggregates.items():
        entry = indicators.get(indicator)
        if isinstance(entry, dict) and isinstance(entry.get(field), int | float):
            values[str(arm)] = float(entry[field])
    return ArmSeries(indicator=indicator, values=values)


# -- figures --------------------------------------------------------------
def grouped_bars(
    primary: ArmSeries,
    secondary: ArmSeries,
    *,
    arms: list[str],
    labels: dict[str, str] | None = None,
    primary_label: str = "primary",
    secondary_label: str = "secondary",
    width: float = 660.0,
) -> str:
    """Two indicators per arm, side by side.

    Used for the central result, where the point is not either bar alone but
    their relation: the shielded arm attains more service *while intervening
    more*, and a single-series chart cannot show that.
    """
    shown = [arm for arm in arms if arm in primary.values or arm in secondary.values]
    if not shown:
        return ""

    labels = labels or {}
    top, floor, left = 22.0, 150.0, 60.0
    span = width - left - 20.0
    ceiling = max(primary.maximum, secondary.maximum, 0.1) * 1.15
    group = span / len(shown)
    bar = min(group * 0.34, 46.0)

    parts: list[str] = []
    for step in range(5):
        y = floor - (floor - top) * step / 4
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - 20:.0f}" y2="{y:.1f}" '
            f'stroke="{RULE}" stroke-width="1"/>'
        )
        parts.append(
            _text(left - 8, y + 3, f"{ceiling * step / 4:.2f}", size=10, fill=FAINT, anchor="end")
        )

    for index, arm in enumerate(shown):
        centre = left + group * (index + 0.5)
        for offset, (series, colour) in enumerate(
            ((primary, ACCENT), (secondary, ACCENT_LIGHT))
        ):
            value = series.values.get(arm)
            if value is None:
                continue
            height = (floor - top) * (value / ceiling)
            x = centre - bar + offset * bar
            parts.append(
                f'<rect x="{x:.1f}" y="{floor - height:.1f}" width="{bar - 2:.1f}" '
                f'height="{max(height, 1.0):.1f}" fill="{colour}"/>'
            )
            parts.append(
                _text(
                    x + bar / 2 - 1,
                    floor - height - 6,
                    f"{value:.3f}",
                    size=10,
                    fill=INK if offset == 0 else MUTED,
                    anchor="middle",
                    weight="500" if offset == 0 else "400",
                )
            )
        parts.append(
            _text(centre, floor + 18, labels.get(arm, arm), size=11.5, anchor="middle")
        )

    legend_y = floor + 36
    parts.append(f'<rect x="{left}" y="{legend_y - 8}" width="9" height="9" fill="{ACCENT}"/>')
    parts.append(_text(left + 14, legend_y, primary_label, size=10.5, fill=MUTED))
    parts.append(
        f'<rect x="{left + 130}" y="{legend_y - 8}" width="9" height="9" fill="{ACCENT_LIGHT}"/>'
    )
    parts.append(_text(left + 144, legend_y, secondary_label, size=10.5, fill=MUTED))

    return _svg(
        width,
        legend_y + 12,
        "".join(parts),
        label=f"{primary_label} and {secondary_label} by arm",
    )


def paired_dots(
    realized: ArmSeries,
    nominal: ArmSeries,
    *,
    arms: list[str],
    labels: dict[str, str] | None = None,
    width: float = 660.0,
) -> str:
    """Two points per arm joined by a line, the gap being the quantity.

    The coverage figure. A bar chart of the error would show its magnitude and
    hide what moved: filled dots barely shift between configurations while
    hollow ones travel, which says the manipulations changed the nominal level
    rather than what the detector sees.
    """
    shown = [arm for arm in arms if arm in realized.values and arm in nominal.values]
    if not shown:
        return ""

    labels = labels or {}
    left, right, top = 150.0, width - 40.0, 40.0
    ceiling = max(realized.maximum, nominal.maximum, 0.05) * 1.2
    row = 30.0

    def x_of(value: float) -> float:
        return left + (right - left) * (value / ceiling)

    parts: list[str] = []
    for step in range(3):
        value = ceiling * step / 2
        x = x_of(value)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 16:.1f}" x2="{x:.1f}" '
            f'y2="{top + row * len(shown):.1f}" stroke="{RULE}" stroke-width="1"/>'
        )
        parts.append(_text(x, top - 22, f"{value:.2f}", size=10, fill=FAINT, anchor="middle"))

    for index, arm in enumerate(shown):
        y = top + row * index
        a, b = x_of(realized.values[arm]), x_of(nominal.values[arm])
        parts.append(
            f'<line x1="{a:.1f}" y1="{y:.1f}" x2="{b:.1f}" y2="{y:.1f}" '
            f'stroke="{AXIS}" stroke-width="1.5"/>'
        )
        parts.append(f'<circle cx="{a:.1f}" cy="{y:.1f}" r="4.5" fill="{ACCENT}"/>')
        parts.append(
            f'<circle cx="{b:.1f}" cy="{y:.1f}" r="4.5" fill="{PAPER}" '
            f'stroke="{ACCENT_DEEP}" stroke-width="1.8"/>'
        )
        parts.append(_text(left - 12, y + 4, labels.get(arm, arm), size=11, anchor="end"))

    legend_y = top + row * len(shown) + 22
    parts.append(f'<circle cx="{left + 5}" cy="{legend_y - 4}" r="4.5" fill="{ACCENT}"/>')
    parts.append(_text(left + 16, legend_y, "realized rate", size=10.5, fill=MUTED))
    parts.append(
        f'<circle cx="{left + 135}" cy="{legend_y - 4}" r="4.5" fill="{PAPER}" '
        f'stroke="{ACCENT_DEEP}" stroke-width="1.8"/>'
    )
    parts.append(_text(left + 146, legend_y, "nominal level", size=10.5, fill=MUTED))
    parts.append(
        _text(left + 270, legend_y, "line length is the coverage error", size=10.5, fill=FAINT)
    )

    return _svg(width, legend_y + 12, "".join(parts), label="realized rate against nominal level")


def evidence_bars(
    hypotheses: list[dict[str, Any]],
    *,
    alpha: float = 0.05,
    familywise_threshold: float | None = None,
    width: float = 660.0,
) -> str:
    """Evidence per hypothesis on a log scale, with the decision rule marked.

    Log scale because the values span nine orders of magnitude, and a linear
    axis would render every inconclusive result as an indistinguishable stub
    beside the two that decided.

    Which value is plotted matters. A hypothesis established in the predicted
    direction shows its forward e-value to the right; one established in the
    opposite direction shows its *reverse* e-value to the left, because that is
    the test which actually supports the claim. A forward value below one is
    plotted as a stub at the centre and nothing more — it means the bet lost,
    and an earlier version of this figure drew it as though it were evidence
    for the opposite side.
    """
    usable = [
        h
        for h in hypotheses
        if isinstance((h.get("evidence") or {}).get("forward_e_value"), int | float)
        and h["evidence"]["forward_e_value"] > 0
    ]
    if not usable:
        return ""

    def plotted(hypothesis: dict[str, Any]) -> tuple[float, int]:
        """The value to draw and its side: +1 right, -1 left, 0 centre stub."""
        evidence = hypothesis.get("evidence") or {}
        forward = float(evidence["forward_e_value"])
        reverse = evidence.get("reverse_e_value")
        if evidence.get("refuted") and isinstance(reverse, int | float) and reverse > 0:
            return float(reverse), -1
        return forward, 1 if forward >= 1.0 else 0

    threshold = familywise_threshold or (len(usable) / alpha)
    ordered = sorted(usable, key=lambda h: plotted(h)[0], reverse=True)
    left, top, row = 190.0, 34.0, 26.0
    centre = left + 56.0
    right = width - 74.0

    extent = max(
        max(abs(math.log10(plotted(h)[0])) for h in ordered),
        math.log10(threshold),
    )

    def offset(value: float) -> float:
        return (right - centre) * (math.log10(max(value, 1.0)) / extent)

    parts: list[str] = [
        f'<line x1="{centre:.1f}" y1="{top - 8:.1f}" x2="{centre:.1f}" '
        f'y2="{top + row * len(ordered):.1f}" stroke="{AXIS}" stroke-width="1.5"/>',
        f'<line x1="{centre + offset(threshold):.1f}" y1="{top - 8:.1f}" '
        f'x2="{centre + offset(threshold):.1f}" y2="{top + row * len(ordered):.1f}" '
        f'stroke="{ACCENT}" stroke-width="1" stroke-dasharray="3 3"/>',
        f'<line x1="{centre - offset(threshold):.1f}" y1="{top - 8:.1f}" '
        f'x2="{centre - offset(threshold):.1f}" y2="{top + row * len(ordered):.1f}" '
        f'stroke="{ACCENT}" stroke-width="1" stroke-dasharray="3 3"/>',
        _text(centre, top - 14, "e = 1", size=10, fill=FAINT, anchor="middle"),
        _text(
            centre + offset(threshold) + 6,
            top - 14,
            f"familywise threshold {threshold:.0f}",
            size=10,
            fill=ACCENT_DEEP,
        ),
    ]

    for index, hypothesis in enumerate(ordered):
        value, side = plotted(hypothesis)
        y = top + row * index
        span = offset(value) if side != 0 else 0.0
        x = centre + side * span
        decided = bool(hypothesis.get("supported")) or bool(
            (hypothesis.get("evidence") or {}).get("refuted")
        )
        colour = ACCENT_DEEP if decided else (ACCENT_LIGHT if side > 0 else FAINT)
        start = centre if side >= 0 else x
        parts.append(
            f'<rect x="{start:.1f}" y="{y:.1f}" width="{max(span, 2.0):.1f}" '
            f'height="15" fill="{colour}"/>'
        )
        parts.append(
            _text(
                left - 10,
                y + 12,
                f"{hypothesis['id']} · {hypothesis['indicator'].split('.')[-1]}",
                size=11,
                anchor="end",
            )
        )
        formatted = f"{value:.3g}" if value < 1000 else f"{value:.2e}"
        label = formatted if side >= 0 else f"{formatted} (opposite)"
        if side >= 0:
            parts.append(
                _text(x + 6, y + 12, label, size=10.5, fill=INK, weight="500")
            )
        else:
            parts.append(_text(x - 6, y + 12, label, size=10.5, fill=INK, anchor="end"))

    foot = top + row * len(ordered) + 20
    parts.append(
        _text(left - 10, foot, "← established opposite", size=10.5, fill=FAINT, anchor="end")
    )
    parts.append(_text(right, foot, "as predicted →", size=10.5, fill=FAINT, anchor="end"))

    return _svg(width, foot + 12, "".join(parts), label="evidence by hypothesis, log scale")


def strategic_scatter(strategic: dict[str, Any], *, width: float = 660.0) -> str:
    """Value of information against total value creation.

    Both axes signed, and the zero line on the value-of-information axis drawn
    explicitly: the finding is which side of it each arm falls on, and an axis
    that started at the data's minimum would hide that entirely.
    """
    arms = strategic.get("arms")
    if not isinstance(arms, dict) or not arms:
        return ""

    points = [
        (
            str(arm),
            float(value["value_of_governance"]),
            float(value["tvc"]),
        )
        for arm, value in arms.items()
        if isinstance(value.get("value_of_governance"), int | float)
        and isinstance(value.get("tvc"), int | float)
    ]
    if not points:
        return ""

    left, right, top, floor = 90.0, width - 30.0, 34.0, 190.0
    vog = [p[1] for p in points]
    tvc = [p[2] for p in points]
    # Zero is always in range on the horizontal axis: which side of it an arm
    # sits on is the finding.
    x_low, x_high = min(min(vog), 0.0), max(max(vog), 0.0)
    y_low, y_high = min(tvc), max(tvc)
    x_pad = max((x_high - x_low) * 0.15, 0.05)
    y_pad = max((y_high - y_low) * 0.2, 0.02)

    def x_of(value: float) -> float:
        return left + (right - left) * (value - x_low + x_pad) / (x_high - x_low + 2 * x_pad)

    def y_of(value: float) -> float:
        return floor - (floor - top) * (value - y_low + y_pad) / (y_high - y_low + 2 * y_pad)

    parts: list[str] = [
        f'<line x1="{x_of(0.0):.1f}" y1="{top:.1f}" x2="{x_of(0.0):.1f}" '
        f'y2="{floor:.1f}" stroke="{AXIS}" stroke-width="1" stroke-dasharray="3 3"/>',
        _text(x_of(0.0) + 5, top + 10, "VoG = 0", size=10, fill=FAINT),
    ]
    for value in (y_low, y_high):
        y = y_of(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right:.0f}" y2="{y:.1f}" '
            f'stroke="{RULE}" stroke-width="1"/>'
        )
        parts.append(_text(left - 8, y + 3, f"{value:+.2f}", size=10, fill=FAINT, anchor="end"))

    for _arm, value_of_governance, total in sorted(points):
        x, y = x_of(value_of_governance), y_of(total)
        colour = ACCENT_DEEP if value_of_governance > 0 else ACCENT
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{colour}" opacity="0.85"/>')

    parts.append(_text(left, floor + 22, "value of governance (signed) →", size=10.5, fill=MUTED))
    parts.append(
        _text(left - 8, top - 8, "total value creation ↑", size=10.5, fill=MUTED, anchor="start")
    )

    return _svg(
        width,
        floor + 34,
        "".join(parts),
        label="value of governance against total value creation",
    )


# -- assembly -------------------------------------------------------------
def build_figures(
    bundle: dict[str, Any],
    aggregates: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> dict[str, str]:
    """Every figure the bundle supports, keyed by filename.

    A figure whose inputs are missing is absent from the result rather than
    emitted empty, so a caller can tell what the campaign did not produce.
    """
    campaign = bundle.get("campaign", {})
    figures: dict[str, str] = {}

    # The physical network comes first: a reader needs to see what is being
    # governed before seeing how. Emitted per topology the campaign used, so a
    # five-echelon run documents its own network rather than inheriting the
    # three-echelon picture.
    if project_root is not None:
        for name in ("three_echelon", "five_echelon", "single_echelon"):
            path = project_root / "configs" / "network" / f"{name}.yaml"
            if not path.exists():
                continue
            try:
                nodes = load_network(path)
            except (OSError, ValueError, KeyError):
                continue
            diagram = network_diagram(
                nodes, title=f"{name.replace('_', ' ')} · {len(nodes)} nodes"
            )
            if diagram:
                figures[f"fig0_network_{name}.svg"] = diagram

    arms = sorted(aggregates)
    service = series_from_aggregates(aggregates, "operational.service_level")
    intervention = series_from_aggregates(aggregates, "governance.intervention_rate")
    if service.values:
        # Annotated so the literal tuple does not narrow the element type: the
        # comprehension over a tuple of literals infers list[Literal[...]],
        # which is not a list[str] as far as the checker is concerned.
        headline: list[str] = [a for a in ("ungoverned", "governed", "shielded") if a in arms]
        chart = grouped_bars(
            service,
            intervention,
            arms=headline or arms[:3],
            primary_label="service level",
            secondary_label="intervention rate",
        )
        if chart:
            figures["fig1_service_and_intervention.svg"] = chart

    realized = series_from_aggregates(aggregates, "guarantees.ood_rate")
    nominal = series_from_aggregates(aggregates, "guarantees.mean_nominal_level")
    if realized.values and nominal.values:
        chart = paired_dots(realized, nominal, arms=arms)
        if chart:
            figures["fig2_coverage_gap.svg"] = chart

    scatter = strategic_scatter(bundle.get("campaign", {}).get("strategic", {}))
    if scatter:
        figures["fig3_strategic_value.svg"] = scatter

    bars = evidence_bars(
        campaign.get("hypotheses", []),
        alpha=float(campaign.get("plan", {}).get("alpha", 0.05)),
        familywise_threshold=(
            campaign.get("multiplicity", {}).get("familywise_threshold")
        ),
    )
    if bars:
        figures["fig4_evidence.svg"] = bars

    return figures


def write_figures(
    *,
    bundle: dict[str, Any],
    aggregates: dict[str, Any],
    directory: Path,
    project_root: Path | None = None,
) -> dict[str, Path]:
    """Render the figures to disk and return where each landed."""
    figures = build_figures(bundle, aggregates, project_root=project_root)
    if not figures:
        return {}
    target = directory / "figures"
    target.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, markup in figures.items():
        path = target / name
        path.write_text(markup + "\n", encoding="utf-8")
        written[name] = path
    return written
