#!/usr/bin/env python3
"""Figure 6: fallback versus projection, under a fixed and a learned policy.

The central figure of the restructured manuscript. Two panels share an axis:
the opaque DRO policy on the left, tabular Q-learning on the right. Each panel
shows the same three architectures --- no oversight, fallback shield, projection
shield --- so the reader compares one operator change under two policies.

Values are read from the study bundle; nothing is typed in by hand. The
service-level bar carries its between-replicate standard deviation, the
intervention rate sits beside it, and the bracket over the two shields carries
the e-value of the preregistered contrast (RQ10, RQ11).

    python scripts/render_operator_figure.py --dpi 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "outputs/campaigns/governance_value"
TEXT_WIDTH_IN = 170 / 25.4

INK = "#292b31"
MUTED = "#595d6c"
RULE = "#e4e7f5"
SERVICE = "#796cbf"
INTERVENTION = "#b5abfc"
BRACKET = "#5d5294"

PANELS: tuple[tuple[str, str, tuple[str, str, str]], ...] = (
    ("Fixed policy (robust DRO)", "RQ10", ("ungoverned", "fallback_shield", "shielded")),
    ("Learned policy (tabular Q-learning)", "RQ11", ("q_ungoverned", "q_fallback", "q_projection")),
)
LABELS = ("No oversight", "Fallback\nshield", "Projection\nshield")


def load() -> tuple[dict[str, Any], dict[str, float]]:
    aggregates = json.loads((CAMPAIGN / "arm_aggregates.json").read_text(encoding="utf-8"))
    bundle = json.loads((CAMPAIGN / "study_bundle.json").read_text(encoding="utf-8"))
    evidence: dict[str, float] = {}
    for h in bundle["campaign"]["hypotheses"]:
        ev = h.get("evidence") or {}
        value = ev.get("forward_e_value", ev.get("e_value"))
        if isinstance(value, int | float):
            evidence[str(h["id"])] = float(value)
    return aggregates, evidence


def stat(aggregates: dict[str, Any], arm: str, indicator: str, key: str) -> float:
    entry = aggregates.get(arm, {}).get(indicator)
    if not isinstance(entry, dict) or not isinstance(entry.get(key), int | float):
        raise SystemExit(f"missing {indicator}.{key} for arm {arm!r}; run `make study` first")
    return float(entry[key])


def sci(value: float) -> str:
    exponent = len(str(int(value))) - 1
    return f"{value / 10**exponent:.2f}" + r"$\times10^{" + str(exponent) + "}$"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--out", type=Path, default=CAMPAIGN / "figures/print/fig6-operator.png")
    args = parser.parse_args(argv)

    aggregates, evidence = load()
    plt.rcParams.update({
        "font.family": ["Inter", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": MUTED,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.0), sharey=True)
    width = 0.36

    for ax, (title, rq, arms) in zip(axes, PANELS, strict=True):
        for i, arm in enumerate(arms):
            service = stat(aggregates, arm, "operational.service_level", "mean")
            spread = stat(aggregates, arm, "operational.service_level", "sd")
            rate = stat(aggregates, arm, "governance.intervention_rate", "mean")
            ax.bar(i - width / 2, service, width, color=SERVICE, zorder=3,
                   yerr=spread, error_kw={"ecolor": INK, "elinewidth": 0.8, "capsize": 2.5})
            ax.text(i - width / 2, service + spread + 0.025, f"{service:.3f}",
                    ha="center", va="bottom", color=INK, fontsize=7.5, fontweight="bold")
            if rate > 0.0:
                ax.bar(i + width / 2, rate, width, color=INTERVENTION, zorder=3)
                ax.text(i + width / 2, rate + 0.02, f"{rate:.3f}",
                        ha="center", va="bottom", color=MUTED, fontsize=7)

        # Bracket over the two shields, set above the taller of their error bars.
        tops = [
            stat(aggregates, a, "operational.service_level", "mean")
            + stat(aggregates, a, "operational.service_level", "sd")
            for a in arms[1:]
        ]
        y = max(tops) + 0.14
        x0, x1 = 1 - width / 2, 2 - width / 2
        ax.plot([x0, x0, x1, x1], [y - 0.02, y, y, y - 0.02], color=BRACKET, lw=0.9)
        if rq in evidence:
            ax.text((x0 + x1) / 2, y + 0.012, f"{rq}: $e$ = {sci(evidence[rq])}",
                    ha="center", va="bottom", color=BRACKET, fontsize=7.5)

        ax.set_title(title, fontsize=8.5, color=INK, pad=6, loc="left")
        ax.set_xticks(range(3), LABELS)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", color=RULE, lw=0.6, zorder=0)
        ax.tick_params(axis="both", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("Rate")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SERVICE),
        plt.Rectangle((0, 0), 1, 1, color=INTERVENTION),
    ]
    fig.legend(handles, ["Service level (whiskers: ±1 sd)", "Intervention rate"],
               loc="lower center", ncol=2, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=2.0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    fig.savefig(args.out.with_suffix(".svg"), facecolor="white")
    print(f"  wrote   {args.out.relative_to(ROOT)}")
    print(f"  wrote   {args.out.with_suffix('.svg').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
