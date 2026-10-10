#!/usr/bin/env python
"""Render the manuscript figures with the MDPI editor's corrections applied.

Runs scripts/render_figures.py unchanged, but intercepts every save so the
figures leave with the fixes the English-editing round asked for:

* fig0  labels capitalized (Disruptions, Material, Orders, Demand) and the
        "inv ... lead time" key removed, since the caption now carries it;
* fig2  legend moved below the axes so it no longer overlaps the points, and
        the first letter of every y-axis label capitalized;
* fig3  each "five" written as "5";
* fig4  the familywise-threshold annotation lifted above the axes so it no
        longer collides with the top tick labels, "Familywise" capitalized,
        and the legend moved below the axes.

The data, colors, sizes and dpi come from render_figures.py as before; this
wrapper only edits text and placement, so no number in any figure can change.

    python scripts/render_figures_mdpi.py --dpi 300
"""

from __future__ import annotations

import re
import runpy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator
from matplotlib.transforms import blended_transform_factory

_CAPITALIZE = {"disruptions": "Disruptions", "material": "Material",
               "orders": "Orders", "demand": "Demand"}
_original_savefig = Figure.savefig


def _cap_first(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _fix_text(text: str) -> str:
    text = re.sub(r"\bfive\b", "5", text)
    text = text.replace("familywise", "Familywise")
    return _CAPITALIZE.get(text.strip(), text)


def _legend_below(ax, offset: float = -0.11) -> None:  # type: ignore[no-untyped-def]
    legend = ax.get_legend()
    if legend is None:
        return
    handles = list(getattr(legend, "legend_handles", None) or legend.legendHandles)
    labels = [t.get_text() for t in legend.get_texts()]
    legend.remove()
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, offset),
              ncol=len(labels), frameon=False)


def _patched_savefig(self: Figure, fname, *args, **kwargs):  # type: ignore[no-untyped-def]
    name = Path(str(fname)).name

    for text in self.findobj(Text):
        raw = text.get_text()
        if "initial inventory" in raw and "lead time" in raw:
            text.set_visible(False)  # the key now lives in the caption
            continue
        fixed = _fix_text(raw)
        if fixed != raw:
            text.set_text(fixed)

    for ax in self.axes:
        # Only categorical axes: on a numeric axis the labels follow a locator,
        # and freezing them would detach the text from the ticks it names.
        if isinstance(ax.yaxis.get_major_locator(), FixedLocator):
            ax.set_yticklabels(
                [_cap_first(_fix_text(t.get_text())) for t in ax.get_yticklabels()]
            )
        ax.set_xlabel(_cap_first(ax.get_xlabel()))
        ax.set_ylabel(_cap_first(ax.get_ylabel()))

        if name.startswith("fig4"):
            for text in ax.texts:
                if "threshold" in text.get_text():
                    text.set_transform(blended_transform_factory(ax.transData, ax.transAxes))
                    text.set_y(1.02)
                    text.set_va("bottom")
                    text.set_clip_on(False)
        if name.startswith(("fig1", "fig2", "fig4")):
            # fig1's legend sat on the top gridline beside the 0.8 tick label.
            # fig4's log-scale tick labels carry superscripts and stand taller,
            # so its legend needs more clearance to stay below the axis label.
            offset = {"fig1": -0.16, "fig4": -0.19}.get(name[:4], -0.11)
            _legend_below(ax, offset=offset)

        if name.startswith("fig3"):
            # The legend sits inside the axes, and the x = -0.5 gridline ran
            # through "of" in "Positive value of information". An opaque,
            # borderless backing hides the grid behind the legend only.
            legend = ax.get_legend()
            if legend is not None:
                frame = legend.get_frame()
                legend.set_frame_on(True)
                frame.set_facecolor("white")
                frame.set_edgecolor("none")
                frame.set_alpha(1.0)
                legend.set_zorder(6)

        if name.startswith(("fig1", "fig3", "fig4")):
            # A paper-colored backing keeps a reference line or grid line from
            # running through a label, without moving the label off its point.
            for text in ax.texts:
                if text.get_visible() and text.get_text().strip():
                    text.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 1.0})
                    text.set_zorder(5)

        if name.startswith("fig4"):
            # Left-going bars end near 0.04, and their value labels sat on top
            # of the y tick labels. Extending the axis left by more than an
            # order of magnitude opens the space those labels need.
            low, high = ax.get_xlim()
            ax.set_xlim(low / 40.0, high)

    # Text moved outside the axes would be cropped by a fixed bounding box.
    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("pad_inches", 0.04)
    return _original_savefig(self, fname, *args, **kwargs)


Figure.savefig = _patched_savefig  # type: ignore[method-assign]

if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "render_figures.py"
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")
