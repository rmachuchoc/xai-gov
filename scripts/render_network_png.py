#!/usr/bin/env python
"""Render the network diagram (fig0) as a 300 dpi PNG for the manuscript.

The diagram is built by xai_gov.analysis.figures.network_diagram from the same
network configuration the simulator reads, so the figure cannot disagree with
the twin. That function emits SVG; this script rasterizes it at the manuscript
text width, trying the converters that are commonly available on Ubuntu in
order: cairosvg (Python), rsvg-convert (librsvg2-bin), inkscape.

    python scripts/render_network_png.py --dpi 300
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xai_gov.analysis.figures import load_network, network_diagram  # noqa: E402

TEXT_WIDTH_MM = 170.0
FONT_STACK = "Inter,DejaVu Sans,sans-serif"


def standalone_svg(svg: str) -> str:
    """Give the inline SVG an absolute size and a namespace a converter needs.

    The inline form sizes itself with CSS (width:100%), which converters
    ignore; without explicit width and height they rasterize at a default
    size and the dpi request has nothing to scale.
    """
    box = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    if box is None:
        raise ValueError("network diagram has no viewBox")
    w, h = float(box.group(1)), float(box.group(2))
    svg = svg.replace('style="width:100%;height:auto"', f'width="{w:.0f}" height="{h:.0f}"', 1)
    if "xmlns=" not in svg:
        svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    return svg.replace("Inter,system-ui,sans-serif", FONT_STACK)


def rasterize(svg_path: Path, png_path: Path, width_px: int) -> str:
    try:
        import cairosvg  # type: ignore[import-not-found]

        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=width_px)
        return "cairosvg"
    except ImportError:
        pass
    if shutil.which("rsvg-convert"):
        subprocess.run(
            ["rsvg-convert", "-w", str(width_px), "-b", "white",
             "-o", str(png_path), str(svg_path)],
            check=True,
        )
        return "rsvg-convert"
    if shutil.which("inkscape"):
        subprocess.run(
            ["inkscape", str(svg_path), "--export-type=png",
             f"--export-filename={png_path}", f"--export-width={width_px}",
             "--export-background=white"],
            check=True,
        )
        return "inkscape"
    raise SystemExit(
        "no SVG rasterizer found; install one of:\n"
        "  .venv/bin/pip install cairosvg\n"
        "  sudo apt install librsvg2-bin"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--network", default="three_echelon")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/campaigns/governance_value/figures/print/fig0-network.png",
    )
    args = parser.parse_args()

    nodes = load_network(ROOT / "configs" / "network" / f"{args.network}.yaml")
    if not nodes:
        raise SystemExit(f"no nodes in configs/network/{args.network}.yaml")
    # No title in the image: the manuscript caption carries it.
    svg = standalone_svg(network_diagram(nodes))

    width_px = round(TEXT_WIDTH_MM / 25.4 * args.dpi)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        svg_path = Path(tmp) / "network.svg"
        svg_path.write_text(svg, encoding="utf-8")
        tool = rasterize(svg_path, args.out, width_px)
    args.out.with_suffix(".svg").write_text(svg, encoding="utf-8")

    print(f"network   {args.network}  ({len(nodes)} nodes)")
    size = f"{TEXT_WIDTH_MM:.0f} mm, {width_px} px wide"
    print(f"format    png at {args.dpi} dpi ({size}) via {tool}")
    print(f"  wrote   {args.out.relative_to(ROOT)}")
    print(f"  wrote   {args.out.with_suffix('.svg').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
