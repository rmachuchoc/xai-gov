#!/usr/bin/env python
"""Apply the MDPI corrections to the SVG network diagram as well.

The PNG figures go through render_figures_mdpi.py. The network diagram also
exists as an SVG built in src/xai_gov/analysis/figures.py, and that copy must
say the same thing, so this patch capitalizes its labels, removes the
"inv ... lead time" key, and updates the two tests that read those labels.
Idempotent: running it twice changes nothing the second time.

    python scripts/patch_network_svg.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "src/xai_gov/analysis/figures.py"
TEST = ROOT / "tests/unit/test_network_diagram.py"


def patch(path: Path, edits: list[tuple[str, str]]) -> int:
    text = path.read_text(encoding="utf-8")
    changed = 0
    for old, new in edits:
        if old in text:
            text = text.replace(old, new)
            changed += 1
    path.write_text(text, encoding="utf-8")
    return changed


LABELS = ("disruptions", "material", "orders", "demand")
fig_edits = [(f'"{w}"', f'"{w.capitalize()}"') for w in LABELS]
n = patch(FIG, fig_edits)

# Remove the inv/cap/s/L key: the caption carries it now.
src = FIG.read_text(encoding="utf-8")
key = re.compile(
    r"\n[ \t]*parts\.append\(\n[ \t]*_text\(\n[ \t]*left \+ 200, legend,\n"
    r"[ \t]*\"inv initial inventory[^\"]*\",\n[ \t]*size=10, fill=FAINT,\n[ \t]*\)\n[ \t]*\)"
)
src, removed = key.subn("", src)
FIG.write_text(src, encoding="utf-8")

t = patch(TEST, [
    (
        'assert "material" in svg and "orders" in svg',
        'assert "Material" in svg and "Orders" in svg',
    ),
    ('assert "demand" in svg', 'assert "Demand" in svg'),
    ('assert "disruptions" in svg', 'assert "Disruptions" in svg'),
])
print(f"figures.py: {n} labels capitalized, {removed} key removed; tests: {t} assertions updated")
