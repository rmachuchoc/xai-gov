#!/usr/bin/env python
"""Second round of overlap fixes for the network diagram (fig0).

* The "Disruptions" label sat 2 px from the top of the canvas and was cropped:
  the top margin is raised so the label and its arrow fit inside the image.
* The lead-time labels ("L 3+-0.1") were wider than the 26 px gap between
  cells and ran into the box borders: the gap is widened so each label sits
  entirely between the two cells it connects.

Geometry only; no parameter or label text changes. Idempotent.

    python scripts/patch_network_layout.py
"""

from __future__ import annotations

from pathlib import Path

FIG = Path(__file__).resolve().parents[1] / "src/xai_gov/analysis/figures.py"

EDITS = [
    ("top = 52.0 if title else 34.0", "top = 70.0 if title else 52.0"),
    ("cell_h, gap = 76.0, 26.0", "cell_h, gap = 76.0, 60.0"),
]

text = FIG.read_text(encoding="utf-8")
done = 0
for old, new in EDITS:
    if old in text:
        text = text.replace(old, new)
        done += 1
    elif new in text:
        done += 1  # already applied
FIG.write_text(text, encoding="utf-8")
print(f"network layout: {done}/{len(EDITS)} edits in place")
