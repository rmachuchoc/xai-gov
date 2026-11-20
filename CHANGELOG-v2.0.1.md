# v2.0.1

This release carries the figures printed in the revised manuscript. The
simulator, the configurations, the sealed plan, the runs, the campaign reports
and the CSV table behind every figure are identical to those of v2.0.0, so
every reported number is the same in both releases.

## Changed

- **Resolution.** Every figure is written as an opaque RGB PNG at 1200 dpi at
  its printed width. MDPI asks for at least 600 dpi; the higher value keeps
  thin strokes and 7-8 pt text sharp when the PDF is zoomed.
- **Intervals.** Means and intervals are drawn as capped whiskers, |–●–|. An
  interval whose caps would touch its dot is left out, and the dot stands for
  it, as the captions state.
- **Evidence figure.** Labels read treatment − control, the axis spans −1 to 1
  with a cue for each sign, and the e-values sit in their own column headed by
  the support threshold of 240.
- **Envelope figure.** The axis spans 0 to 0.72, which shows the
  Clopper–Pearson intervals that lie near each bound. Only the dark bound line
  marks the area next to zero.
- **Network figure.** Each dashed arrow ends in a solid head.
- **Power figure.** The plateaus at 0 and 1 sit clear of the axis lines.
- **Gridlines.** Figures that print their values beside each point carry
  horizontal gridlines only.
- **Title.** The README and the Zenodo record use the revised title of the
  manuscript, which spells out every acronym.

## Unchanged

The engine, `configs/`, `specs/v2/confirmatory_v2.seal.json`, the 12,400 runs,
the campaign reports, the sweeps, the calibration results and the CSV tables.
The tests (844) pass.
