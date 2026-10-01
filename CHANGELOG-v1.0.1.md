# v1.0.1

Corrects the two-sided e-value used for RQ8.

The previous construction bet on the magnitudes |d| of the paired differences.
That is not a valid test of "no difference": under the null E|d| > 0 whenever d
has any spread, so the bet accumulates evidence where there is no effect. A
new test shows it rejects a true null in most of 200 noise streams.

The corrected two-sided e-value pools a bet in each direction on the signed
differences, e = (e_up + e_down) / 2, which is valid under the point null
without correction.

Effect on the study: only RQ8's reported e-value changes, by about a factor of
two. Its verdict, raw difference, standardized effect and confidence sequence
are unchanged, and no other hypothesis uses the two-sided test.

Version numbers in pyproject.toml and xai_gov/__init__.py are aligned with the
release tag (they read 0.2.0 in v1.0.0).
