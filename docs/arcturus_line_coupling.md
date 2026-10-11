# Line coupling on the Arcturus atlas

What AER's first-order (Rosenkranz) line coupling does to the Arcturus fit,
measured 2026-10-10 on every page where CO2 is fitted: 117 pages, both
epochs, 213 page-epochs. Numbers are from `docs/arcturus_line_coupling.json`,
written by `scripts/compare_arcturus_coupling.py`. The committed atlas record
is unchanged; it was fitted without coupling.

## How it was measured

Two runs of `fit_arcturus_batch.py` over the same pages, one with
`--line-coupling` and one without as the control
(`data/corrected/atlas_coupling_test/{cpl,nocpl}/`, not in git). The control
reproduces the committed full-coverage record to within 0.16% in residual on
all 213 page-epochs, so every difference below is coupling and nothing else.
(Compare against the record, not `data/corrected/atlas/summary.json`: that
summary is the superseded trimmed run, and differs by up to 125%.)

## Where it helped

Residual rms over noise, median, on the 196 page-epochs both runs fitted:

| CO2 band | page-epochs | without | with | >2% better | >2% worse |
|---|---:|---:|---:|---:|---:|
| 4.8 um (1850-2140 cm-1) | 48 | 6.04 | 6.05 | 2 | 0 |
| 4.3 um (2140-2450) | 10 | 7.50 | 7.50 | 0 | 0 |
| 2.7 um (3300-3800) | 16 | 7.38 | 7.38 | 0 | 0 |
| 2.0 um (4750-5150) | 32 | 3.39 | 3.38 | 2 | 0 |
| 1.6 um (6150-6400) | 20 | 7.02 | 7.02 | 0 | 0 |
| elsewhere | 70 | 4.00 | 4.00 | 0 | 0 |

The gains are local and sit on the CO2 Q-branches:

| page | window (cm-1) | without | with | change |
|---|---|---:|---:|---:|
| ab4975_ summer | 4972.1-5002.1 | 1.94 | 1.70 | -12.2% |
| ab4975_ winter | 4972.1-5002.1 | 2.56 | 2.27 | -11.3% |
| ab2070_ winter | 2069.0-2078.0 | 5.56 | 5.40 | -2.9% |
| ab2070_ summer | 2069.0-2078.0 | 5.17 | 5.02 | -2.8% |
| ab4950_ summer | 4947.1-4977.1 | 1.45 | 1.42 | -2.0% |
| ab4850_ winter | 4847.1-4877.1 | 4.52 | 4.44 | -1.8% |
| ab5000_ winter | 4997.1-5027.1 | 4.11 | 4.05 | -1.5% |
| ab2077_ summer | 2076.0-2085.0 | 5.55 | 5.46 | -1.5% |

ab4975_ holds the 2.0 um band centre, the same place coupling helped on IGRINS
(K89, `docs/igrins_a0v.md`); ab2070_ and ab2077_ hold the 2076.9 cm-1 Q-branch
whose artefact coupling removed from the Kitt Peak solar fit (22-31% in the
corrected spectrum there). Nowhere does a page get meaningfully worse: the
largest degradation is +1.6% (ab4950_ winter), and CO2 columns move by a median
0.0005 in log scale. Arcturus's own residuals, 4-7 sigma and dominated by the
stellar model, dilute what the telluric model gains; on the better-modelled Sun
the same physics moved residuals by 25-35%.

## Where it failed

17 page-epochs could not be fitted with coupling. The precomputed-opacity
backends (`ArrayOpacityBackend`, `LinearizedOpacityBackend` in `model.py`)
refuse a negative cross section, and first-order coupling produces them: the
dispersion term `Y p Im w` is negative on one side of every coupled line, and
far from a band's lines nothing outweighs it. Measured on each failed window,
Kitt Peak 1994 profile, vertical optical depth:

| page | window (cm-1) | species | min, uncoupled | min, coupled | most negative layer |
|---|---|---|---:|---:|---:|
| ab2400_ (both) | 2398.5-2416.5 | CO2 | +0.084 | **-0.236** | -0.030 |
| ab3135_ (both) | 3133.5-3156.0 | CH4 | +7.4e-4 | +1.8e-4 | -7.1e-4 |
| ab3150_ (both) | 3148.5-3171.0 | CH4 | +4.0e-4 | +8.8e-5 | -8.0e-5 |
| ab3165_ (both) | 3163.5-3186.0 | CH4 | +8.0e-5 | +1.4e-4 | -4.5e-6 |
| ab3180_ (both) | 3178.5-3201.0 | CH4 | +3.2e-5 | +4.0e-6 | -1.2e-6 |
| ab3195_ (both) | 3193.5-3216.0 | CH4 | +1.1e-5 | -1.3e-5 | -2.2e-6 |
| ab3210_ (both) | 3208.5-3231.0 | CH4 | +3.4e-6 | -1.4e-5 | -2.4e-6 |
| ab8300_ (both) | 8297.5-8327.5 | CO2 | +3.2e-6 | -3.5e-7 | -6.3e-8 |
| ab1951_ winter | 1950.0-1959.0 | CO2 | +1.5e-4 | +4.2e-5 | -9.0e-8 |

Two different things:

- **ab2400_ is a real failure of the physics.** Just past the 4.3 um CO2 band,
  the strongest in the atlas, the coupled optical depth goes to -0.24 where the
  uncoupled is +0.08. In the far wing of so strong a band the first-order
  dispersion terms within the 25 cm-1 cutoff do not cancel; LBLRTM pairs its
  coupling there with a CO2 continuum, carrying sub-Lorentzian wing factors,
  that was defined together with it. Coupling must stay off here until it is
  compared with LBLRTM on this window.
- **The rest are rounding-level.** In the 3.3 um CH4 band and at 8300 and
  1951 cm-1 the negatives are 1e-8 to 7e-4 per layer, between lines where the
  total is 1e-5 or less. They would fit if the precompute tolerated them (clip
  each layer at zero, or accept a negative layer when the vertical total is
  positive). The CH4 windows deserve one check against LBLRTM first: coupling
  removes up to 87% of their between-line absorption (ab3135_: 7.4e-4 to
  1.8e-4).

ab1951_ failed in winter only, in the first precompute, which is built from
the profile alone and so is the same calculation in both epochs; the summer
fit passed it. At 1e-7, whether a window trips the check is rounding
(`mixed_precision` evaluates wings in float32).

## Cost

Measured on the 196 page-epochs both runs fitted, one GPU each:

| stage | without (s) | with (s) | ratio |
|---|---:|---:|---:|
| precompute (opacity at the profile) | 942 | 1571 | 1.67 |
| products (opacity refrozen at the fit) | 35 | 96 | 2.72 |
| compile objective | 753 | 742 | 0.99 |
| fit stages | 2503 | 2490 | 0.99 |
| everything | 4746 | 5408 | 1.14 |

+14% wall time in total, +8% median per page, all of it in the two places
the line-by-line kernel runs: the precompute and the final refreeze. The fit
itself costs nothing more, because it iterates on precomputed arrays whether
coupling is on or off. A page with no coupled lines in its window costs
nothing extra (`direct_sparse` turns coupling off when the database carries no
coefficients).

## As a per-page option

Coupling is decided per page already in the sense that matters for cost; what
remains is deciding where it is *allowed*. A page-level rule, recorded per row
in the record, would let one atlas run use it where it helps and nowhere else:

1. Fit with coupling on.
2. If the coupled precompute has a negative vertical optical depth beyond a
   tolerance (ab2400_'s -0.24, not ab8300_'s -3.5e-7), refit that page without
   it; clip rounding-level negative layers at zero instead of refusing them.
3. Record `line_coupling` per row (`write_record(extra_columns=...)`), since a
   run-level flag would then be wrong for some rows, and have
   `rebuild_arcturus_page.py` and the exporters read it per row.

Not implemented. The improvement on Arcturus is too small to justify a new
atlas run on its own; this belongs with the next physics change that does.
